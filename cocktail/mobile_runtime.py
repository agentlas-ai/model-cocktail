"""ModelRuntime adapter for a locally implemented mobile host bridge."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
import math
import re
from threading import Lock
from typing import Any

from .models import ModelRuntimeError


BRIDGE_VERSION = 1
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_MAX_TEXT_BYTES = 16 * 1024
_MAX_IMAGE_BYTES = 4 * 1024 * 1024
_MAX_EMBED_BATCH = 32
_MAX_EMBED_DIMENSIONS = 4096


def _activation_policy() -> dict[str, int | bool]:
    return {"max_resident_models": 1, "release_after_call": True}


class MobileRuntime:
    """Adapt a synchronous, in-process host that owns local inference.

    Each request includes `version` and each reply echoes it. The bridge receives
    `generate`, `embed`, `models`, or `resident`. Successful replies supply
    `text`, `embeddings`, or `models`, respectively. Refusals can supply an
    `error` field. The host resolves model IDs to installed weights, enforces
    the schema, honors the activation policy, and checks token budgets after
    encoding templates and images.
    Image references are scoped to one call and carry raw bytes with a MIME type.
    The host must consume those bytes locally and bound image decoding resources.
    The bridge is serial per runtime instance and must not call the adapter back.
    """

    def __init__(
        self,
        call: Callable[[str, dict[str, Any]], dict[str, Any]],
        *,
        context_tokens: int = 2048,
    ) -> None:
        if not callable(call):
            raise ValueError("A local bridge callback is required")
        if type(context_tokens) is not int or not 256 <= context_tokens <= 4096:
            raise ValueError("Mobile context_tokens must be between 256 and 4096")
        self._call = call
        self._context_tokens = context_tokens
        self._lock = Lock()

    def _invoke(self, operation: str, **payload: Any) -> dict[str, Any]:
        try:
            with self._lock:
                response = self._call(operation, {"version": BRIDGE_VERSION, **payload})
        except Exception as error:
            raise ModelRuntimeError(f"Mobile bridge {operation} failed ({type(error).__name__})") from error
        if (
            not isinstance(response, dict)
            or type(response.get("version")) is not int
            or response["version"] != BRIDGE_VERSION
        ):
            raise ModelRuntimeError(f"Mobile bridge {operation} returned an unsupported response version")
        if "error" in response:
            raise ModelRuntimeError(f"Mobile bridge rejected {operation}")
        return response

    def generate(
        self,
        model: str,
        system: str,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        image: bytes | None = None,
        max_tokens: int = 512,
    ) -> str:
        if not _valid_model_id(model):
            raise ValueError("Mobile inference requires a local model ID")
        if type(max_tokens) is not int or not 0 < max_tokens < self._context_tokens:
            raise ValueError("Generation must reserve tokens within the mobile context budget")
        if _text_size(system) + _text_size(prompt) > _MAX_TEXT_BYTES or not prompt.strip():
            raise ValueError("Mobile input requires a prompt and at most 16 KiB of text")
        payload: dict[str, Any] = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "context_tokens": self._context_tokens,
            "max_tokens": max_tokens,
            "truncate": False,
            "activation": _activation_policy(),
        }
        if schema is not None:
            if not isinstance(schema, dict):
                raise ValueError("Mobile generation schema must be a JSON object")
            try:
                encoded = json.dumps(schema, allow_nan=False)
            except (TypeError, ValueError, RecursionError) as error:
                raise ValueError("Mobile generation schema must be valid JSON") from error
            if len(encoded) > _MAX_TEXT_BYTES:
                raise ValueError("Mobile generation schema must be at most 16 KiB")
            payload["schema"] = json.loads(encoded)
        if image is not None:
            payload["image"] = _image_artifact(image)
        output = self._invoke("generate", **payload).get("text")
        try:
            if _text_size(output) > _MAX_TEXT_BYTES or not output.strip():
                raise ValueError("Empty or oversized output")
            if schema is not None:
                json.loads(output, parse_constant=_reject_nonfinite_json)
        except (ValueError, RecursionError) as error:
            raise ModelRuntimeError("Mobile generation returned invalid text") from error
        return output.strip()

    def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        if not _valid_model_id(model):
            raise ValueError("Mobile inference requires a local model ID")
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes, bytearray)):
            raise ValueError("Mobile embedding input must be a sequence of text items")
        if len(texts) > _MAX_EMBED_BATCH or sum(map(_text_size, texts)) > _MAX_TEXT_BYTES:
            raise ValueError("Mobile embeddings accept at most 32 items and 16 KiB of text")
        if not texts:
            return []
        result = self._invoke(
            "embed", model=model, texts=list(texts),
            context_tokens=self._context_tokens, truncate=False, activation=_activation_policy(),
        ).get("embeddings")
        if not isinstance(result, list) or len(result) != len(texts):
            raise ModelRuntimeError("Mobile embedding count does not match the request")
        dimension = len(result[0]) if isinstance(result[0], list) else 0
        if not 1 <= dimension <= _MAX_EMBED_DIMENSIONS:
            raise ModelRuntimeError("Mobile embedding dimensions are invalid")
        try:
            valid = all(
                isinstance(row, list) and len(row) == dimension
                and all(type(value) in (int, float) and math.isfinite(value) for value in row)
                for row in result
            )
        except OverflowError:
            valid = False
        if not valid:
            raise ModelRuntimeError("Mobile embeddings must be finite and have matching dimensions")
        return [[float(value) for value in row] for row in result]

    def models(self) -> dict[str, dict[str, Any]]:
        return {item["model"]: item for item in self._model_rows("models")}

    def resident(self) -> list[dict[str, Any]]:
        return self._model_rows("resident")

    def _model_rows(self, operation: str) -> list[dict[str, Any]]:
        models = self._invoke(operation).get("models")
        if not isinstance(models, list) or len(models) > 128:
            raise ModelRuntimeError(f"Mobile bridge {operation} returned an invalid model list")
        rows = {}
        for item in models:
            if not isinstance(item, dict) or not _valid_model_id(item.get("model")):
                raise ModelRuntimeError(f"Mobile bridge {operation} returned an invalid model ID")
            model = item["model"]
            digest = item.get("digest")
            if model in rows or (digest is not None and not _valid_model_id(digest)):
                raise ModelRuntimeError(f"Mobile bridge {operation} returned invalid model identity")
            row = {"name": model, "model": model}
            if digest is not None:
                row["digest"] = digest
            for field in ("size", "size_vram", "context_length"):
                if field in item:
                    if type(item[field]) is not int or item[field] < 0:
                        raise ModelRuntimeError(f"Mobile bridge {operation} returned invalid {field}")
                    row[field] = item[field]
            rows[model] = row
        return list(rows.values())


def _valid_model_id(value: Any) -> bool:
    return (isinstance(value, str) and _MODEL_ID.fullmatch(value) is not None
            and not value.lower().endswith((":cloud", "-cloud")))


def _text_size(value: Any) -> int:
    if not isinstance(value, str):
        raise ValueError("Mobile text fields must be UTF-8 strings")
    return len(value.encode("utf-8"))


def _reject_nonfinite_json(_: str) -> None:
    raise ValueError("Mobile JSON cannot contain nonfinite numbers")


def _image_artifact(image: bytes) -> dict[str, Any]:
    if not isinstance(image, bytes) or not image or len(image) > _MAX_IMAGE_BYTES:
        raise ValueError("Mobile image must contain 1 byte to 4 MiB of encoded data")
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif image.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    else:
        raise ValueError("Mobile image must be a PNG or JPEG")
    return {"ref": "input-image", "mime": mime, "bytes": image}
