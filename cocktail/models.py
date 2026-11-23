"""Runtime adapters for local generation, embeddings, and image OCR."""

from __future__ import annotations

import base64
import http.client
import ipaddress
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Protocol, Sequence
from urllib.parse import urlsplit


class ModelRuntimeError(RuntimeError):
    """A configured model runtime could not complete a local request."""


class OCRUnavailableError(ModelRuntimeError):
    """The native OCR adapter is unavailable or did not return text."""


class ModelRuntime(Protocol):
    def generate(
        self,
        model: str,
        system: str,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        image: bytes | None = None,
        max_tokens: int = 512,
    ) -> str: ...

    def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]: ...

    def models(self) -> dict[str, dict[str, Any]]: ...

    def resident(self) -> list[dict[str, Any]]: ...


class OllamaRuntime:
    """Use an existing loopback Ollama server and installed local models."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
        timeout: float = 90.0,
        keep_alive: int | str = 0,
    ) -> None:
        url = urlsplit(base_url)
        if (
            url.scheme != "http"
            or not url.hostname
            or url.username is not None
            or url.path not in ("", "/")
            or url.query
            or url.fragment
        ):
            raise ValueError("Ollama URL must be a loopback HTTP origin")
        host = "127.0.0.1" if url.hostname == "localhost" else url.hostname
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError("Ollama URL must use a loopback address")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Ollama timeout must be finite and positive")
        self._host = host
        self._port = 11434 if url.port is None else url.port
        self._timeout = timeout
        self._keep_alive = keep_alive

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        # A direct connection avoids environment proxies and automatic redirects.
        connection = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        try:
            if payload is None:
                connection.request("GET", path)
            else:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                connection.request("POST", path, body, {"Content-Type": "application/json"})
            with connection.getresponse() as response:
                if response.status != 200:
                    raise ModelRuntimeError(f"Ollama {path} returned HTTP {response.status}")
                result = json.loads(response.read())
        except (OSError, http.client.HTTPException, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelRuntimeError(f"Ollama {path} request failed ({type(error).__name__})") from error
        finally:
            connection.close()
        if not isinstance(result, dict) or "error" in result:
            raise ModelRuntimeError(f"Ollama {path} returned an invalid response")
        return result

    def _model_list(self, path: str) -> list[dict[str, Any]]:
        models = self._request(path).get("models")
        if models is None:
            return []
        if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
            raise ModelRuntimeError(f"Ollama {path} returned an invalid model list")
        return models

    def models(self) -> dict[str, dict[str, Any]]:
        installed = {}
        for item in self._model_list("/api/tags"):
            name = item.get("name") or item.get("model")
            if not isinstance(name, str) or not name:
                raise ModelRuntimeError("Ollama returned an installed model without a name")
            installed[name] = item
        return installed

    def resident(self) -> list[dict[str, Any]]:
        return self._model_list("/api/ps")

    def _require_local_model(self, model: str) -> dict[str, Any]:
        reference = model.rpartition("/")[2].lower()
        if "://" in model or reference.endswith((":cloud", "-cloud")):
            raise ModelRuntimeError(f"Cloud model is unavailable in the local runtime: {model}")
        name = model if ":" in reference else f"{model}:latest"
        installed = self.models()
        entry = installed.get(model) or installed.get(name)
        if entry is None:
            raise ModelRuntimeError(f"Model is not installed in local Ollama: {model}")
        if entry.get("remote_host") or entry.get("remote_model"):
            raise ModelRuntimeError(f"Cloud model is unavailable in the local runtime: {model}")
        return entry

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
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        entry = self._require_local_model(model)
        user_message: dict[str, Any] = {"role": "user", "content": prompt}
        if image is not None:
            user_message["images"] = [base64.b64encode(image).decode("ascii")]
        messages = [{"role": "system", "content": system}] if system else []
        messages.append(user_message)
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096},
        }
        if schema is not None:
            request["format"] = schema
        family = (entry.get("details") or {}).get("family", "")
        reference = model.rpartition("/")[2]
        if family.startswith("qwen3") or reference.startswith("qwen3"):
            request["think"] = False
        result = self._request("/api/chat", request)
        message = result.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if schema is None and isinstance(content, str) and family.startswith("qwen3"):
            if "</think>" in content:
                content = content.rsplit("</think>", 1)[1]
        if not isinstance(content, str) or not content.strip():
            reason = " (output token limit)" if result.get("done_reason") == "length" else ""
            raise ModelRuntimeError(f"Ollama model returned no answer: {model}{reason}")
        return content.strip()

    def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._require_local_model(model)
        result = self._request(
            "/api/embed",
            {"model": model, "input": list(texts), "truncate": False, "keep_alive": self._keep_alive},
        )
        vectors = result.get("embeddings")
        if (
            not isinstance(vectors, list)
            or len(vectors) != len(texts)
            or any(not isinstance(vector, list) or not vector for vector in vectors)
        ):
            raise ModelRuntimeError(f"Ollama returned invalid embedding rows: {model}")
        dimension = len(vectors[0])
        if any(
            len(vector) != dimension
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise ModelRuntimeError(f"Ollama returned invalid embedding values: {model}")
        return [[float(value) for value in vector] for vector in vectors]


def recognize_image(path: Path, *, native_binary: Path | None = None) -> str:
    """Read text from an image with Apple Vision through a local subprocess."""
    if sys.platform != "darwin":
        raise OCRUnavailableError("Apple Vision OCR requires macOS")
    if native_binary is None:
        swift = shutil.which("swift")
        if swift is None:
            raise OCRUnavailableError("Apple Vision OCR requires Swift or a compiled OCR binary")
        command = [swift, str(Path(__file__).with_name("native_ocr.swift").resolve())]
    else:
        command = [str(native_binary.resolve())]
    try:
        result = subprocess.run(
            [*command, str(path.resolve())],
            capture_output=True,
            encoding="utf-8",
            timeout=45,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise OCRUnavailableError(f"Apple Vision OCR could not run ({type(error).__name__})") from error
    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip().splitlines()
        raise OCRUnavailableError(detail[-1] if detail else "Apple Vision OCR found no text")
    return result.stdout.strip()
