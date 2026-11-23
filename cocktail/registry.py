"""Versioned capability slots and independently replaceable providers."""

from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class Provider:
    backend: str
    model: str | None = None


MODEL_SLOTS = {
    "decision": "text", "conversation": "text", "reasoning": "text",
    "vision": "vision", "embedding": "embedding",
}
ENGINE_SLOTS = {
    "ocr": "native_ocr", "memory": "sqlite_memory", "action": "local_action",
    "document": "markdown", "math": "calculator",
}

DEFAULT_MODELS = {
    "decision": ["qwen3:1.7b", "qwen3:4b"],
    "conversation": ["qwen3:4b", "qwen3:1.7b"],
    "reasoning": ["qwen3:4b"],
    "vision": ["qwen3-vl:2b"],
    "embedding": ["qwen3-embedding:0.6b"],
}


@dataclass(frozen=True)
class Registry:
    slots: dict[str, tuple[Provider, ...]]

    @property
    def models(self) -> dict[str, tuple[str, ...]]:
        model_names = {
            part.model for providers in self.slots.values() for part in providers if part.model is not None
        }
        return {
            name: tuple(slot for slot, providers in self.slots.items() if any(p.model == name for p in providers))
            for name in sorted(model_names)
        }

    def candidates(self, capability: str) -> tuple[Provider, ...]:
        return self.slots.get(capability, ())

    def replace(self, capability: str, model: str) -> "Registry":
        if capability not in MODEL_SLOTS:
            raise ValueError(f"Model replacement is unavailable for {capability!r}")
        candidates = (Provider(MODEL_SLOTS[capability], checked_model(model)),)
        return Registry({**self.slots, capability: candidates})


def checked_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A model entry must contain a local model tag")
    return model.strip()


def load_registry(config: Path | None = None, *, baseline: str | None = None) -> Registry:
    models = dict(DEFAULT_MODELS)
    engines = {slot: [backend] for slot, backend in ENGINE_SLOTS.items()}
    if config is not None:
        data = json.loads(config.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or not set(data) <= {"version", "models", "engines"}
                or data.get("version") != 1 or not isinstance(data.get("models", {}), dict)
                or not isinstance(data.get("engines", {}), dict)):
            raise ValueError("Expected a version 1 model registry")
        for capability, candidates in data.get("models", {}).items():
            if capability not in MODEL_SLOTS or not isinstance(candidates, list):
                raise ValueError(f"Invalid model registry slot {capability!r}")
            tags = [checked_model(model) for model in candidates]
            if len(set(tags)) != len(tags):
                raise ValueError(f"Duplicate provider in {capability!r}")
            models[capability] = tags
        for capability, candidates in data.get("engines", {}).items():
            if (capability not in ENGINE_SLOTS or not isinstance(candidates, list)
                    or any(not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                           for name in candidates)):
                raise ValueError(f"Invalid engine registry slot {capability!r}")
            if capability == "action" and candidates not in ([], ["local_action"]):
                raise ValueError("The action slot requires the approved local action port")
            engines[capability] = candidates
    if baseline is not None:
        for capability in ("decision", "conversation", "reasoning", "vision"):
            models[capability] = [checked_model(baseline)]
    slots = {
        capability: tuple(Provider(MODEL_SLOTS[capability], item) for item in candidates)
        for capability, candidates in models.items()
    }
    slots.update({slot: tuple(Provider(name) for name in names) for slot, names in engines.items()})
    if not slots["decision"]:
        raise ValueError("Register at least one decision model")
    return Registry(slots)
