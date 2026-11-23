"""Command line for the shared local Model Cocktail entry point."""

import argparse
import json
from pathlib import Path
import sys

from .harness import Harness
from .models import ModelRuntimeError, OllamaRuntime
from .registry import load_registry


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m cocktail")
    parser.add_argument("--state", type=Path, default=Path.home() / ".model-cocktail",
                        help="Directory for local tasks, notes, and artifacts")
    parser.add_argument("--registry", type=Path, help="Versioned JSON capability registry")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434", help="Loopback Ollama URL")
    parser.add_argument("--keep-alive", default="0", help="Optional Ollama model retention, such as 15s")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Process a user request")
    run.add_argument("request")
    run.add_argument("--file", action="append", type=Path, default=[])
    resume = commands.add_parser("resume", help="Approve the exact stored mock action and resume")
    resume.add_argument("task_id")
    resume.add_argument("--approve", required=True, metavar="TOKEN")
    recover = commands.add_parser("continue", help="Continue an interrupted running task from its last checkpoint")
    recover.add_argument("task_id")
    note = commands.add_parser("remember", help="Store an explicit local note for semantic retrieval")
    note.add_argument("key")
    note.add_argument("text")
    inspect = commands.add_parser("inspect", help="Read persisted task, provenance, and execution events")
    inspect.add_argument("task_id")
    commands.add_parser("models", help="List configured models and their installation status")
    args = parser.parse_args(argv)
    keep_alive = 0 if args.keep_alive == "0" else args.keep_alive
    try:
        runtime = OllamaRuntime(args.ollama, keep_alive=keep_alive)
        registry = load_registry(args.registry)
        harness = Harness(args.state, runtime, registry)
        if args.command == "run":
            result = harness.run(args.request, args.file)
        elif args.command == "resume":
            result = harness.resume(args.task_id, args.approve)
        elif args.command == "continue":
            result = harness.continue_task(args.task_id)
        elif args.command == "remember":
            harness.add_memory(args.key, args.text)
            result = {"status": "completed", "key": args.key}
        elif args.command == "inspect":
            result = harness.inspect(args.task_id)
        else:
            installed = runtime.models()
            result = {
                "models": {name: {"capabilities": capabilities, "installed": name in installed}
                           for name, capabilities in registry.models.items()},
                "slots": {slot: [{"backend": item.backend, "model": item.model} for item in providers]
                          for slot, providers in registry.slots.items()},
            }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return int(result.get("status") == "failed")
    except (ValueError, ModelRuntimeError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
