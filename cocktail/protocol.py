"""JSON-facing task, call, artifact, result, and event contracts."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4


PROTOCOL_VERSION = 1
ATTACHMENTS = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".txt": "text/plain", ".md": "text/markdown", ".wav": "audio/wav",
    ".mp3": "audio/mpeg", ".pdf": "application/pdf",
}
MAX_ATTACHMENT = 12 * 1024 * 1024
MAX_REQUEST = 16000


@dataclass(frozen=True)
class ArtifactRef:
    id: str
    mime: str
    label: str
    path: str

    def descriptor(self) -> dict:
        return {"id": self.id, "mime": self.mime, "label": self.label}


@dataclass(frozen=True)
class CapabilityInput:
    capability: str
    input: str
    ref: str
    title: str
    date: str


@dataclass(frozen=True)
class Result:
    capability: str
    model: str | None
    data: dict
    sources: list[str]


@dataclass
class Task:
    task_id: str
    request: str
    artifacts: list[ArtifactRef]
    version: int = PROTOCOL_VERSION
    status: str = "running"
    answer: str = ""
    results: list[Result] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    approval: dict | None = None
    revision: int = 0

    def event(self, event_type: str, **fields) -> None:
        self.events.append({"seq": len(self.events) + 1, "type": event_type, **fields})

    def snapshot(self) -> dict:
        return asdict(self)

    def result(self) -> dict:
        snapshot = self.snapshot()
        snapshot.pop("revision")
        return snapshot

    def file(self, ref: str, *, kinds: tuple[str, ...]) -> ArtifactRef:
        for artifact in self.artifacts:
            if artifact.id == ref and artifact.mime in kinds:
                return artifact
        raise ValueError("Attachment reference or type is unavailable")

    def model_context(self) -> dict:
        return {
            "request": self.request,
            "attachments": [item.descriptor() for item in self.artifacts],
            "results": [
                {"id": f"result-{index}", "capability": result.capability,
                 "data": result.data, "sources": result.sources}
                for index, result in enumerate(self.results, 1)
            ],
        }


class TaskStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.db = self.root / "tasks.sqlite3"
        with sqlite3.connect(self.db) as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS tasks "
                "(id TEXT PRIMARY KEY, revision INTEGER NOT NULL, snapshot TEXT NOT NULL)"
            )

    def begin(self, request: str, files: list[Path]) -> Task:
        if not request.strip() or len(request) > MAX_REQUEST or len(files) > 8:
            raise ValueError("A request requires 1–16000 characters and at most eight attachments")
        task_id = uuid4().hex
        task_dir = self.root / "artifacts" / task_id
        prepared = [self._read_upload(Path(file)) for file in files]
        artifacts = []
        if prepared:
            task_dir.mkdir(mode=0o700, parents=True)
        for index, (original, mime, contents) in enumerate(prepared, 1):
            ref = f"file-{index}"
            destination = task_dir / (ref + original.suffix.lower())
            with destination.open("xb") as attached:
                attached.write(contents)
            artifacts.append(ArtifactRef(ref, mime, original.name, str(destination)))
        task = Task(task_id, request.strip(), artifacts)
        task.event("task_started")
        with sqlite3.connect(self.db) as database:
            database.execute("INSERT INTO tasks VALUES (?, ?, ?)",
                             (task.task_id, task.revision, json.dumps(task.snapshot(), ensure_ascii=False)))
        return task

    @staticmethod
    def _read_upload(file: Path):
        mime = ATTACHMENTS.get(file.suffix.lower())
        if not mime or not file.is_file():
            raise ValueError(f"Unsupported or unavailable attachment {file.name!r}")
        with file.open("rb") as source:
            content = source.read(MAX_ATTACHMENT + 1)
        if not content or len(content) > MAX_ATTACHMENT:
            raise ValueError("Attachment must contain 1 byte to 12 MiB")
        return file, mime, content

    def load(self, task_id: str) -> Task:
        if not re.fullmatch(r"[a-f0-9]{32}", task_id):
            raise ValueError("Invalid task identifier")
        with sqlite3.connect(self.db) as database:
            row = database.execute("SELECT snapshot FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise ValueError("Task not found")
        snapshot = json.loads(row[0])
        if snapshot["version"] != PROTOCOL_VERSION:
            raise ValueError("Unsupported task version")
        snapshot["artifacts"] = [ArtifactRef(**item) for item in snapshot["artifacts"]]
        snapshot["results"] = [Result(**item) for item in snapshot["results"]]
        return Task(**snapshot)

    def save(self, task: Task) -> None:
        revision = task.revision + 1
        snapshot = {**task.snapshot(), "revision": revision}
        with sqlite3.connect(self.db) as database:
            update = database.execute(
                "UPDATE tasks SET revision=?, snapshot=? WHERE id=? AND revision=?",
                (revision, json.dumps(snapshot, ensure_ascii=False), task.task_id, task.revision),
            )
            if update.rowcount != 1:
                raise RuntimeError("Task was advanced by another process")
        task.revision = revision
