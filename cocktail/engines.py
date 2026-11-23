"""Bounded, local deterministic engines for the prototype."""

import ast
from contextlib import closing
from datetime import date
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from heapq import nsmallest
import json
import math
import operator
import os
from pathlib import Path
import re
import sqlite3
from tempfile import NamedTemporaryFile
from uuid import uuid4


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_MAX_EXPRESSION_LENGTH = 256
_MAX_EXPRESSION_NODES = 64
_MAX_RESULT_BITS = 1024
_MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
_ARTIFACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def calculate(expression: str) -> str:
    """Evaluate arithmetic and return an exact decimal or reduced fraction."""
    if not isinstance(expression, str) or len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression must be text no longer than 256 characters")
    expression = expression.strip()
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid arithmetic expression") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_EXPRESSION_NODES:
        raise ValueError("Expression has too many operations")
    try:
        value = _arithmetic(tree.body, expression)
    except ZeroDivisionError as exc:
        raise ValueError("Division by zero") from exc
    return _format_fraction(value)


def _arithmetic(node: ast.expr, source: str) -> Fraction:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        value = Fraction(node.value)
    elif isinstance(node, ast.Constant) and type(node.value) is float:
        literal = ast.get_source_segment(source, node)
        try:
            decimal = Decimal(literal)
        except InvalidOperation as exc:
            raise ValueError("Invalid decimal literal") from exc
        if abs(decimal.as_tuple().exponent) > 128:
            raise ValueError("Decimal exponent exceeds the supported range")
        value = Fraction(decimal)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        value = _arithmetic(node.operand, source)
        if isinstance(node.op, ast.USub):
            value = -value
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _arithmetic(node.left, source)
        right = _arithmetic(node.right, source)
        if isinstance(node.op, ast.Pow) and (right.denominator != 1 or abs(right) > 16):
            raise ValueError("Power must be an integer from -16 to 16")
        value = Fraction(_BINARY_OPERATORS[type(node.op)](left, right))
    else:
        raise ValueError("Only numeric literals and arithmetic operators are supported")
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > _MAX_RESULT_BITS:
        raise ValueError("Arithmetic result exceeds the supported range")
    return value


def _format_fraction(value: Fraction) -> str:
    denominator = value.denominator
    powers = []
    for prime in (2, 5):
        count = 0
        while denominator % prime == 0:
            denominator //= prime
            count += 1
        powers.append(count)
    if denominator != 1:
        return str(value)
    places = max(powers)
    if places == 0:
        return str(value.numerator)
    digits = str(abs(value.numerator) * 10**places // value.denominator)
    digits = digits.zfill(places + 1)
    sign = "-" if value < 0 else ""
    return f"{sign}{digits[:-places]}.{digits[-places:]}"


def read_document(path: Path, mime: str) -> str:
    """Read regular UTF-8 text or Markdown files up to two MiB."""
    if not isinstance(mime, str) or mime.partition(";")[0].strip().lower() not in {
        "text/plain", "text/markdown", "text/x-markdown",
    }:
        raise ValueError(f"Unsupported document MIME type: {mime!r}. Expected UTF-8 text or Markdown.")
    if not path.is_file():
        raise ValueError(f"Document is not a regular file: {path.name}")
    with path.open("rb") as source:
        data = source.read(_MAX_DOCUMENT_BYTES + 1)
    if len(data) > _MAX_DOCUMENT_BYTES:
        raise ValueError("Document exceeds 2 MiB")
    if b"\0" in data:
        raise ValueError("Document contains binary null bytes")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Document is not valid UTF-8") from exc


def write_document(output_dir: Path, title: str, text: str, *, artifact_id: str) -> Path:
    """Atomically write an artifact inside the caller's task-scoped directory."""
    if not isinstance(artifact_id, str) or not _ARTIFACT_ID.fullmatch(artifact_id):
        raise ValueError("Artifact ID must be a bounded filename-safe identifier")
    title = _single_line_title(title)
    if not isinstance(text, str):
        raise ValueError("Document body must be text")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"{artifact_id}.md"
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, delete=False) as staged:
            temporary = Path(staged.name)
            staged.write(f"# {title}\n\n{text}")
            if not text.endswith("\n"):
                staged.write("\n")
            staged.flush()
            os.fsync(staged.fileno())
        os.replace(temporary, artifact)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return artifact


def _normalized_vector(vector: list[float]) -> list[float]:
    if not isinstance(vector, list) or not 1 <= len(vector) <= 8192:
        raise ValueError("Embedding must contain between 1 and 8192 coordinates")
    try:
        if any(type(value) not in (float, int) or not math.isfinite(value) for value in vector):
            raise ValueError("Embedding coordinates must be finite numbers")
    except OverflowError as exc:
        raise ValueError("Embedding coordinates exceed the supported range") from exc
    max_component = max(abs(value) for value in vector)
    if max_component == 0:
        raise ValueError("Embedding must have nonzero magnitude")
    scaled = [value / max_component for value in vector]
    norm = math.hypot(*scaled)
    return [value / norm for value in scaled]


class MemoryStore:
    """Store normalized embeddings by key, model digest, and dimension."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    key TEXT NOT NULL,
                    embedding_id TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    PRIMARY KEY (embedding_id, dimensions, key)
                )
            """)

    def put(self, key: str, text: str, vector: list[float], embedding_id: str) -> None:
        _identifier(key, "Memory key")
        _identifier(embedding_id, "Embedding ID")
        if not isinstance(text, str):
            raise ValueError("Memory content must be text")
        vector = _normalized_vector(vector)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("""
                INSERT INTO memory_vectors (key, embedding_id, dimensions, text, vector)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (embedding_id, dimensions, key) DO UPDATE SET
                    text = excluded.text, vector = excluded.vector
            """, (key, embedding_id, len(vector), text, json.dumps(vector)))

    def search(self, vector: list[float], embedding_id: str, limit: int = 3) -> list[dict]:
        _identifier(embedding_id, "Embedding ID")
        if type(limit) is not int or limit < 0:
            raise ValueError("Search limit must be a nonnegative integer")
        vector = _normalized_vector(vector)
        if limit == 0:
            return []
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute("""
                SELECT key, text, vector FROM memory_vectors
                WHERE embedding_id = ? AND dimensions = ?
            """, (embedding_id, len(vector)))
            results = (
                {"key": key, "text": text, "score": _cosine(vector, json.loads(stored_vector))}
                for key, text, stored_vector in rows
            )
            return nsmallest(limit, results, key=lambda row: (-row["score"], row["key"]))


def _cosine(left: list[float], right: list[float]) -> float:
    dot_product = math.fsum(a * b for a, b in zip(left, right, strict=True))
    return max(-1.0, min(1.0, dot_product))


def _identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"{label} must be nonempty text")


def _single_line_title(title: str) -> str:
    if not isinstance(title, str):
        raise ValueError("Title must be text")
    title = title.strip()
    if not 1 <= len(title) <= 160 or not title.isprintable():
        raise ValueError("Title must be a single line containing 1 to 160 printable characters")
    return title


def preview_action(action: str, args: dict) -> dict:
    """Validate and normalize a proposed mock action without writing state."""
    if action != "calendar.create_mock":
        raise ValueError(f"Unsupported action: {action!r}")
    if not isinstance(args, dict) or args.keys() != {"title", "date"}:
        raise ValueError("Mock calendar creation requires exactly title and date")
    day = args["date"]
    if not isinstance(day, str) or not _ISO_DATE.fullmatch(day):
        raise ValueError("Date must be an ISO YYYY-MM-DD date")
    return {
        "action": action,
        "args": {"title": _single_line_title(args["title"]), "date": date.fromisoformat(day).isoformat()},
    }


class ActionStore:
    """Execute authorized mock actions exactly once per idempotency key."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS mock_calendar_events (
                    idempotency_key TEXT PRIMARY KEY NOT NULL,
                    event_id TEXT NOT NULL UNIQUE,
                    action TEXT NOT NULL,
                    args TEXT NOT NULL
                )
            """)

    def execute(self, action: str, args: dict, idempotency_key: str) -> dict:
        _identifier(idempotency_key, "Idempotency key")
        normalized = preview_action(action, args)
        payload = json.dumps(normalized["args"], sort_keys=True, ensure_ascii=False)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("""
                SELECT event_id, action, args FROM mock_calendar_events WHERE idempotency_key = ?
            """, (idempotency_key,)).fetchone()
            if row is None:
                event_id = uuid4().hex
                connection.execute("""
                    INSERT INTO mock_calendar_events (idempotency_key, event_id, action, args)
                    VALUES (?, ?, ?, ?)
                """, (idempotency_key, event_id, normalized["action"], payload))
            else:
                event_id, prior_action, prior_payload = row
                if (prior_action, prior_payload) != (normalized["action"], payload):
                    raise ValueError("Idempotency key already belongs to a different action")
        return {**normalized, "status": "executed", "event_id": event_id, "idempotency_key": idempotency_key}
