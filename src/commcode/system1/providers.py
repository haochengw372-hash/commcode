"""Standard-library HTTP with durable, idempotent raw-response journals."""

try:
    import fcntl
except ImportError:  # Windows import remains usable; POSIX path is the tested release platform.
    fcntl = None
    import msvcrt
import hashlib
import http.client
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from .models import ValidationError
from .parsing import load_json


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class DispatchUncertain(RuntimeError):
    """A claimed request may have been charged; it must not be automatically resent."""


class ProviderRejected(RuntimeError):
    """A provider explicitly rejected the request. Retry is an explicit new attempt."""


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def lock(self, signature):
        with (self.path / f"{signature}.lock").open("a") as stream:
            if fcntl is not None:
                fcntl.flock(stream, fcntl.LOCK_EX)
            else:
                stream.write("0")
                stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(stream, fcntl.LOCK_UN)
                else:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

    def read(self, signature):
        path = self.path / f"{signature}.json"
        if not path.exists():
            return None
        record = load_json(path.read_text(encoding="utf-8"))
        if record["signature"] != signature or digest(record["signature_payload"]) != signature:
            raise ValidationError("Journal signature corrupted")
        if (
            digest(record["request"]) != record["request_sha256"]
            or record["request"] != record["signature_payload"]["request"]
        ):
            raise ValidationError("Journal request corrupted")
        if "response" in record and digest(record["response"]) != record["response_sha256"]:
            raise ValidationError("Journal response corrupted")
        return record

    def save(self, signature, record):
        atomic_json(self.path / f"{signature}.json", record)


class Provider:
    def __init__(
        self, kind="jev", model=None, endpoint=None, api_key=None, transport=None, timeout=120
    ):
        if kind not in ("jev", "llm"):
            raise ValidationError("Provider kind must be jev or llm")
        self.kind = kind
        self.model = model or ("jev-latest" if kind == "jev" else "deepseek-flash")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
        self.endpoint = endpoint or (
            "https://api.typesafe.ai/v1/systemone" if kind == "jev" else base + "/chat/completions"
        )
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme not in ("https", "http")
            or not parsed.hostname
            or parsed.username
            or parsed.query
            or parsed.fragment
            or (
                parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1")
            )
        ):
            raise ValidationError("Endpoint must be a URL without credentials or query parameters")
        self.api_key = api_key or os.environ.get(
            "TYPESAFE_API_KEY" if kind == "jev" else "OPENAI_API_KEY"
        )
        self.transport = transport or self._http
        self.timeout = timeout
        self.local = threading.local()

    def _http(self, endpoint, body, headers):
        parsed = urlsplit(endpoint)
        origin = (parsed.scheme, parsed.hostname, parsed.port)
        if getattr(self.local, "origin", None) != origin:
            cls = (
                http.client.HTTPSConnection
                if parsed.scheme == "https"
                else http.client.HTTPConnection
            )
            self.local.connection = cls(parsed.hostname, parsed.port, timeout=self.timeout)
            self.local.origin = origin
        connection = self.local.connection
        try:
            connection.request("POST", parsed.path or "/", body=canonical(body), headers=headers)
            response = connection.getresponse()
            raw = response.read().decode("utf-8")
        except Exception:
            connection.close()
            self.local.origin = None
            raise
        # Preserve raw HTTP body even if JSON decoding fails after a paid response.
        return {"_http_status": response.status, "_http_body": raw}

    def invoke(self, body, journal, cache_epoch="v1", retry_rejected=False):
        if body.get("model") != self.model:
            raise ValidationError("Request model differs from provider model")
        payload = {
            "endpoint": self.endpoint,
            "kind": self.kind,
            "request": body,
            "cache_epoch": cache_epoch,
            "engine_version": "system1-2",
        }
        signature = digest(payload)
        with journal.lock(signature):
            prior = journal.read(signature)
            if prior is not None:
                if prior["status"] == "received":
                    return dict(prior, cache_hit=True)
                if prior["status"] != "rejected" or not retry_rejected:
                    raise DispatchUncertain("Prior dispatch is unresolved; raw journal retained")
                atomic_json(journal.path / f"{signature}.attempt{prior['attempt']}.json", prior)
            if not self.api_key and self.transport == self._http:
                raise ValidationError("Provider credential not configured")
            record = {
                "signature": signature,
                "signature_payload": payload,
                "request": body,
                "request_sha256": digest(body),
                "status": "dispatching",
                "attempt": 1 if prior is None else prior["attempt"] + 1,
                "started_at": now(),
                "cache_hit": False,
                "journal_path": str(journal.path / f"{signature}.json"),
            }
            journal.save(signature, record)  # Durable claim precedes network dispatch.
            started = time.perf_counter()
            try:
                response = self.transport(
                    self.endpoint,
                    body,
                    {
                        "Authorization": f"Bearer {self.api_key or ''}",
                        "Content-Type": "application/json",
                    },
                )
            except Exception as exc:
                record.update(
                    status="uncertain",
                    error_type=type(exc).__name__,
                    elapsed_seconds=time.perf_counter() - started,
                    finished_at=now(),
                )
                journal.save(signature, record)
                raise DispatchUncertain("Dispatch failed; no automatic resend") from exc
            record.update(
                response=response,
                response_sha256=digest(response),
                status="received",
                elapsed_seconds=time.perf_counter() - started,
                finished_at=now(),
            )
            journal.save(signature, record)  # Paid raw bytes survive all parsing failures.
            if "_http_status" in response:
                status = response["_http_status"]
                if status >= 400:
                    record["status"] = "rejected" if status == 429 else "uncertain"
                    record["http_status"] = status
                    journal.save(signature, record)
                    raise ProviderRejected(f"HTTP {status}; raw response retained")
            return record

    @staticmethod
    def response(record):
        response = record["response"]
        return load_json(response["_http_body"]) if "_http_body" in response else response
