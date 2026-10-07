import dataclasses
import json
import mimetypes
import os
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
DEFAULT_MEDICINE_FILE = ROOT / "data" / "medicines" / "indian_pharmaceutical_products_clean.csv"
DEFAULT_DB = ROOT / "data" / "bharosa.db"


def load_dotenv():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_dotenv()

from bharosa.medicine.loader import load_medicines
from bharosa.medicine.search import search_medicine
from bharosa.index.zones import search_schemes
from bharosa.text.zones import load_zone_documents
from bharosa.rag.claimcheck import check_claim


def jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if dataclasses.is_dataclass(value):
        return {
            field.name: jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        return {
            str(k): jsonable(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }

    return str(value)


@lru_cache(maxsize=1)
def medicine_records(path_str):
    records, _stats = load_medicines(Path(path_str))
    return tuple(records)


@lru_cache(maxsize=1)
def scheme_documents(path_str):
    return tuple(load_zone_documents(Path(path_str)))


def medicine_endpoint(query, k):
    path = Path(os.environ.get("MEDICINE_FILE", str(DEFAULT_MEDICINE_FILE)))
    if not path.is_file():
        raise FileNotFoundError(f"Medicine dataset not found: {path}")

    result = search_medicine(
        query,
        records=medicine_records(str(path)),
        k=k,
    )

    payload = jsonable(result)
    candidates = payload.get("candidates", [])

    if not isinstance(candidates, list):
        candidates = []

    for item in candidates:
        if isinstance(item, dict) and not item.get("title"):
            item["title"] = item.get("brand", item.get("name", ""))

    return {
        "query": query,
        "candidates": candidates,
    }


def scheme_endpoint(query, k):
    db_path = Path(os.environ.get("BHAROSA_DB", str(DEFAULT_DB)))
    if not db_path.is_file():
        raise FileNotFoundError(f"Bharosa database not found: {db_path}")

    hits = search_schemes(
        query,
        k=k,
        documents=scheme_documents(str(db_path)),
    )

    serialized = jsonable(hits)

    if not isinstance(serialized, list):
        serialized = []

    for item in serialized:
        if isinstance(item, dict) and not item.get("title"):
            item["title"] = item.get("doc_id", "")

    return {
        "query": query,
        "hits": serialized,
    }


def claim_endpoint(message, k):
    db_path = Path(os.environ.get("BHAROSA_DB", str(DEFAULT_DB)))
    if not db_path.is_file():
        raise FileNotFoundError(f"Bharosa database not found: {db_path}")

    documents = scheme_documents(str(db_path))

    def retrieve(query):
        return search_schemes(
            query,
            k=k,
            documents=documents,
        )

    verdict = check_claim(message, retrieve)
    payload = jsonable(verdict)

    if not isinstance(payload, dict):
        payload = {"verdict": str(payload)}

    label = (
        getattr(verdict, "label", None)
        or getattr(verdict, "verdict", None)
        or payload.get("label")
        or payload.get("verdict")
        or ""
    )

    explanation = (
        getattr(verdict, "explanation", None)
        or payload.get("explanation")
        or ""
    )

    payload["label"] = str(label)
    payload["verdict"] = str(label)
    payload["explanation"] = str(explanation)

    return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "BharosaAPI/1.0"

    def send_json(self, status, payload):
        body = json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/api/medicine/search":
                q = query.get("q", [""])[0].strip()
                k = int(query.get("k", ["5"])[0])

                if not q:
                    return self.send_json(400, {"error": "Missing query parameter: q"})

                return self.send_json(200, medicine_endpoint(q, k))

            if path == "/api/schemes/search":
                q = query.get("q", [""])[0].strip()
                k = int(query.get("k", ["5"])[0])

                if not q:
                    return self.send_json(400, {"error": "Missing query parameter: q"})

                return self.send_json(200, scheme_endpoint(q, k))

            self.serve_frontend(parsed.path)

        except Exception as exc:
            self.send_json(
                500,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path != "/api/claims/check":
            return self.send_json(404, {"error": "Not found"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))

            message = str(data.get("message", "")).strip()
            k = int(data.get("k", 5))

            if not message:
                return self.send_json(400, {"error": "Missing message"})

            return self.send_json(200, claim_endpoint(message, k))

        except Exception as exc:
            self.send_json(
                500,
                {
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    def serve_frontend(self, request_path):
        relative = request_path.lstrip("/") or "index.html"
        candidate = (FRONTEND / relative).resolve()

        if FRONTEND.resolve() not in candidate.parents and candidate != FRONTEND.resolve():
            return self.send_json(403, {"error": "Forbidden"})

        if not candidate.is_file():
            return self.send_json(404, {"error": "File not found"})

        content = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main():
    host = os.environ.get("BHAROSA_HOST", "127.0.0.1")
    port = int(os.environ.get("BHAROSA_PORT", "8000"))

    server = ThreadingHTTPServer((host, port), Handler)

    print(f"Bharosa server running at http://{host}:{port}")
    print("Frontend: http://127.0.0.1:8000/")
    print("Medicine: /api/medicine/search?q=...")
    print("Schemes:  /api/schemes/search?q=...")
    print("Claims:   /api/claims/check")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Bharosa server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
