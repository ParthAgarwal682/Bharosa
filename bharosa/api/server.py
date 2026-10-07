import base64
import dataclasses
import io
import json
import mimetypes
import os
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import parse_qs, quote, urlparse

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


VERIFIED_MEDICINE_PRODUCT_URLS: dict[str, str] = {
    "azithral 500 tablet": "https://www.1mg.com/drugs/azithral-500-tablet-47672",
    "azithral 500": "https://www.1mg.com/drugs/azithral-500-tablet-47672",
    "dolo 650 tablet": "https://www.1mg.com/drugs/dolo-650-tablet-74467",
    "dolo 650": "https://www.1mg.com/drugs/dolo-650-tablet-74467",
    "calpol 500mg tablet": "https://www.1mg.com/drugs/calpol-500mg-tablet-25039",
    "calpol 500 tablet": "https://www.1mg.com/drugs/calpol-500mg-tablet-25039",
    "calpol 500": "https://www.1mg.com/drugs/calpol-500mg-tablet-25039",
    "aciloc 150 tablet": "https://www.1mg.com/drugs/aciloc-150-tablet-132338",
    "aciloc 150": "https://www.1mg.com/drugs/aciloc-150-tablet-132338",
}


def get_medicine_purchase_url(candidate: dict) -> tuple[str, bool]:
    """Return (purchase_url, is_direct) for a validated medicine candidate.

    Uses verified exact product URLs for explicitly mapped medications.
    For unmatched medicines, generates a pharmacy search URL strictly constructed
    from the candidate's validated brand, strength, and form (never raw user query).
    """
    brand = str(candidate.get("brand") or candidate.get("title") or "").strip()
    norm_brand = re.sub(r"\s+", " ", brand.lower()).strip()

    if norm_brand in VERIFIED_MEDICINE_PRODUCT_URLS:
        return VERIFIED_MEDICINE_PRODUCT_URLS[norm_brand], True

    strength_val = candidate.get("strength_value")
    strength_unit = str(candidate.get("strength_unit") or "").strip()
    form = str(candidate.get("form") or "").strip()

    strength_str = ""
    if strength_val is not None:
        try:
            val = float(strength_val)
            strength_str = f"{int(val) if val.is_integer() else val}{strength_unit}"
        except (ValueError, TypeError):
            strength_str = f"{strength_val}{strength_unit}"

    parts = [brand]
    if strength_str and strength_str.lower() not in brand.lower():
        parts.append(strength_str)
    if form and form.lower() not in brand.lower():
        parts.append(form)

    search_query = " ".join(parts).strip()
    return f"https://www.1mg.com/search/all?name={quote(search_query)}", False


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
        if isinstance(item, dict):
            if not item.get("title"):
                item["title"] = item.get("brand", item.get("name", ""))
            purchase_url, is_direct = get_medicine_purchase_url(item)
            item["purchase_url"] = purchase_url
            item["is_direct_url"] = is_direct

    return {
        "query": query,
        "candidates": candidates,
    }


def _extract_scheme_title(item):
    text = (item.get("text") or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        first = lines[0]
        if first.lower() in ("about scheme", "about the scheme", "objective", "objectives"):
            if len(lines) > 1:
                return lines[1][:110].rstrip("–-:")
        if len(first) <= 120:
            return first.rstrip("–-:")
        return first[:100].rsplit(" ", 1)[0] + "..."
    url = (item.get("url") or "").strip()
    if url:
        slug = url.split("?")[0].rstrip("/").split("/")[-1]
        if slug and not slug.isdigit():
            return slug.replace("-", " ").replace("_", " ").title()
    return f"Scheme {item.get('doc_id', '')[:8]}"


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
        if isinstance(item, dict):
            if not item.get("title") or item.get("title") == item.get("doc_id"):
                item["title"] = _extract_scheme_title(item)

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

        if parsed.path == "/api/claims/extract-text":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 10 * 1024 * 1024:
                    return self.send_json(413, {"error": "File size exceeds 10MB limit."})
                raw = self.rfile.read(length)
                data = json.loads(raw.decode("utf-8"))
                filename = str(data.get("filename", "")).strip()
                b64_content = str(data.get("file_base64") or data.get("content") or "").strip()

                if not b64_content:
                    if "text" in data and str(data["text"]).strip():
                        return self.send_json(200, {"filename": filename, "text": str(data["text"]).strip()})
                    return self.send_json(400, {"error": "No file content provided."})

                file_bytes = base64.b64decode(b64_content)
                lower_name = filename.lower()

                if lower_name.endswith(".pdf"):
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        extracted = "\n\n".join(
                            (page.extract_text() or "").strip() for page in pdf.pages
                        ).strip()
                    if not extracted:
                        return self.send_json(
                            422,
                            {
                                "error": "No readable text found in PDF. Note: scanned/image-only PDFs require OCR which is not supported.",
                            },
                        )
                    return self.send_json(200, {"filename": filename, "text": extracted})

                elif lower_name.endswith(".txt"):
                    text = file_bytes.decode("utf-8", errors="replace").strip()
                    if not text:
                        return self.send_json(422, {"error": "The uploaded text file is empty."})
                    return self.send_json(200, {"filename": filename, "text": text})

                else:
                    return self.send_json(
                        400,
                        {"error": "Unsupported file format. Please upload a .txt or .pdf file."},
                    )
            except Exception as exc:
                return self.send_json(500, {"error": f"Failed to extract file text: {exc}"})

        if parsed.path != "/api/claims/check":
            return self.send_json(404, {"error": "Not found"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 10 * 1024 * 1024:
                return self.send_json(413, {"error": "Message size exceeds 10MB limit."})
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
