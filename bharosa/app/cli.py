"""Explicit search commands over the existing Bharosa modules.

IR concept: a query front end. ``medicine`` loads a caller-supplied
file with ``load_medicines`` and passes those records to
``search_medicine``. ``scheme`` loads zone documents from the crawler
database and calls ``search_schemes``. ``claim`` calls ``check_claim``.
Nothing here scores, expands, or cites on its own. A missing module
stops the command. Results are not replaced with a stand-in.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _REPO_ROOT / "data" / "bharosa.db"


def main(argv: list[str] | None = None) -> None:
    """Dispatch one subcommand. Intermediate IR state is behind ``--verbose``."""
    _configure_stdio()
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "medicine":
        _cmd_medicine(
            args.query,
            k=args.k,
            medicine_file=args.medicine_file,
            verbose=args.verbose,
        )
    elif args.command == "scheme":
        _cmd_scheme(args)
    elif args.command == "claim":
        _cmd_claim(args.message, verbose=args.verbose)
    else:
        parser.error(f"unknown command {args.command!r}")


def _cmd_medicine(query: str, *, k: int, medicine_file: str, verbose: bool) -> None:
    """Load one medicine file, then rank candidates from those records.

    IR concept: brand retrieval over an explicit corpus. The file path
    is the caller's. ``load_medicines`` returns ``(records, stats)``.
    Only ``records`` are passed to ``search_medicine``. Tokens come from
    the shared tokeniser. Constraints come from ``parse_medicine_query``.
    Ranking stays inside ``search_medicine``. This command does not
    expand the query and does not invent rows when the load fails.
    """
    path = _require_medicine_file(medicine_file)
    parse_medicine_query, search_medicine, tokenize = _load_medicine()
    try:
        parsed = parse_medicine_query(query)
        tokens = tokenize(query)
    except (TypeError, ValueError) as exc:
        _fail(f"medicine query was rejected: {exc}")

    records, stats = _load_medicine_records(path)

    print(f"query: {query}")
    print(f"k: {k}")
    print(f"medicine file: {path}")
    print(f"records: {len(records)}")
    if verbose:
        _section("tokens")
        _print_tokens(tokens)
        _section("query expansion")
        print("search_medicine does not expand the query")
        _section("filters")
        print(f"brand: {_shown(parsed.brand)}")
        print(f"strength_value: {_shown(parsed.strength_value)}")
        print(f"strength_unit: {_shown(parsed.strength_unit)}")
        print(f"form: {_shown(parsed.form)}")
        print(f"release: {_shown(parsed.release)}")
        print(f"ambiguous: {parsed.ambiguous}")
        _section("load")
        _print_load_stats(stats)
        _section("postings")
        print(
            "search_medicine ranks with character n-gram cosine "
            "(bharosa.medicine.ngram). It does not read an inverted index."
        )

    try:
        result = search_medicine(query, records=records, k=k)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        _fail(f"medicine search failed: {exc}")

    if verbose:
        _section("weights")
        if result.candidates:
            for candidate in result.candidates:
                print(f"rank {candidate.rank}  brand={candidate.brand}")
                print(candidate.similarity.format_score())
        else:
            print("no candidate was returned")
    _section("scores")
    print(result.format_result())
    if verbose:
        _section("top-K zones")
        print("search_medicine returns medicine candidates, listed under scores")
        _section("citations")
        print("search_medicine does not generate cited sentences")


def _cmd_scheme(args: argparse.Namespace) -> None:
    """Load the crawl corpus and return ranked zones from ``search_schemes``.

    IR concept: zone retrieval. The corpus is ``load_zone_documents``.
    Filters and flags are passed through. ``--verbose`` prints tokens,
    lexicon expansion, parametric postings, inverted postings, and the
    ranker's own weight dump. The ranked list is still ``search_schemes``.
    """
    _reject_unused_scheme_options(args)
    load_zone_documents, search_schemes, tokenize = _load_scheme()
    weights = _net_weights(args)
    zone_weights = _parse_zone_weights(args.zone_weight)
    filters = _scheme_filters(args)
    flags = _scheme_flags(args)

    try:
        documents = load_zone_documents(args.db)
    except (OSError, TypeError, ValueError) as exc:
        _fail(f"scheme corpus is unavailable: {exc}")

    lexicon = None
    expansion = None
    if args.hinglish:
        lexicon, expansion = _load_expansion(args.lexicon, args.query)
        tokens = expansion.query_tokens
    else:
        try:
            tokens = tuple(tokenize(args.query))
        except (TypeError, ValueError) as exc:
            _fail(f"scheme query was rejected: {exc}")
    scoring_text = " ".join(tokens)

    print(f"query: {args.query}")
    print(f"k: {args.k}")
    print(f"database: {args.db}")
    print(f"corpus: {len(documents)} zone documents")
    print("flags: " + (_flag_text(flags) if flags else "{}"))
    print("filters: " + (_filter_text(filters) if filters else "{}"))

    if args.verbose:
        try:
            _print_scheme_trace(
                documents,
                args.query,
                tokens,
                expansion,
                filters,
                scoring_text,
                k=args.k,
                use_bm25=bool(flags and flags.get("bm25")),
            )
        except (ImportError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            _fail(f"scheme trace failed: {exc}")

    kwargs: dict[str, object] = {"documents": documents}
    if weights is not None:
        kwargs["weights"] = weights
    if zone_weights is not None:
        kwargs["zone_weights"] = zone_weights
    if args.freshness_scale_days is not None:
        kwargs["freshness_scale_days"] = args.freshness_scale_days
    if args.as_of is not None:
        kwargs["as_of"] = args.as_of
    if lexicon is not None:
        kwargs["lexicon"] = lexicon

    try:
        hits = search_schemes(args.query, filters, args.k, flags, **kwargs)
    except (ImportError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        _fail(f"scheme search failed: {exc}")

    _print_scores(hits)
    _print_zones(hits)
    if args.verbose:
        _print_citations(args.query, hits)


def _cmd_claim(message: str, *, verbose: bool) -> None:
    """Pass a pasted message to ``check_claim``.

    IR concept: a claim checked against retrieved official text. The
    comparison lives in ``bharosa.rag.claimcheck``. This command does
    not invent a verdict when that module cannot be imported.
    """
    check_claim = _load_check_claim()
    try:
        verdict = check_claim(message)
    except (ImportError, RuntimeError, TypeError, ValueError, OSError) as exc:
        _fail(f"claim check failed: {exc}")

    print(f"message: {message}")
    if verbose:
        _section("tokens")
        print("check_claim owns tokenisation of the pasted message")
        _section("query expansion")
        print("check_claim owns any expansion it applies")
        _section("filters")
        print("check_claim owns any filters it applies")
        _section("postings")
        print("check_claim owns any postings it reads")
        _section("weights")
        print("check_claim owns any weights it computes")
        _section("scores")
        print("check_claim owns the comparison score")
        _section("top-K zones")
        print("check_claim owns the official zone it retrieves")
    _section("citations")
    _print_value(verdict)


def _print_scheme_trace(
    documents: list[object],
    query: str,
    tokens: tuple[str, ...],
    expansion: object,
    filters: dict[str, object] | None,
    scoring_text: str,
    *,
    k: int,
    use_bm25: bool,
) -> None:
    """Print analyser and index dumps from the public format methods.

    IR concept: an inspectable run. Postings, parametric buckets, and
    weights are the modules' own dumps for the same zone texts
    ``search_schemes`` receives. The dump is not a second ranking.
    """
    from bharosa.index.inverted import InvertedIndex
    from bharosa.index.params import ParametricIndex
    from bharosa.index.positional import PositionalIndex
    from bharosa.rank.tfidf import TfidfRanker

    _section("tokens")
    _print_tokens(tokens)
    _section("query expansion")
    if expansion is None:
        print("hinglish is off, so search_schemes scores the analysed query tokens")
    else:
        print(expansion.format_debug())  # type: ignore[attr-defined]
    _section("filters")
    print(_filter_text(filters) if filters else "{}")

    texts = {doc.doc_id: doc.text for doc in documents}  # type: ignore[attr-defined]
    inverted = InvertedIndex.from_documents(texts)
    positional = PositionalIndex.from_documents(texts)
    parametric = ParametricIndex()
    parametric.add_documents(
        doc.as_contract() for doc in documents  # type: ignore[attr-defined]
    )

    _section("postings")
    _block(inverted.format_postings(tokens) if tokens else "")
    _section("positional postings")
    _block(positional.format_postings(tokens) if tokens else "")
    stripped = query.strip()
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        _section("phrase")
        print(positional.format_phrase(query))
    _section("parametric postings")
    print(parametric.format_index())
    matched = " ".join(parametric.matching_ids(filters))
    print("matching_ids: " + (matched if matched else "<none>"))

    _section("weights")
    print("lnc.ltc")
    print(
        "These weights cover every loaded zone. "
        "The scores section is search_schemes after filters."
    )
    print(TfidfRanker(inverted).format_scores(scoring_text, k=k))
    if use_bm25:
        from bharosa.rank.bm25 import BM25Baseline

        print("bm25 baseline")
        print(BM25Baseline(texts).format_scores(scoring_text, k=k))


def _print_scores(hits: list[object]) -> None:
    """Print the scores ``search_schemes`` stored on each hit."""
    _section("scores")
    if not hits:
        print("none")
        return
    for hit in hits:
        print(
            f"{hit.rank}  doc_id={hit.doc_id}  zone={hit.zone}  "  # type: ignore[attr-defined]
            f"cosine={hit.cosine:.6f}  "  # type: ignore[attr-defined]
            f"g_component={hit.g_component:.6f}  "  # type: ignore[attr-defined]
            f"freshness={hit.freshness:.6f}  "  # type: ignore[attr-defined]
            f"zone_weight={hit.zone_weight:.6f}  "  # type: ignore[attr-defined]
            f"net={_score(hit.net)}  "  # type: ignore[attr-defined]
            f"bm25_score={_score(hit.bm25_score)}"  # type: ignore[attr-defined]
        )


def _print_zones(hits: list[object]) -> None:
    """Print the top-K zone text returned by ``search_schemes``."""
    _section("top-K zones")
    if not hits:
        print("none")
        return
    for hit in hits:
        changed = hit.last_changed_at  # type: ignore[attr-defined]
        print(f"{hit.rank}  doc_id={hit.doc_id}  zone={hit.zone}")  # type: ignore[attr-defined]
        print(f"  url: {hit.url}")  # type: ignore[attr-defined]
        print(f"  last_changed_at: {_shown(changed)}")
        print("  text:")
        text = hit.text  # type: ignore[attr-defined]
        for line in text.splitlines() or [""]:
            print(f"    {line}")


def _print_citations(query: str, hits: list[object]) -> None:
    """Call the citation checker when it is installed.

    IR concept: a generated sentence checked against its cited zone.
    ``answer`` and ``check_citations`` are optional teammates' modules.
    A failed import is reported. No citation is filled in locally.
    """
    _section("citations")
    loaded = _load_rag()
    if isinstance(loaded, str):
        print(loaded)
        return
    answer, check_citations = loaded
    try:
        produced = answer(query, hits)
        checks = check_citations(produced, hits)
    except (ImportError, RuntimeError, TypeError, ValueError, OSError) as exc:
        _fail(f"citation check failed: {exc}")
    print("answer:")
    _print_value(produced)
    print("checks:")
    _print_value(checks)


def _require_medicine_file(medicine_file: str) -> Path:
    """Stop when the caller did not point at a medicine file.

    The dataset path is never filled in here. A missing path is reported
    before search, and no stand-in corpus is built.
    """
    path = Path(medicine_file)
    if not path.exists():
        _fail(f"medicine file not found: {path}")
    if not path.is_file():
        _fail(f"medicine file is not a file: {path}")
    return path


def _load_medicine_records(path: Path) -> tuple[object, object]:
    """Call ``load_medicines(path)`` and keep the record list and the stats.

    IR concept: ingestion before retrieval. The loader's pair is
    unpacked here. An empty record list is a failed load, not a search
    over nothing. The stats object is not passed into search.
    """
    try:
        from bharosa.medicine.loader import load_medicines
    except ImportError as exc:
        _fail(
            "medicine loader is unavailable: "
            f"bharosa.medicine.loader.load_medicines could not be imported ({exc}). "
            "No records were substituted."
        )
    if not callable(load_medicines):
        _fail(
            "medicine loader is unavailable: "
            "bharosa.medicine.loader.load_medicines is not callable. "
            "No records were substituted."
        )
    try:
        loaded = load_medicines(path)
    except Exception as exc:
        _fail(f"medicine load failed: {type(exc).__name__}: {exc}")
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        _fail(
            "medicine load failed: load_medicines() must return (records, stats), "
            f"got {type(loaded).__name__}"
        )
    records, stats = loaded
    if records is None or isinstance(records, (str, bytes, bytearray, Mapping)):
        _fail(
            "medicine load failed: records must be a sequence of medicine rows, "
            f"got {type(records).__name__}"
        )
    try:
        count = len(records)
    except TypeError as exc:
        _fail(f"medicine load failed: {type(exc).__name__}: {exc}")
    if count == 0:
        _fail(
            "medicine file produced no records "
            f"({_stat_summary(stats)}). Search was not run."
        )
    return records, stats


def _print_load_stats(stats: object) -> None:
    """Print loader accounting fields the stats object actually has."""
    fields = getattr(type(stats), "__dataclass_fields__", None)
    names: tuple[str, ...]
    if isinstance(fields, dict) and fields:
        names = tuple(fields)
    else:
        names = (
            "total_rows",
            "accepted_rows",
            "dropped_rows",
            "duplicate_counts",
            "reason_counts",
        )
    printed = False
    for name in names:
        if not hasattr(stats, name):
            continue
        print(f"{name}: {getattr(stats, name)}")
        printed = True
    if not printed:
        print(stats)


def _stat_summary(stats: object) -> str:
    """One line of loader counts for a failed empty load."""
    parts: list[str] = []
    for name in ("total_rows", "accepted_rows", "dropped_rows", "duplicate_counts"):
        if hasattr(stats, name):
            parts.append(f"{name}={getattr(stats, name)}")
    return " ".join(parts) if parts else "no load statistics"


def _load_medicine() -> tuple[object, object, object]:
    """Import medicine search. Refuse to continue when the import fails."""
    try:
        from bharosa.medicine.search import parse_medicine_query, search_medicine
        from bharosa.text.normalize import tokenize
    except ImportError as exc:
        _fail(
            "medicine route is unavailable: "
            f"bharosa.medicine.search could not be imported ({exc}). "
            "No mock results were used."
        )
    return parse_medicine_query, search_medicine, tokenize


def _load_scheme() -> tuple[object, object, object]:
    """Import scheme retrieval and the crawler corpus loader."""
    try:
        from bharosa.index.zones import search_schemes
        from bharosa.text.normalize import tokenize
        from bharosa.text.zones import load_zone_documents
    except ImportError as exc:
        _fail(
            "scheme route is unavailable: "
            "bharosa.index.zones.search_schemes or "
            f"bharosa.text.zones.load_zone_documents could not be imported ({exc}). "
            "No mock zones were used."
        )
    return load_zone_documents, search_schemes, tokenize


def _load_expansion(path: str | None, query: str) -> tuple[object, object]:
    """Load the reviewed lexicon and expand the query once for the trace.

    IR concept: dictionary query expansion. The same lexicon object is
    passed into ``search_schemes``, which expands again on the user
    string. This command does not write lexicon rows.
    """
    try:
        from bharosa.text.hinglish import HinglishLexicon

        lexicon = HinglishLexicon.load(path)
        expansion = lexicon.expand(query)
    except ImportError as exc:
        _fail(
            "query expansion is unavailable: "
            f"bharosa.text.hinglish could not be imported ({exc}). "
            "No expansion list was substituted."
        )
    except (OSError, TypeError, ValueError) as exc:
        _fail(f"query expansion is unavailable: {exc}")
    return lexicon, expansion


def _load_check_claim() -> object:
    """Import ``check_claim``. Stop when the claim module is absent."""
    try:
        from bharosa.rag.claimcheck import check_claim
    except ImportError as exc:
        _fail(
            "claim route is unavailable: "
            f"bharosa.rag.claimcheck.check_claim could not be imported ({exc}). "
            "No mock verdict was used."
        )
    if not callable(check_claim):
        _fail(
            "claim route is unavailable: "
            "bharosa.rag.claimcheck.check_claim is not callable. "
            "No mock verdict was used."
        )
    return check_claim


def _load_rag() -> tuple[object, object] | str:
    """Import ``answer`` and ``check_citations``, or describe the import error."""
    try:
        from bharosa.rag.answer import answer
        from bharosa.rag.citations import check_citations
    except ImportError as exc:
        return (
            "citations are unavailable: "
            "bharosa.rag.answer.answer and "
            "bharosa.rag.citations.check_citations could not be imported "
            f"({exc}). No citation was invented."
        )
    if not callable(answer) or not callable(check_citations):
        return (
            "citations are unavailable: answer() or check_citations() "
            "is missing. No citation was invented."
        )
    return answer, check_citations


def _scheme_filters(args: argparse.Namespace) -> dict[str, object] | None:
    """Build the parametric filter dict from the flags the user set."""
    filters: dict[str, object] = {}
    if args.state is not None:
        filters["state"] = args.state
    if args.zone is not None:
        filters["zone"] = args.zone
    if args.condition:
        filters["conditions"] = list(args.condition)
    return filters or None


def _scheme_flags(args: argparse.Namespace) -> dict[str, bool] | None:
    """Build the contract flag dict. Absent switches stay out of the dict."""
    flags: dict[str, bool] = {}
    if args.hinglish:
        flags["hinglish"] = True
    if args.bm25:
        flags["bm25"] = True
    if args.g_score:
        flags["g_score"] = True
    if args.freshness:
        flags["freshness"] = True
    if args.zone_weights:
        flags["zone_weights"] = True
    return flags or None


def _reject_unused_scheme_options(args: argparse.Namespace) -> None:
    """Reject options that would be ignored by the search call."""
    if args.lexicon and not args.hinglish:
        _fail("--lexicon applies only with --hinglish")
    if args.as_of is not None and not args.freshness:
        _fail("--as-of applies only with --freshness")
    if args.freshness_scale_days is not None and not args.freshness:
        _fail("--freshness-scale-days applies only with --freshness")
    if args.zone_weight and not args.zone_weights:
        _fail("--zone-weight applies only with --zone-weights")


def _net_weights(args: argparse.Namespace) -> object:
    """Build ``NetScoreWeights`` only when the user supplied every coefficient."""
    values = (args.w_cosine, args.w_g_score, args.w_freshness)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        _fail(
            "pass --w-cosine, --w-g-score, and --w-freshness together. "
            "Missing coefficients are not filled in."
        )
    try:
        from bharosa.rank.netscore import NetScoreWeights
    except ImportError as exc:
        _fail(
            "net-score weights are unavailable: "
            f"bharosa.rank.netscore could not be imported ({exc})."
        )
    try:
        return NetScoreWeights(args.w_cosine, args.w_g_score, args.w_freshness)
    except (TypeError, ValueError) as exc:
        _fail(f"net-score weights were rejected: {exc}")


def _parse_zone_weights(items: list[str] | None) -> dict[str, float] | None:
    """Parse repeated ``zone=number`` options into a weight table."""
    if not items:
        return None
    table: dict[str, float] = {}
    for item in items:
        name, sep, raw = item.partition("=")
        if sep != "=" or name == "" or raw == "":
            _fail(
                f"zone weight must be zone=number, got {item!r}. "
                "Example: --zone-weight eligibility=1.5"
            )
        try:
            value = float(raw)
        except ValueError:
            _fail(f"zone weight {item!r} is not a number")
        if name in table:
            _fail(f"zone weight for {name!r} was given twice")
        table[name] = value
    return table


def _parser() -> argparse.ArgumentParser:
    """Build the medicine, scheme, and claim commands."""
    parser = argparse.ArgumentParser(
        prog="python -m bharosa.app.cli",
        description=(
            "Search with an explicit command. "
            "medicine calls search_medicine. "
            "scheme calls search_schemes. "
            "claim calls check_claim."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    medicine = sub.add_parser("medicine", help="rank same-salt medicine candidates")
    medicine.add_argument("query", help="brand query, for example: pantocid 40")
    medicine.add_argument("-k", "--k", type=int, default=5)
    medicine.add_argument(
        "--medicine-file",
        required=True,
        metavar="PATH",
        help="CSV path passed to load_medicines",
    )
    medicine.add_argument(
        "--verbose",
        action="store_true",
        help="print tokens, filters, load statistics, n-gram weights, and scores",
    )

    scheme = sub.add_parser("scheme", help="rank official scheme zones")
    scheme.add_argument("query", help="scheme query")
    scheme.add_argument("-k", "--k", type=int, default=5)
    scheme.add_argument(
        "--verbose",
        action="store_true",
        help="print tokens, expansion, postings, weights, scores, and citations",
    )
    scheme.add_argument(
        "--db",
        default=str(_DEFAULT_DB),
        help=f"crawler SQLite database (default: {_DEFAULT_DB})",
    )
    scheme.add_argument("--state", help="parametric state filter")
    scheme.add_argument("--zone", help="parametric zone filter")
    scheme.add_argument(
        "--condition",
        action="append",
        help="parametric condition filter; repeat to require every condition",
    )
    scheme.add_argument("--hinglish", action="store_true", help="expand with the reviewed lexicon")
    scheme.add_argument("--lexicon", help="reviewed lexicon CSV, with --hinglish")
    scheme.add_argument("--bm25", action="store_true", help="rank with the BM25 baseline")
    scheme.add_argument("--g-score", action="store_true", help="include g(d) in the net score")
    scheme.add_argument("--freshness", action="store_true", help="include freshness in the net score")
    scheme.add_argument(
        "--zone-weights",
        action="store_true",
        help="multiply the net score by a zone weight",
    )
    scheme.add_argument("--w-cosine", type=float)
    scheme.add_argument("--w-g-score", type=float)
    scheme.add_argument("--w-freshness", type=float)
    scheme.add_argument(
        "--zone-weight",
        action="append",
        help="zone=number, with --zone-weights. Example: eligibility=1.5",
    )
    scheme.add_argument("--freshness-scale-days", type=float)
    scheme.add_argument("--as-of", help="ISO datetime for freshness, with --freshness")

    claim = sub.add_parser("claim", help="check a pasted fee or deadline claim")
    claim.add_argument("message", help="pasted message")
    claim.add_argument(
        "--verbose",
        action="store_true",
        help="print the claim checker's verdict with section labels",
    )
    return parser


def _print_tokens(tokens: tuple[str, ...] | list[str]) -> None:
    """Print one analysed token sequence."""
    print(" ".join(tokens) if tokens else "<empty>")


def _print_value(value: object) -> None:
    """Print a module object with its formatter when it has one."""
    for name in ("format_result", "format_debug", "format_scores"):
        formatter = getattr(value, name, None)
        if callable(formatter):
            print(formatter())
            return
    print(value)


def _flag_text(flags: dict[str, bool]) -> str:
    return " ".join(f"{key}={value}" for key, value in flags.items())


def _filter_text(filters: dict[str, object]) -> str:
    parts: list[str] = []
    for key, value in filters.items():
        if isinstance(value, list):
            rendered = ",".join(str(item) for item in value)
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


def _block(text: str) -> None:
    print(text if text else "<empty>")


def _section(title: str) -> None:
    print(f"\n[{title}]")


def _shown(value: object) -> str:
    if value is None:
        return "<not recorded>"
    if value == "":
        return "<empty>"
    return str(value)


def _score(value: object) -> str:
    if value is None:
        return "<not scored>"
    return f"{value:.6f}"


def _fail(message: str) -> NoReturn:
    """Print one reason and stop. The caller does not continue with a stand-in."""
    sys.stdout.flush()
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _configure_stdio() -> None:
    """Ask the console for UTF-8 so Devanagari queries can be printed."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (OSError, ValueError):
            continue


if __name__ == "__main__":
    if __package__ in {None, ""}:
        sys.path.insert(0, str(_REPO_ROOT))
    main()
