"""Known-answer tests for lexicon query expansion.

The mappings below are labelled fixtures. They are not a health lexicon
and they are not translations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bharosa.text import hinglish
from bharosa.text.hinglish import (
    DEFAULT_LEXICON_PATH,
    ExpansionDebug,
    ExpansionHit,
    HinglishLexicon,
)
from bharosa.text.normalize import tokenize

# Labels only. ``testcanona`` is also a source so a chained lookup would
# be visible if the expander walked more than one hop. ``testaliasa``
# shares ``testcanona`` so a reverse lookup would be visible too.
REVIEWED = {
    "testsourcea": "testcanona",
    "testsourceb": ("testcanonb", "testcanonc"),
    "testcanona": "notachain",
    "testaliasa": "testcanona",
}


def _lexicon() -> HinglishLexicon:
    return HinglishLexicon(REVIEWED)


def test_appends_reviewed_targets_and_keeps_originals() -> None:
    result = _lexicon().expand("testsourcea plain testsourceb")

    assert result == ExpansionDebug(
        original_tokens=("testsourcea", "plain", "testsourceb"),
        expanded_tokens=("testcanona", "testcanonb", "testcanonc"),
        query_tokens=(
            "testsourcea",
            "plain",
            "testsourceb",
            "testcanona",
            "testcanonb",
            "testcanonc",
        ),
        hits=(
            ExpansionHit("testsourcea", ("testcanona",)),
            ExpansionHit("testsourceb", ("testcanonb", "testcanonc")),
        ),
    )
    assert result.query_tokens[: len(result.original_tokens)] == result.original_tokens
    assert result.expanded_tokens == tuple(
        token for hit in result.hits for token in hit.added
    )


def test_target_order_is_the_reviewed_order() -> None:
    lexicon = HinglishLexicon(
        {"testsourcea": ["testcanonc", "testcanona", "testcanonb"]}
    )

    assert lexicon.entries() == (
        ("testsourcea", ("testcanonc", "testcanona", "testcanonb")),
    )
    assert lexicon.expand("testsourcea").expanded_tokens == (
        "testcanonc",
        "testcanona",
        "testcanonb",
    )


def test_one_hop_does_not_expand_canonical_targets() -> None:
    result = _lexicon().expand("testsourcea")

    assert result.original_tokens == ("testsourcea",)
    assert result.expanded_tokens == ("testcanona",)
    assert "notachain" not in result.query_tokens
    assert result.hits == (ExpansionHit("testsourcea", ("testcanona",)),)

    # The canonical form is expanded only when the user actually typed it.
    typed = _lexicon().expand("testcanona")
    assert typed.original_tokens == ("testcanona",)
    assert typed.expanded_tokens == ("notachain",)
    assert typed.query_tokens == ("testcanona", "notachain")


def test_does_not_pull_in_another_source_for_the_same_target() -> None:
    result = _lexicon().expand("testsourcea")

    assert "testaliasa" not in result.query_tokens
    assert result.expanded_tokens == ("testcanona",)


def test_shared_target_is_appended_once_from_the_earlier_source() -> None:
    lexicon = HinglishLexicon(
        {
            "testsourcea": ("testcanona", "testcanonb"),
            "testsourceb": ("testcanonb", "testcanonc"),
        }
    )
    result = lexicon.expand("testsourcea testsourceb")

    assert result.expanded_tokens == ("testcanona", "testcanonb", "testcanonc")
    assert result.hits == (
        ExpansionHit("testsourcea", ("testcanona", "testcanonb")),
        ExpansionHit("testsourceb", ("testcanonc",)),
    )
    assert result.query_tokens == (
        "testsourcea",
        "testsourceb",
        "testcanona",
        "testcanonb",
        "testcanonc",
    )


def test_does_not_repeat_a_token_the_query_already_has() -> None:
    result = _lexicon().expand("testcanonb testsourceb")

    assert result.original_tokens == ("testcanonb", "testsourceb")
    assert result.expanded_tokens == ("testcanonc",)
    assert result.query_tokens == ("testcanonb", "testsourceb", "testcanonc")
    assert result.hits == (ExpansionHit("testsourceb", ("testcanonc",)),)


def test_duplicate_originals_stay_and_the_addition_is_once() -> None:
    result = _lexicon().expand("testsourcea testsourcea")

    assert result.original_tokens == ("testsourcea", "testsourcea")
    assert result.expanded_tokens == ("testcanona",)
    assert result.query_tokens == ("testsourcea", "testsourcea", "testcanona")


def test_unknown_tokens_and_an_empty_mapping_add_nothing() -> None:
    empty = HinglishLexicon({})
    result = empty.expand("testsourcea plain")

    assert result == ExpansionDebug(
        original_tokens=("testsourcea", "plain"),
        expanded_tokens=(),
        query_tokens=("testsourcea", "plain"),
        hits=(),
    )
    assert empty.entries() == ()
    assert empty.expand("").query_tokens == ()


def test_words_absent_from_the_mapping_stay_untranslated() -> None:
    result = HinglishLexicon({}).expand("sasta dawai yojana")

    assert result.original_tokens == ("sasta", "dawai", "yojana")
    assert result.expanded_tokens == ()
    assert result.query_tokens == result.original_tokens
    assert result.hits == ()


def test_debug_lists_original_and_expanded_tokens() -> None:
    result = _lexicon().expand("testsourcea plain")

    assert result.format_debug() == "\n".join(
        [
            "original_tokens: testsourcea plain",
            "expanded_tokens: testcanona",
            "query_tokens: testsourcea plain testcanona",
            "testsourcea -> testcanona",
        ]
    )

    untouched = _lexicon().expand("plain")
    assert untouched.format_debug() == "\n".join(
        [
            "original_tokens: plain",
            "expanded_tokens:",
            "query_tokens: plain",
        ]
    )
    assert _lexicon().expand("").format_debug() == "\n".join(
        [
            "original_tokens:",
            "expanded_tokens:",
            "query_tokens:",
        ]
    )


def test_same_query_expands_the_same_way_twice() -> None:
    lexicon = _lexicon()
    first = lexicon.expand("  TestSourceB, plain TESTSOURCEA ")
    second = lexicon.expand("  TestSourceB, plain TESTSOURCEA ")

    assert first == second
    assert first.format_debug() == second.format_debug()
    assert first.original_tokens == ("testsourceb", "plain", "testsourcea")
    assert first.expanded_tokens == ("testcanonb", "testcanonc", "testcanona")


def test_original_tokens_match_the_shared_analyser() -> None:
    query = "  TESTSOURCEA, plain "
    result = _lexicon().expand(query)

    assert tokenize(query) == ["testsourcea", "plain"]
    assert result.original_tokens == ("testsourcea", "plain")
    assert result.expanded_tokens == ("testcanona",)


def test_stopwords_are_applied_before_lookup() -> None:
    lexicon = _lexicon()

    dropped = lexicon.expand("the testsourcea", remove_stopwords=True)
    assert dropped.original_tokens == ("testsourcea",)
    assert dropped.expanded_tokens == ("testcanona",)
    assert dropped.query_tokens == ("testsourcea", "testcanona")

    kept = lexicon.expand("the testsourcea")
    assert kept.original_tokens == ("the", "testsourcea")
    assert kept.expanded_tokens == ("testcanona",)

    blocked = lexicon.expand(
        "testsourcea plain",
        remove_stopwords=True,
        stopwords=["TESTSOURCEA"],
    )
    assert blocked.original_tokens == ("plain",)
    assert blocked.expanded_tokens == ()
    assert blocked.query_tokens == ("plain",)


def test_indic_source_is_one_token() -> None:
    lexicon = HinglishLexicon({"टोकन": "testcanond"})
    result = lexicon.expand("टोकन plain")

    assert lexicon.entries() == (("टोकन", ("testcanond",)),)
    assert result.original_tokens == ("टोकन", "plain")
    assert result.expanded_tokens == ("testcanond",)
    assert result.query_tokens == ("टोकन", "plain", "testcanond")


def test_casefolded_keys_merge_in_insertion_order() -> None:
    lexicon = HinglishLexicon(
        {"TESTSOURCEA": "testcanonb", "testsourcea": "testcanona testcanonb"}
    )

    assert lexicon.entries() == (("testsourcea", ("testcanonb", "testcanona")),)


def test_mapping_given_to_the_constructor_is_copied() -> None:
    raw = {"testsourcea": "testcanona"}
    lexicon = HinglishLexicon(raw)
    raw["testsourceb"] = "testcanonb"

    assert lexicon.expand("testsourceb").expanded_tokens == ()
    assert lexicon.entries() == (("testsourcea", ("testcanona",)),)


def test_self_target_adds_nothing() -> None:
    lexicon = HinglishLexicon({"testsourcea": "testsourcea"})
    result = lexicon.expand("testsourcea")

    assert result.original_tokens == ("testsourcea",)
    assert result.expanded_tokens == ()
    assert result.query_tokens == ("testsourcea",)
    assert result.hits == ()


def test_rejects_bad_mappings() -> None:
    with pytest.raises(TypeError):
        HinglishLexicon()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        HinglishLexicon([("testsourcea", "testcanona")])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        HinglishLexicon({1: "testcanona"})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        HinglishLexicon({"testsourcea": {"testcanona"}})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        HinglishLexicon({"testsourcea": ["testcanona", 1]})  # type: ignore[list-item]

    with pytest.raises(ValueError):
        HinglishLexicon({"test source": "testcanona"})
    with pytest.raises(ValueError):
        HinglishLexicon({"testsourcea-extra": "testcanona"})
    with pytest.raises(ValueError):
        HinglishLexicon({"": "testcanona"})
    with pytest.raises(ValueError):
        HinglishLexicon({"testsourcea": ""})
    with pytest.raises(ValueError):
        HinglishLexicon({"testsourcea": "???"})

    with pytest.raises(TypeError):
        _lexicon().expand(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        HinglishLexicon.load(1)  # type: ignore[arg-type]


def test_load_labelled_csv_fixture(tmp_path: Path) -> None:
    path = tmp_path / "hinglish_lexicon.csv"
    path.write_text(
        "\n".join(
            [
                "# labelled fixture only. not a health lexicon",
                "hinglish,canonical",
                "",
                "testsourcea,testcanona",
                "TESTSOURCEB,testcanonb testcanonc",
                "testsourceb,testcanonb",
                "टोकन,testcanond",
            ]
        ),
        encoding="utf-8",
    )

    lexicon = HinglishLexicon.load(path)

    assert lexicon.entries() == (
        ("testsourcea", ("testcanona",)),
        ("testsourceb", ("testcanonb", "testcanonc")),
        ("टोकन", ("testcanond",)),
    )
    result = lexicon.expand("TestSourceA टोकन")
    assert result.original_tokens == ("testsourcea", "टोकन")
    assert result.expanded_tokens == ("testcanona", "testcanond")
    assert result.query_tokens == ("testsourcea", "टोकन", "testcanona", "testcanond")


def test_load_headerless_csv(tmp_path: Path) -> None:
    path = tmp_path / "pairs.csv"
    path.write_text("testsourcea,testcanona\ntestsourceb,testcanonb\n", encoding="utf-8")

    lexicon = HinglishLexicon.load(path)

    assert lexicon.entries() == (
        ("testsourcea", ("testcanona",)),
        ("testsourceb", ("testcanonb",)),
    )


def test_empty_reviewed_file_adds_nothing(tmp_path: Path) -> None:
    path = tmp_path / "hinglish_lexicon.csv"
    path.write_text("source,canonical\n", encoding="utf-8")

    lexicon = HinglishLexicon.load(path)
    result = lexicon.expand("testsourcea")

    assert lexicon.entries() == ()
    assert result.original_tokens == ("testsourcea",)
    assert result.expanded_tokens == ()
    assert result.query_tokens == ("testsourcea",)


def test_missing_lexicon_file_is_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "absent.csv"
    assert not missing.exists()

    with pytest.raises(FileNotFoundError, match="reviewed lexicon file not found"):
        HinglishLexicon.load(missing)

    assert not missing.exists()


def test_csv_rejects_bad_rows(tmp_path: Path) -> None:
    wide = tmp_path / "wide.csv"
    wide.write_text("testsourcea,testcanona,extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="wide.csv:1: expected 2 columns"):
        HinglishLexicon.load(wide)

    split_source = tmp_path / "split.csv"
    split_source.write_text(
        "source,canonical\ntestsourcea extra,testcanona\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split.csv:2: .*exactly one token"):
        HinglishLexicon.load(split_source)

    empty_target = tmp_path / "empty.csv"
    empty_target.write_text("testsourcea,???\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty.csv:1: .*at least one token"):
        HinglishLexicon.load(empty_target)


def test_default_path_is_the_config_csv() -> None:
    repo = Path(hinglish.__file__).resolve().parents[2]
    assert DEFAULT_LEXICON_PATH == repo / "config" / "hinglish_lexicon.csv"
    assert DEFAULT_LEXICON_PATH.name == "hinglish_lexicon.csv"


def test_module_does_not_embed_lexicon_entries() -> None:
    source = Path(hinglish.__file__).read_text(encoding="utf-8")
    for word in ("sasta", "dawai", "yojana", "दवाई"):
        assert word not in source
