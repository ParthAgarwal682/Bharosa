"""Deterministic Soundex phonetic encoding for medicine brand names.

IR concept: Phonetic hashing for vocabulary mismatch and spelling variants.
Users frequently misspell trade names (e.g. 'Pantocid' vs 'Pantosid' vs 'Pantocide').
Soundex collapses phonetically similar consonant sequences into a canonical
4-character alphanumeric code (e.g. 'P532'), acting as a baseline recall-expansion
technique before character n-gram ranking.
"""

from __future__ import annotations

import re

from bharosa.text.normalize import normalize as normalize_text


# Soundex consonant mapping (letters B, F, P, V -> 1, etc.)
_SOUNDEX_MAP: dict[str, str] = {
    "B": "1",
    "F": "1",
    "P": "1",
    "V": "1",
    "C": "2",
    "G": "2",
    "J": "2",
    "K": "2",
    "Q": "2",
    "S": "2",
    "X": "2",
    "Z": "2",
    "D": "3",
    "T": "3",
    "L": "4",
    "M": "5",
    "N": "5",
    "R": "6",
}


def soundex(text: str) -> str:
    """Compute the standard American Soundex code for a single word.

    Rules:
    1. Retain the first letter (uppercase).
    2. Map consonants to digits 1-6 according to phonetic categories.
    3. Collapse adjacent identical digit codes. Letters H and W do not
       reset adjacent code suppression; vowels (A, E, I, O, U, Y) reset it.
    4. Return first letter + 3 digits, zero-padded or truncated to length 4.
    5. Returns empty string if text has no alphabetic characters.
    """
    if not text or not isinstance(text, str):
        return ""

    # Retain alphabetic ASCII letters
    cleaned = "".join(ch for ch in text.strip() if ch.isalpha() and ch.isascii())
    if not cleaned:
        return ""

    cleaned = cleaned.upper()
    first_letter = cleaned[0]

    digits: list[str] = []
    prev_code = _SOUNDEX_MAP.get(first_letter, "")

    for ch in cleaned[1:]:
        curr_code = _SOUNDEX_MAP.get(ch, "")
        if curr_code:
            if curr_code != prev_code:
                digits.append(curr_code)
                prev_code = curr_code
        else:
            # H and W do not break adjacent duplicates; vowels do
            if ch not in ("H", "W"):
                prev_code = ""

    soundex_digits = "".join(digits)[:3]
    return (first_letter + soundex_digits).ljust(4, "0")


def soundex_brand(brand: str) -> str:
    """Extract and encode the primary brand token into a Soundex representation.

    Normalizes the input brand text and returns the Soundex code of the
    leading alphabetic brand token (e.g. 'Pantocid' from 'Pantocid 40 Tablet').
    Returns an empty string if no valid alphabetic brand token is found.
    """
    if not brand or not isinstance(brand, str):
        return ""

    norm = normalize_text(brand)
    if not norm:
        return ""

    # Split into words and find the first word containing alphabetic characters
    for token in norm.split():
        code = soundex(token)
        if code:
            return code
    return ""


def soundex_brand_tokens(brand: str) -> tuple[str, ...]:
    """Compute Soundex codes for all alphabetic tokens in a brand name."""
    if not brand or not isinstance(brand, str):
        return ()

    norm = normalize_text(brand)
    if not norm:
        return ()

    codes: list[str] = []
    for token in norm.split():
        code = soundex(token)
        if code:
            codes.append(code)
    return tuple(codes)
