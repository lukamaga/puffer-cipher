"""base64url: RFC 4648 vectors, strictness and canonical form."""

from __future__ import annotations

import base64

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from puffer.crypto import FormatError, b64url_decode, b64url_encode

PROPERTY = settings(derandomize=True, database=None, deadline=None, max_examples=200)

# RFC 4648 section 10 test vectors, minus the padding.
RFC4648 = [
    (b"", ""),
    (b"f", "Zg"),
    (b"fo", "Zm8"),
    (b"foo", "Zm9v"),
    (b"foob", "Zm9vYg"),
    (b"fooba", "Zm9vYmE"),
    (b"foobar", "Zm9vYmFy"),
]


@pytest.mark.parametrize(("raw", "text"), RFC4648)
def test_rfc4648_vectors(raw: bytes, text: str) -> None:
    assert b64url_encode(raw) == text
    assert b64url_decode(text) == raw


def test_uses_url_safe_alphabet() -> None:
    assert base64.b64encode(b"\xfb\xff") == b"+/8="
    assert b64url_encode(b"\xfb\xff") == "-_8"
    assert b64url_decode("-_8") == b"\xfb\xff"


@pytest.mark.parametrize("text", ["Zg==", "Zm8=", "Zm9vYg==", "Zm9vYmE="])
def test_accepts_correct_padding(text: str) -> None:
    assert b64url_decode(text) == base64.urlsafe_b64decode(text)


@pytest.mark.parametrize(
    "text",
    [
        "Zm9v Yg",  # whitespace
        "Zm9vYg\n",
        "+/8",  # standard alphabet
        "Zm9/",
        "Zg=",  # wrong amount of padding
        "Zg===",
        "Zm9vYg=",
        "Zm8==",
        "Zm9v=",
        "==",
        "Z",  # impossible length
        "Zm9vY",
        "Zh",  # non-canonical: nonzero trailing bits, would also decode to b"f"
        "Zm9",
        "Zm9vYmF",
        "é",
        "Zg=a",
    ],
)
def test_rejects_garbage(text: str) -> None:
    with pytest.raises(FormatError):
        b64url_decode(text)


def test_rejects_non_str() -> None:
    with pytest.raises(FormatError, match="must be str"):
        b64url_decode(b"Zm9v")  # type: ignore[arg-type]


def test_format_error_is_value_error() -> None:
    with pytest.raises(ValueError, match="base64url"):
        b64url_decode("!!")


@PROPERTY
@given(st.binary(max_size=256))
def test_round_trip_is_canonical(data: bytes) -> None:
    text = b64url_encode(data)
    assert set(text) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    assert b64url_decode(text) == data
    assert b64url_decode(text + "=" * (-len(text) % 4)) == data


@PROPERTY
@given(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_="))
def test_decode_accepts_only_canonical_text(text: str) -> None:
    """Whatever decodes must re-encode to the same text: no two spellings per value."""
    try:
        data = b64url_decode(text)
    except FormatError:
        return
    assert b64url_encode(data) == text.rstrip("=")
