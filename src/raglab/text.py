from __future__ import annotations

import re

_TOKEN_PATTERN = re.compile(r"[^\W_]+")

_STOPWORD_LIST = (
    "a an and are as at be but by can could do does for from had has have how i if in into "
    "is it its me my of on or our so that the their them then there these they this to was "
    "we were what when where which who why will with would you your"
)

STOPWORDS = frozenset(_STOPWORD_LIST.split())


def normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(text.lower())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return [normalize_token(token) for token in tokens]
