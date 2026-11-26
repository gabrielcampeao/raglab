from raglab.text import normalize_token, tokenize


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Refund POLICY, explained!") == ["refund", "policy", "explained"]


def test_tokenize_removes_stopwords_by_default() -> None:
    assert tokenize("What is the refund window?") == ["refund", "window"]


def test_tokenize_can_keep_stopwords() -> None:
    assert tokenize("What is it", remove_stopwords=False) == ["what", "is", "it"]


def test_tokenize_normalizes_simple_plurals() -> None:
    assert tokenize("refunds notebooks policies class") == ["refund", "notebook", "policy", "class"]


def test_tokenize_keeps_accented_letters() -> None:
    assert tokenize("Café não") == ["café", "não"]


def test_normalize_token_leaves_short_and_special_endings_alone() -> None:
    assert [normalize_token(word) for word in ("bus", "plus", "analysis", "gas")] == [
        "bus",
        "plus",
        "analysis",
        "gas",
    ]
