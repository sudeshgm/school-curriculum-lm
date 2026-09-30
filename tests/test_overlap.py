"""Planned 12-gram word-overlap filter."""

from sclm.locks import MATCH_TOKENS
from sclm.overlap import filter_documents, ngrams, shares_ngram, word_tokens

TWELVE = "one two three four five six seven eight nine ten eleven twelve"
ELEVEN = "one two three four five six seven eight nine ten eleven"


def test_match_tokens_locked_at_12():
    assert MATCH_TOKENS == 12


def test_exact_12_gram_is_contamination():
    train = f"intro words {TWELVE} outro words here"
    assert shares_ngram(train, TWELVE) is True
    assert shares_ngram(train, TWELVE, n=12) is True


def test_11_gram_overlap_is_not_contamination():
    train = f"intro words {ELEVEN} outro words here now"
    assert shares_ngram(train, ELEVEN, n=12) is False
    assert shares_ngram(train, TWELVE, n=11) is True


def test_case_and_punctuation_are_normalized():
    train = "Intro words. One, two; three: four five six seven eight nine ten eleven twelve!"
    eval_text = "ONE two three four five six seven eight nine ten eleven twelve"
    assert shares_ngram(train, eval_text, n=12) is True
    assert word_tokens("Hello, WORLD!") == ["hello", "world"]


def test_hyphen_splits_into_separate_tokens():
    assert word_tokens("well-known fact") == ["well", "known", "fact"]


def test_12_gram_detected_in_the_middle_of_a_long_document():
    filler = "alpha beta gamma delta epsilon zeta eta theta"
    train = f"{filler} {TWELVE} {filler}"
    assert shares_ngram(train, f"prefix {TWELVE} suffix") is True


def test_eval_shorter_than_12_tokens_does_not_match():
    train = TWELVE + " extra tokens that make this long enough for grams"
    assert shares_ngram(train, ELEVEN, n=12) is False
    assert ngrams(word_tokens(ELEVEN), 12) == set()


def test_filter_drops_only_contaminated_documents():
    clean = "cats sit on mats and do not share the forbidden dozen words at all today"
    dirty = f"A textbook page quotes {TWELVE} while discussing boats."
    other = "totally different material about clouds and kites and nothing else here now"
    kept, dropped = filter_documents([clean, dirty, other], [TWELVE], n=12)
    assert dropped == [1]
    assert kept == [clean, other]


def test_filter_matches_any_eval_item():
    first = "no overlap with the train document in these few words today at all"
    second = TWELVE
    train = f"see the span {TWELVE} inside the chapter"
    kept, dropped = filter_documents([train], [first, second], n=12)
    assert kept == []
    assert dropped == [0]


def test_default_n_is_the_locked_12():
    train = f"see {TWELVE} here"
    assert shares_ngram(train, TWELVE) is True
    # Without an explicit n, an 11-gram-only overlap must not be dropped.
    almost = f"see {ELEVEN} zebra"
    kept, dropped = filter_documents([almost], [ELEVEN])
    assert dropped == []
    assert kept == [almost]


def test_repeated_whitespace_does_not_create_empty_tokens():
    assert word_tokens("  one \n two\t three  ") == ["one", "two", "three"]


def test_digits_count_as_word_tokens():
    gram = "class 1 has twelve red boats near the old dock by school"
    assert len(word_tokens(gram)) == 12
    assert shares_ngram("Note: " + gram + " end", gram) is True
