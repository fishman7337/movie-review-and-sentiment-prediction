from movie_sentiment_rnn.text import (
    lemmatise_text,
    preprocess_review,
    remove_punctuation,
    remove_stopwords,
    tokenize,
)


def test_remove_punctuation_normalises_spaces() -> None:
    assert remove_punctuation("Great!!!  Movie.") == "Great Movie"


def test_tokenize_lowercases_words() -> None:
    assert tokenize("The Movie, Again") == ["the", "movie", "again"]


def test_remove_stopwords_supports_english_and_malay() -> None:
    assert remove_stopwords("the movie was great", "english") == "movie great"
    assert remove_stopwords("saya suka filem ini", "malay") == "suka filem"


def test_lemmatise_text_falls_back_safely() -> None:
    result = lemmatise_text("movies", "english")
    assert isinstance(result, str)
    assert result


def test_preprocess_review_runs_full_pipeline() -> None:
    assert preprocess_review("The movie was GREAT!!!", "english") == "movie great"
