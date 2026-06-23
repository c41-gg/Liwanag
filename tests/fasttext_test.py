import pytest # type: ignore
from liwanag.correction.fasttext_scorer import FastTextScorer


MODEL_PATH = "data/models/tagalog_fasttext.model"

@pytest.fixture(scope="module")
def scorer():
    return FastTextScorer(MODEL_PATH)


# ── Vector tests ───────────────────────────────────────────────

def test_vector_shape(scorer):
    vec = scorer.vector("nagluto")
    assert vec.shape == (300,)

def test_oov_vector_shape(scorer):
    # word likely never seen in training
    vec = scorer.vector("nakapaglulutong")
    assert vec.shape == (300,)


# ── Morphological similarity tests ────────────────────────────

def test_aspect_change_is_high(scorer):
    # nagluto → nagluluto is an aspect change only
    sim = scorer.morphological_similarity("nagluluto", "nagluto")
    assert sim > 0.90, f"Expected > 0.90, got {sim}"

def test_prefix_change_is_moderate(scorer):
    # nag- → mag- is a meaningful prefix difference
    sim = scorer.morphological_similarity("magluto", "nagluto")
    assert 0.40 < sim < 0.80, f"Expected 0.40–0.80, got {sim}"

def test_unrelated_word_is_low(scorer):
    sim = scorer.morphological_similarity("bahay", "nagluto")
    assert sim < 0.50, f"Expected < 0.50, got {sim}"


# ── Filter tests ───────────────────────────────────────────────

def test_filter_removes_distant_candidates(scorer):
    candidates = ["nagluluto", "magluto", "kumain", "bahay"]
    filtered   = scorer.filter_candidates(candidates, "nagluto", threshold=0.65)
    assert "bahay"  not in filtered
    assert "kumain" not in filtered

def test_filter_fallback_on_empty(scorer):
    # if nothing passes threshold, return top 3 rather than empty list
    candidates = ["kumain", "bahay", "mesa", "silya"]
    filtered   = scorer.filter_candidates(candidates, "nagluto", threshold=0.99)
    assert len(filtered) == 3


# ── Score dict test ────────────────────────────────────────────

def test_score_returns_expected_keys(scorer):
    result = scorer.score(
        candidate     = "nagluluto",
        original      = "nagluto",
        context_words = ["ang", "nanay", "ng", "kanin"]
    )
    assert "candidate"   in result
    assert "morph_sim"   in result
    assert "context_fit" in result
    assert "ft_score"    in result
    assert 0.0 <= result["ft_score"] <= 1.0