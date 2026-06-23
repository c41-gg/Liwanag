import numpy as np
from gensim.models import FastText


class FastTextScorer:
    def __init__(self, model_path: str):
        self.model = FastText.load(model_path)

    # ── Core vector utility ────────────────────────────────────

    def vector(self, word: str) -> np.ndarray:
        """
        Returns the 300-dim embedding for any word.
        Works even for OOV words via subword composition.
        """
        return self.model.wv[word.lower()]

    def cosine_similarity(self, a: str, b: str) -> float:
        va = self.vector(a)
        vb = self.vector(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    # ── Two scores the ranker needs ────────────────────────────

    def morphological_similarity(
        self, candidate: str, original: str
    ) -> float:
        """
        How morphologically close is the candidate to the
        original word? High score = minor affix change.
        Low score = completely different word.

        nagluto  ↔ nagluluto  → ~0.99  (aspect change only)
        nagluto  ↔ magluto    → ~0.54  (prefix change)
        nagluto  ↔ kumain     → ~0.40  (different root)
        """
        return self.cosine_similarity(candidate, original)

    def contextual_fit(
        self, candidate: str, context_words: list[str]
    ) -> float:
        """
        How well does the candidate fit with the surrounding
        words in the sentence?
        Averages cosine similarity against all context tokens.
        """
        if not context_words:
            return 0.0
        sims = [
            self.cosine_similarity(candidate, ctx)
            for ctx in context_words
        ]
        return float(np.mean(sims))

    # ── Combined score the ranker calls directly ───────────────

    def score(
        self,
        candidate: str,
        original: str,
        context_words: list[str],
        morph_weight: float = 0.6,
        context_weight: float = 0.4,
    ) -> dict:
        """
        Single entry point for the correction ranker.

        Returns a dict so the ranker can use individual
        scores for logging and debugging, not just the final.
        """
        morph   = self.morphological_similarity(candidate, original)
        context = self.contextual_fit(candidate, context_words)
        final   = morph_weight * morph + context_weight * context

        return {
            "candidate":   candidate,
            "morph_sim":   round(morph,   4),
            "context_fit": round(context, 4),
            "ft_score":    round(final,   4),
        }

    # ── Filter utility — Option B from earlier discussion ──────

    def filter_candidates(
        self,
        candidates: list[str],
        original: str,
        threshold: float = 0.65,
    ) -> list[str]:
        """
        Removes candidates that are morphologically too distant
        from the original word. Prevents the ranker from picking
        a completely different word when a minor fix is needed.

        Falls back to top-3 if nothing passes the threshold.
        """
        passed = [
            c for c in candidates
            if self.morphological_similarity(c, original) >= threshold
        ]
        return passed if passed else candidates[:3]