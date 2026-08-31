"""
Fix words that the decoder glued together.

Spoken Sinhala barely pauses between words, so the decoder sometimes writes
several words as one long non-word: යපාගන්නවගේදේකට instead of
යපා ගන්න වගේ දේකට. This module repairs that after decoding.

The idea is simple. For every long word that is NOT in the dictionary:

  1. find every way to cut it into pieces that ARE dictionary words
  2. score each option (including "leave it alone") with the language model
  3. keep the best one

Only long non-words are touched. Short words are left alone, and so is
anything already in the dictionary, so a correct word can never be damaged.

Use it like this:

    import kenlm
    from unfuse import WordRepair

    words = {line.strip() for line in open("lm/unigrams.txt")}
    repair = WordRepair(words, kenlm.Model("lm/sinhala.arpa"))
    print(repair.fix("මම යපාගන්නවගේදේකට කැමතියි"))
"""
from __future__ import annotations

import unicodedata


class WordRepair:
    """Splits glued-together words back into dictionary words."""

    def __init__(
        self,
        dictionary: set[str],
        language_model,
        *,
        word_bonus: float = 3.0,
        non_word_penalty: float = 14.0,
        min_word_length: int = 12,
    ) -> None:
        """
        dictionary        -- set of valid words
        language_model    -- a kenlm.Model; scores candidate word sequences
        word_bonus        -- added per word, so that a split into several real
                             words can beat one glued non-word (an n-gram model
                             on its own prefers fewer tokens)
        non_word_penalty  -- subtracted per word that is not in the dictionary
        min_word_length   -- only words at least this long are examined; short
                             unknown words are usually real (names, loanwords),
                             not glue accidents
        """
        self.dictionary = dictionary
        self.lm = language_model
        self.word_bonus = word_bonus
        self.non_word_penalty = non_word_penalty
        self.min_word_length = min_word_length

    def fix(self, text: str) -> str:
        """Return the text with any glued words split."""
        words = unicodedata.normalize("NFC", text).split()
        result: list[str] = []
        for i, word in enumerate(words):
            previous = result[-1] if result else None
            following = words[i + 1] if i + 1 < len(words) else None
            result.extend(self._repair(word, previous, following))
        return " ".join(result)

    # ------------------------------------------------------------ internals

    def _repair(self, word: str, previous: str | None,
                following: str | None) -> list[str]:
        """One word in, one-or-more words out."""
        if len(word) < self.min_word_length or word in self.dictionary:
            return [word]

        candidates = self._splits(word)
        if not candidates:
            return [word]

        best = [word]
        best_score = self._score([word], previous, following)
        for candidate in candidates:
            score = self._score(candidate, previous, following)
            if score > best_score:
                best, best_score = candidate, score
        return best

    def _splits(self, word: str, max_parts: int = 4,
                min_part: int = 2) -> list[list[str]]:
        """Every way to cut `word` into 2..max_parts dictionary words."""
        results: list[list[str]] = []
        parts: list[str] = []

        def walk(start: int) -> None:
            if len(parts) > max_parts:
                return
            if start == len(word):
                if len(parts) >= 2:
                    results.append(parts.copy())
                return
            for end in range(start + min_part, len(word) + 1):
                piece = word[start:end]
                if piece in self.dictionary:
                    parts.append(piece)
                    walk(end)
                    parts.pop()

        walk(0)
        return results

    def _score(self, words: list[str], previous: str | None,
               following: str | None) -> float:
        """Language-model score of a candidate, in its sentence context.

        The bonus and penalty exist because an n-gram model alone always
        prefers the glued form: one unknown word costs a single unknown-word
        hit, while its correct four-word split pays four word probabilities.
        """
        context = ([previous] if previous else []) + words \
            + ([following] if following else [])
        score = self.lm.score(" ".join(context),
                              bos=previous is None, eos=following is None)
        score += self.word_bonus * len(words)
        score -= self.non_word_penalty * sum(
            1 for w in words if w not in self.dictionary)
        return score


# Kept so existing imports keep working.
class Unfuser(WordRepair):
    def __init__(self, lex, lm, max_parts=4, min_part=2, margin=0.0,
                 beta=3.0, oov_penalty=14.0, min_len=12):
        super().__init__(lex, lm, word_bonus=beta,
                         non_word_penalty=oov_penalty, min_word_length=min_len)


if __name__ == "__main__":
    import sys
    import kenlm

    if len(sys.argv) < 2:
        sys.exit('usage: python unfuse.py "text to repair" '
                 "(needs lm/sinhala.arpa and lm/unigrams.txt)")
    words = {line.strip() for line in open("lm/unigrams.txt", encoding="utf-8")
             if line.strip()}
    repair = WordRepair(words, kenlm.Model("lm/sinhala.arpa"))
    print(repair.fix(sys.argv[1]))
