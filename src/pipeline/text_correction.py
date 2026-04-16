from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from src.config import TextCorrectionConfig


BUILTIN_TR_LEXICON = {
    "çocuk", "çocuklar", "dergi", "dergisi", "hikaye", "hikayeler", "masal", "masallar",
    "oyun", "oyuncak", "resim", "görsel", "okul", "öğrenci", "öğretmen", "sınıf",
    "sınav", "geçiş", "kitap", "kitaplar", "okuma", "yazma", "başlık", "içerik",
    "türkiye", "istanbul", "ankara", "antalya", "fethiye", "norveç", "dünya", "tarih",
    "tarihi", "yol", "yolu", "gezi", "gezmek", "gençler", "merhaba", "sevgili",
    "kampüs", "kayıt", "online", "burs", "kontenjan", "katılabilir", "sokak",
    "hayvan", "hayvanlar", "mutlu", "yardım", "proje", "projeler", "yayın", "sahibi",
    "gazetecilik", "müdür", "sorumlu", "reklam", "satış", "hazırlayan", "telefon",
    "ile", "için", "olarak", "kadar", "başlayıp", "uzanan", "uzun", "mesafeli",
    "yürüyüş", "tehlikeli", "virajlar", "dalgalar", "inşa", "edilen", "öğrencileri",
    "katılım", "imkanı", "imkani", "varan", "beylikdüzü", "bahçelievler", "ıspartakule",
    "diye", "yok", "yanlışım", "büyüklerimiz", "şey", "şeyi", "şehit", "şükrederdi",
    "dışarıya", "çıkışır", "görür", "saygın", "gün", "komşu", "sey", "seyahat", "hayvanlarıyla",
}


WORD_RE = re.compile(r"\b[\wÇĞİÖŞÜçğıöşü]+\b", flags=re.UNICODE)
TR_ALPHA_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]")

CONFUSABLE_GROUPS = [
    ("y", "v"),
    ("i", "ı"),
    ("o", "ö"),
    ("u", "ü"),
    ("g", "ğ"),
    ("s", "ş"),
    ("c", "ç"),
]
CONFUSABLE_PAIRS = {
    (a, b) for a, b in (
        *[(x, y) for x, y in CONFUSABLE_GROUPS],
        *[(y, x) for x, y in CONFUSABLE_GROUPS],
    )
}


@dataclass
class TextCorrector:
    cfg: TextCorrectionConfig
    lexicon: set[str]

    @classmethod
    def from_config(cls, cfg: TextCorrectionConfig) -> "TextCorrector":
        words = set(BUILTIN_TR_LEXICON)
        if cfg.lexicon_path:
            p = Path(cfg.lexicon_path)
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    w = line.strip().lower()
                    if w:
                        words.add(w)
        return cls(cfg=cfg, lexicon=words)

    @staticmethod
    def _norm(word: str) -> str:
        return word.lower()

    @staticmethod
    def _char_sub_cost(a: str, b: str) -> float:
        if a == b:
            return 0.0
        if (a, b) in CONFUSABLE_PAIRS:
            return 0.24
        return 1.0

    @classmethod
    def _weighted_edit_distance(cls, a: str, b: str) -> float:
        if not a:
            return float(len(b))
        if not b:
            return float(len(a))
        n, m = len(a), len(b)
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = float(i)
        for j in range(1, m + 1):
            dp[0][j] = float(j)
        for i in range(1, n + 1):
            ca = a[i - 1]
            for j in range(1, m + 1):
                cb = b[j - 1]
                sub = dp[i - 1][j - 1] + cls._char_sub_cost(ca, cb)
                ins = dp[i][j - 1] + 1.0
                dele = dp[i - 1][j] + 1.0
                dp[i][j] = min(sub, ins, dele)
        return dp[n][m]

    @classmethod
    def _weighted_similarity(cls, a: str, b: str) -> float:
        base = max(1, max(len(a), len(b)))
        dist = cls._weighted_edit_distance(a, b)
        return max(0.0, 1.0 - (dist / float(base)))

    @staticmethod
    def _confusion_variants(token: str) -> set[str]:
        variants = {token}
        for a, b in CONFUSABLE_GROUPS:
            new_variants: set[str] = set()
            for w in variants:
                if a in w:
                    new_variants.add(w.replace(a, b))
                if b in w:
                    new_variants.add(w.replace(b, a))
            variants |= new_variants
        return variants

    def _best_match(self, token: str) -> str | None:
        token_n = self._norm(token)
        if token_n in self.lexicon:
            return None

        # Fast path for common OCR confusions such as y<->v, even for short tokens.
        for var in self._confusion_variants(token_n):
            if var in self.lexicon and var != token_n:
                return var

        if len(token_n) < self.cfg.min_word_len:
            return None
        if token_n.isdigit():
            return None
        if "'" in token or "’" in token:
            return None
        if "@" in token or "www." in token.lower() or "http" in token.lower():
            return None

        # Protect uppercase words/acronyms and mixed alnum codes.
        if token.isupper():
            return None
        if TR_ALPHA_RE.search(token) and re.search(r"\d", token):
            return None

        candidates = [w for w in self.lexicon if abs(len(w) - len(token_n)) <= 2]
        if not candidates:
            return None

        best = None
        best_ratio = 0.0
        second_ratio = 0.0
        for cand in candidates:
            ratio = self._weighted_similarity(token_n, cand)
            if ratio > best_ratio:
                second_ratio = best_ratio
                best_ratio = ratio
                best = cand
            elif ratio > second_ratio:
                second_ratio = ratio

        if best is None or best_ratio < self.cfg.similarity_threshold:
            return None
        if (best_ratio - second_ratio) < self.cfg.min_margin_over_second:
            return None
        return best

    def correct_text(self, text: str) -> str:
        if not self.cfg.enabled:
            return text
        changes = 0

        def repl(match: re.Match[str]) -> str:
            nonlocal changes
            token = match.group(0)
            if changes >= self.cfg.max_changes_per_block:
                return token
            suggestion = self._best_match(token)
            if suggestion is None:
                return token

            changes += 1
            if token.istitle():
                return suggestion.capitalize()
            return suggestion

        return WORD_RE.sub(repl, text)
