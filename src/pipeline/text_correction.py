from __future__ import annotations

from dataclasses import dataclass
import difflib
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
}


WORD_RE = re.compile(r"\b[\wÇĞİÖŞÜçğıöşü]+\b", flags=re.UNICODE)


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

    def _best_match(self, token: str) -> str | None:
        token_n = self._norm(token)
        if token_n in self.lexicon:
            return None

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
        if re.search(r"[A-Za-zÇĞİÖŞÜçğıöşü]", token) and re.search(r"\d", token):
            return None

        candidates = [w for w in self.lexicon if abs(len(w) - len(token_n)) <= 2]
        if not candidates:
            return None

        best = None
        best_ratio = 0.0
        second_ratio = 0.0
        for cand in candidates:
            ratio = difflib.SequenceMatcher(None, token_n, cand).ratio()
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
