from src.config import TextCorrectionConfig
from src.pipeline.text_correction import TextCorrector


def test_text_corrector_fixes_close_word() -> None:
    cfg = TextCorrectionConfig(
        enabled=True,
        similarity_threshold=0.8,
        min_margin_over_second=0.01,
        min_word_len=4,
    )
    c = TextCorrector.from_config(cfg)
    out = c.correct_text("Turkiyede cocuklar icin harika hikayler var.")
    assert "hikayeler" in out.lower()


def test_text_corrector_preserves_numeric_codes() -> None:
    cfg = TextCorrectionConfig(enabled=True)
    c = TextCorrector.from_config(cfg)
    out = c.correct_text("Iletisim: 444 44 39")
    assert "444 44 39" in out


def test_text_corrector_preserves_uppercase_tokens() -> None:
    cfg = TextCorrectionConfig(enabled=True)
    c = TextCorrector.from_config(cfg)
    out = c.correct_text("IHLAS KOLEJI")
    assert out == "IHLAS KOLEJI"
