from src.config import VisualFilterConfig
from src.pipeline.visual_candidates import CandidateMetrics, keep_candidate


def test_small_visual_candidate_is_filtered() -> None:
    cfg = VisualFilterConfig(
        min_visual_area_ratio=0.0125,
        min_visual_short_side_ratio=0.05,
        min_visual_width_px=48,
        min_visual_height_px=48,
    )
    tiny = CandidateMetrics(
        bbox=(0, 0, 20, 20),
        area_ratio=0.001,
        short_side_ratio=0.02,
        aspect_ratio=1.0,
        edge_density=0.5,
        fill_ratio=0.9,
        text_overlap_ratio=0.0,
        entropy=5.0,
    )
    assert keep_candidate(tiny, cfg, page_size=(1000, 1000))[0] is False


def test_valid_visual_candidate_passes() -> None:
    cfg = VisualFilterConfig()
    c = CandidateMetrics(
        bbox=(10, 10, 220, 220),
        area_ratio=0.08,
        short_side_ratio=0.2,
        aspect_ratio=1.0,
        edge_density=0.12,
        fill_ratio=0.6,
        text_overlap_ratio=0.01,
        entropy=6.0,
    )
    assert keep_candidate(c, cfg, page_size=(1000, 1000))[0] is True
