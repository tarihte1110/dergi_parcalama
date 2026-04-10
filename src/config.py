from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Type, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class OCRConfig:
    backend: str = "surya"
    min_confidence: float = 0.45
    paddle_use_angle_cls: bool = True
    paddle_lang: str = "tr"
    surya_device: str = "mps"
    surya_detection_batch_size: int = 4
    surya_recognition_batch_size: int = 12
    surya_min_batch_size: int = 1
    surya_enable_adaptive_batch: bool = True
    surya_page_batch_size: int = 3
    surya_suppress_internal_progress: bool = True
    surya_highres_scale: float = 1.35
    surya_highres_max_long_edge: int = 2800


@dataclass(frozen=True)
class GroupingConfig:
    max_vertical_gap_ratio: float = 1.6
    max_block_vertical_gap_ratio: float = 2.1
    max_line_height_ratio: float = 1.9
    min_horizontal_overlap_ratio: float = 0.08
    max_alignment_delta_ratio: float = 1.8
    max_reading_flow_dx_ratio: float = 7.0
    barrier_gap_ratio: float = 2.4
    barrier_edge_density: float = 0.08
    absorb_short_text_max_chars: int = 18
    absorb_max_center_distance_ratio: float = 3.0
    line_column_align_ratio: float = 1.4
    block_row_band_ratio: float = 0.9


@dataclass(frozen=True)
class ClassificationConfig:
    headline_height_ratio: float = 1.38
    headline_max_lines: int = 3
    headline_max_chars: int = 140
    content_min_lines: int = 3
    content_height_upper_ratio: float = 1.25
    content_min_chars: int = 60
    pair_max_vertical_ratio: float = 0.35
    headline_max_digit_ratio: float = 0.45
    headline_min_alpha_chars: int = 5
    headline_content_min_overlap: float = 0.15
    prefix_split_min_total_lines: int = 4
    prefix_headline_height_ratio: float = 1.16
    prefix_headline_gap_ratio: float = 1.6
    prefix_headline_gap_height_ratio: float = 0.8
    prefix_headline_width_ratio_max: float = 0.9
    prefix_title_pattern_max_chars: int = 80
    prefix_title_pattern_min_rest_chars: int = 80


@dataclass(frozen=True)
class MaskConfig:
    text_dilation_ratio: float = 0.005
    non_text_open_ratio: float = 0.003
    non_text_close_ratio: float = 0.006
    white_threshold: int = 242
    low_saturation_threshold: int = 20
    black_threshold: int = 18
    background_std_threshold: float = 8.0
    visual_texture_std_threshold: float = 12.0
    visual_color_saturation_threshold: int = 34


@dataclass(frozen=True)
class VisualFilterConfig:
    min_visual_area_ratio: float = 0.0065
    min_visual_short_side_ratio: float = 0.05
    min_visual_width_px: int = 48
    min_visual_height_px: int = 48
    max_aspect_ratio: float = 12.0
    min_edge_density: float = 0.006
    min_fill_ratio: float = 0.09
    merge_iou_threshold: float = 0.25
    max_visual_area_ratio: float = 0.82
    edge_touch_reject_area_ratio: float = 0.42
    edge_touch_rescue_min_edge_density: float = 0.01
    edge_touch_rescue_min_entropy: float = 3.3
    edge_touch_rescue_max_area_ratio: float = 0.35
    max_text_overlap_ratio: float = 0.22
    text_heavy_rescue_min_area_ratio: float = 0.012
    text_heavy_rescue_min_edge_density: float = 0.012
    text_heavy_rescue_min_entropy: float = 3.4
    small_region_high_quality_area_ratio: float = 0.015
    small_region_min_edge_density: float = 0.01
    small_region_min_entropy: float = 3.0
    min_entropy: float = 2.0
    split_trigger_area_ratio: float = 0.75
    split_min_component_area_ratio: float = 0.0035
    split_texture_std_threshold: float = 16.0
    split_saturation_threshold: int = 32
    split_seed_open_ratio: float = 0.002
    split_seed_close_ratio: float = 0.004
    grid_split_rows: int = 3
    grid_split_cols: int = 2
    grid_split_expand_ratio: float = 0.02
    grid_min_non_text_ratio: float = 0.22
    grid_min_visual_score: float = 0.42
    grid_component_min_cell_area_ratio: float = 0.02
    grid_strip_span_ratio: float = 0.9
    grid_strip_other_ratio: float = 0.55
    fullpage_rescue_min_edge_density: float = 0.03
    fullpage_rescue_min_entropy: float = 4.3
    fullpage_rescue_max_white_ratio: float = 0.22
    fullpage_rescue_max_text_ratio: float = 0.16
    fragment_merge_gap_ratio: float = 0.018
    fragment_merge_min_cross_overlap: float = 0.15
    fragment_merge_max_union_area_ratio: float = 0.42
    strip_span_ratio: float = 0.9
    strip_other_ratio: float = 0.55
    strip_projection_min_fill_ratio: float = 0.06
    strip_projection_min_run_ratio: float = 0.08
    strip_projection_pad_ratio: float = 0.01
    fallback_texture_std_threshold: float = 18.0
    fallback_saturation_threshold: int = 36
    fallback_edge_open_ratio: float = 0.0018
    fallback_edge_close_ratio: float = 0.0035
    fallback_min_component_area_ratio: float = 0.0024
    large_box_split_open_ratio: float = 0.0015
    large_box_split_close_ratio: float = 0.003
    large_box_split_min_component_area_ratio: float = 0.015
    panel_merge_gap_ratio: float = 0.02
    panel_merge_min_cross_overlap: float = 0.08
    panel_merge_max_union_area_ratio: float = 0.72
    panel_drop_small_inside_ratio: float = 0.8
    panel_text_heavy_drop_ratio: float = 0.48
    panel_text_heavy_max_edge_density: float = 0.018
    panel_min_area_ratio: float = 0.009
    small_textlike_max_area_ratio: float = 0.08
    small_textlike_max_entropy: float = 4.5
    small_textlike_max_edge_density: float = 0.03
    half_rescue_min_side_area_ratio: float = 0.08
    half_rescue_component_area_ratio: float = 0.008
    half_rescue_texture_std_threshold: float = 11.0
    half_rescue_saturation_threshold: int = 28


@dataclass(frozen=True)
class CropConfig:
    crop_padding_ratio: float = 0.01
    crop_padding_px: int = 6


@dataclass(frozen=True)
class RuntimeConfig:
    input_dir: str = "images"
    output_dir: str = "outputs"
    debug: bool = True
    write_debug_json: bool = True
    suppress_progress_bars: bool = True


@dataclass(frozen=True)
class TextCorrectionConfig:
    enabled: bool = True
    min_word_len: int = 4
    similarity_threshold: float = 0.93
    min_margin_over_second: float = 0.06
    max_changes_per_block: int = 4
    lexicon_path: str = ""


@dataclass(frozen=True)
class AppConfig:
    ocr: OCRConfig = OCRConfig()
    grouping: GroupingConfig = GroupingConfig()
    classification: ClassificationConfig = ClassificationConfig()
    mask: MaskConfig = MaskConfig()
    visual: VisualFilterConfig = VisualFilterConfig()
    crop: CropConfig = CropConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    text_correction: TextCorrectionConfig = TextCorrectionConfig()


def _merge_dataclass(dc_type: Type[T], data: dict[str, Any]) -> T:
    allowed = {f.name for f in fields(dc_type)}
    payload = {k: v for k, v in data.items() if k in allowed}
    return dc_type(**payload)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    if config_path is None:
        return AppConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    default = AppConfig()
    return AppConfig(
        ocr=_merge_dataclass(OCRConfig, raw.get("ocr", asdict(default.ocr))),
        grouping=_merge_dataclass(GroupingConfig, raw.get("grouping", asdict(default.grouping))),
        classification=_merge_dataclass(
            ClassificationConfig,
            raw.get("classification", asdict(default.classification)),
        ),
        mask=_merge_dataclass(MaskConfig, raw.get("mask", asdict(default.mask))),
        visual=_merge_dataclass(VisualFilterConfig, raw.get("visual", asdict(default.visual))),
        crop=_merge_dataclass(CropConfig, raw.get("crop", asdict(default.crop))),
        runtime=_merge_dataclass(RuntimeConfig, raw.get("runtime", asdict(default.runtime))),
        text_correction=_merge_dataclass(
            TextCorrectionConfig,
            raw.get("text_correction", asdict(default.text_correction)),
        ),
    )
