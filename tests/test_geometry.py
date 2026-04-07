from src.utils.geometry import denormalize_bbox_1000, normalize_bbox_1000


def test_bbox_normalize_denormalize_roundtrip() -> None:
    width, height = 2000, 3000
    bbox = (123, 456, 1789, 2890)
    norm = normalize_bbox_1000(bbox, width, height)
    restored = denormalize_bbox_1000(norm, width, height)

    for a, b in zip(bbox, restored):
        assert abs(a - b) <= 3
