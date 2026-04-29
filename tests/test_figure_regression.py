"""Perceptual-hash regression test for committed PNGs.

For every PNG in ``docs/figures/``, we compute :func:`imagehash.phash` and
compare against the reference stored in ``tests/figure_hashes.json``. A
Hamming distance ≤ 6 (out of 64 bits) is the green-light tolerance: this
catches accidental large changes (a wrong axis label, a missing curve,
swapped colour palette) without false-positives on the small font /
rasterisation differences that crop up between matplotlib versions.

If you intentionally change a figure, regenerate the hashes:

    python -c '
    from pathlib import Path
    import imagehash, json
    from PIL import Image
    hashes = {}
    for png in sorted(Path("docs/figures").glob("*.png")):
        with Image.open(png) as img:
            hashes[png.name] = str(imagehash.phash(img))
    Path("tests/figure_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True))
    '
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

imagehash = pytest.importorskip("imagehash")
PIL_Image = pytest.importorskip("PIL.Image")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "docs" / "figures"
REFERENCE_PATH = Path(__file__).resolve().parent / "figure_hashes.json"

HAMMING_TOLERANCE = 6


def _load_references() -> dict[str, str]:
    if not REFERENCE_PATH.exists():
        return {}
    return json.loads(REFERENCE_PATH.read_text())


def _figure_filenames() -> list[str]:
    refs = _load_references()
    if not refs:
        return []
    # Only test the figures we actually have reference hashes for.
    return [name for name in refs if (FIG_DIR / name).exists()]


@pytest.mark.parametrize("filename", _figure_filenames())
def test_figure_perceptual_hash(filename: str) -> None:
    refs = _load_references()
    expected = imagehash.hex_to_hash(refs[filename])
    with PIL_Image.open(FIG_DIR / filename) as img:
        got = imagehash.phash(img)
    distance = expected - got
    assert distance <= HAMMING_TOLERANCE, (
        f"perceptual hash drifted for {filename}: distance {distance} > "
        f"tolerance {HAMMING_TOLERANCE}. Re-render and (if intentional) update "
        f"tests/figure_hashes.json — see this module's docstring."
    )


def test_reference_file_present_and_complete() -> None:
    """Catch the case where a new PNG was committed without updating refs."""
    refs = _load_references()
    assert refs, f"missing reference hashes file at {REFERENCE_PATH}"
    pngs = {p.name for p in FIG_DIR.glob("*.png")}
    missing = pngs - refs.keys()
    assert not missing, (
        f"PNGs without reference hashes: {sorted(missing)}. Regenerate "
        f"tests/figure_hashes.json (see this module's docstring)."
    )
