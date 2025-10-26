# src/ndvi_calculator.py
"""
NDVI Calculator
---------------
Compute NDVI from either:
  A) one multiband GeoTIFF (specify red/nir band indexes), or
  B) two separate single-band rasters (one red, one nir).

NDVI = (NIR - RED) / (NIR + RED)

Examples
--------
# Multiband (e.g., Sentinel-2 L2A GeoTIFF; RED=4, NIR=8)
python ndvi_calculator.py --multiband data/raw/sentinel.tif --red-band 4 --nir-band 8 \
  --out-tif data/processed/ndvi.tif --preview data/processed/ndvi_preview.png

# Landsat 8/9 multiband (RED=4, NIR=5)
python ndvi_calculator.py --multiband data/raw/landsat.tif --red-band 4 --nir-band 5 \
  --out-tif data/processed/ndvi.tif

# Separate files
python ndvi_calculator.py --red data/raw/red.tif --nir data/raw/nir.tif \
  --out-tif data/processed/ndvi.tif --preview data/processed/ndvi.png
"""

import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.transform import Affine
import warnings

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # preview optional


def _read_band(src: DatasetReader, band_index: int) -> np.ndarray:
    """Read a 1-based band as float32, applying the dataset mask."""
    arr = src.read(band_index).astype("float32")
    # Mask nodata/alpha
    mask = np.zeros(arr.shape, dtype=bool)
    # Try explicit nodata
    if src.nodata is not None:
        mask |= np.isclose(arr, src.nodata)
    # Try mask from dataset
    try:
        ds_mask = src.read_masks(band_index) == 0
        mask |= ds_mask
    except Exception:
        pass
    # Replace masked with NaN for safe math
    arr[mask] = np.nan
    return arr


def _resample_to_match(arr: np.ndarray, src: DatasetReader, ref: DatasetReader) -> np.ndarray:
    """If shapes differ, resample 'arr' from src grid to ref grid."""
    if src.width == ref.width and src.height == ref.height and src.transform == ref.transform:
        return arr
    # Build a temporary in-memory reproject via rasterio.vrt (lightweight) or read() with out_shape
    scale_x = src.width / ref.width
    scale_y = src.height / ref.height
    out = src.read(
        1,
        out_shape=(ref.height, ref.width),
        resampling=Resampling.bilinear
    ).astype("float32")
    # Apply nodata/mask again after resample
    if src.nodata is not None:
        out[np.isclose(out, src.nodata)] = np.nan
    return out


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI safely with NaN handling."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
    return ndvi.astype("float32")


def save_geotiff(reference_src: DatasetReader, ndvi: np.ndarray, out_path: Path, nodata_val: float = -9999.0):
    """Save NDVI as a single-band GeoTIFF using the reference dataset's georeferencing."""
    profile = reference_src.profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=nodata_val,
        compress="lzw"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Replace NaN with nodata value
    out = np.where(np.isnan(ndvi), nodata_val, ndvi).astype("float32")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)


def save_preview_png(ndvi: np.ndarray, preview_path: Path):
    """Save a quick PNG preview (requires matplotlib)."""
    if plt is None:
        warnings.warn("matplotlib not available; skipping preview.")
        return
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    # Clip NDVI to [-1, 1] and mask NaNs for display
    ndvi_disp = np.clip(ndvi, -1.0, 1.0)
    ndvi_disp = np.ma.masked_invalid(ndvi_disp)
    plt.figure(figsize=(6, 6))
    im = plt.imshow(ndvi_disp, vmin=-1, vmax=1)
    plt.title("NDVI")
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="NDVI")
    plt.tight_layout()
    plt.savefig(preview_path, dpi=180)
    plt.close()


def run_multiband(multiband_path: Path, red_band: int, nir_band: int, out_tif: Path, preview: Path | None):
    with rasterio.open(multiband_path) as src:
        red = _read_band(src, red_band)
        nir = _read_band(src, nir_band)
        ndvi = compute_ndvi(red, nir)
        save_geotiff(src, ndvi, out_tif)
        if preview:
            save_preview_png(ndvi, preview)


def run_two_files(red_path: Path, nir_path: Path, out_tif: Path, preview: Path | None):
    with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:
        red = _read_band(red_src, 1)
        nir = _read_band(nir_src, 1)
        # Align if needed (simple resample to red grid)
        if (nir_src.width, nir_src.height) != (red_src.width, red_src.height) or nir_src.transform != red_src.transform:
            nir = _resample_to_match(nir, nir_src, red_src)
        ndvi = compute_ndvi(red, nir)
        save_geotiff(red_src, ndvi, out_tif)
        if preview:
            save_preview_png(ndvi, preview)


def parse_args():
    p = argparse.ArgumentParser(description="Compute NDVI from satellite imagery.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--multiband", type=Path, help="Path to multiband GeoTIFF")
    g.add_argument("--red", type=Path, help="Path to single-band RED raster (GeoTIFF)")
    p.add_argument("--nir", type=Path, help="Path to single-band NIR raster (GeoTIFF); required if using --red")
    p.add_argument("--red-band", type=int, default=None, help="1-based RED band index for multiband")
    p.add_argument("--nir-band", type=int, default=None, help="1-based NIR band index for multiband")
    p.add_argument("--out-tif", type=Path, required=True, help="Output NDVI GeoTIFF path")
    p.add_argument("--preview", type=Path, default=None, help="Optional PNG preview output path")
    return p.parse_args()


def main():
    args = parse_args()

    if args.multiband:
        if args.red_band is None or args.nir_band is None:
            raise SystemExit("For --multiband you must provide --red-band and --nir-band (1-based).")
        run_multiband(args.multiband, args.red_band, args.nir_band, args.out_tif, args.preview)
    else:
        if args.nir is None:
            raise SystemExit("When using --red, you must also provide --nir.")
        run_two_files(args.red, args.nir, args.out_tif, args.preview)

    print(f"NDVI written to: {args.out_tif}")
    if args.preview:
        print(f"Preview saved to: {args.preview}")


if __name__ == "__main__":
    main()
