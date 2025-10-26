# tools/generate_synthetic_bands.py
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)

# Create synthetic Red and NIR rasters (100x100 pixels)
red = np.random.uniform(0.05, 0.40, (100, 100)).astype("float32")  # dimmer
nir = np.random.uniform(0.20, 0.80, (100, 100)).astype("float32")  # brighter

transform = from_origin(0, 100, 10, 10)  # arbitrary georeference
profile = {
    "driver": "GTiff",
    "dtype": "float32",
    "count": 1,
    "width": red.shape[1],
    "height": red.shape[0],
    "crs": "EPSG:4326",
    "transform": transform,
    "compress": "lzw",
}

for name, data in {"red": red, "nir": nir}.items():
    with rasterio.open(out_dir / f"{name}.tif", "w", **profile) as dst:
        dst.write(data, 1)

print("Wrote data/raw/red.tif and data/raw/nir.tif")