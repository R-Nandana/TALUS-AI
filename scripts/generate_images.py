"""
TALUS - Rock & Soil Texture Image Generator
===========================================
Creates a labelled image dataset of mining-related geological materials under
data/raw/images/<material>/...

IMPORTANT: these are PROCEDURALLY GENERATED synthetic texture images (numpy +
scipy), NOT photographs. Each class imitates the visual signature of a material
(grain speckle, bedding, laminations, desiccation cracks, angular clasts...).
Moisture state (dry / wet) and lighting are randomised per image.

To use REAL photos instead, drop them into data/raw/images/<material>/ (any .jpg/.png)
and re-run  python scripts/preprocessing.py  - the pipeline reads whatever is there.

Usage:  python scripts/generate_images.py [images_per_class]
"""
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT_DIR, "data", "raw", "images")
META_PATH = os.path.join(ROOT_DIR, "data", "raw", "image_metadata.csv")
S = 128  # image size in pixels

# ----------------------------------------------------------------- helpers
def noise(rng, sigma):
    n = ndi.gaussian_filter(rng.standard_normal((S, S)), sigma, mode="wrap")
    return (n - n.mean()) / (n.std() + 1e-9)

def fbm(rng, sigmas, weights):
    out = sum(w * noise(rng, s) for s, w in zip(sigmas, weights))
    return out / (np.std(out) + 1e-9)

def n01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)

def colorize(g, dark, light):
    dark, light = np.array(dark, float), np.array(light, float)
    return dark + g[..., None] * (light - dark)

def jitter(rng, color, amt=0.02):
    """Natural variation: scale overall brightness, add only a slight tint (keeps colours realistic)."""
    return np.clip(np.array(color, float) * rng.uniform(0.88, 1.12) + rng.normal(0, amt, 3), 0, 1)

def voronoi(rng, n_seeds):
    seeds = rng.uniform(0, S, (n_seeds, 2))
    yy, xx = np.mgrid[0:S, 0:S]
    pts = np.stack([yy.ravel(), xx.ravel()], 1)
    d, idx = cKDTree(seeds).query(pts, k=2)
    return (idx[:, 0].reshape(S, S), d[:, 0].reshape(S, S), (d[:, 1] - d[:, 0]).reshape(S, S))

# ----------------------------------------------------------------- materials
def granite(rng):
    g = n01(fbm(rng, (1.0, 2.5, 7), (0.5, 0.35, 0.15)))
    img = colorize(g, jitter(rng, (0.55, 0.50, 0.50)), jitter(rng, (0.86, 0.78, 0.76)))
    specks = noise(rng, 0.9) > rng.uniform(1.3, 1.6)          # dark mica / hornblende
    img[specks] *= 0.25
    return img

def basalt(rng):
    g = n01(fbm(rng, (0.8, 2.0, 6), (0.5, 0.3, 0.2)))
    img = colorize(g, jitter(rng, (0.12, 0.12, 0.13), .01), jitter(rng, (0.38, 0.38, 0.40), .01))
    yy, xx = np.mgrid[0:S, 0:S]
    for _ in range(rng.integers(6, 16)):                        # vesicles (gas pits)
        cy, cx, r = rng.uniform(0, S), rng.uniform(0, S), rng.uniform(1.5, 4)
        img *= (1 - 0.6 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r))))[..., None]
    return img

def sandstone(rng):
    yy, xx = np.mgrid[0:S, 0:S]
    th = rng.uniform(-0.25, 0.25)
    warp = 3 * noise(rng, 10)
    u = xx * np.sin(th) + yy * np.cos(th) + warp
    period = rng.uniform(9, 18)
    bands = 0.5 + 0.5 * np.sin(2 * np.pi * u / period + 2 * noise(rng, 15))
    g = n01(0.55 * bands + 0.3 * noise(rng, 0.8) + 0.15 * noise(rng, 3))
    return colorize(g, jitter(rng, (0.62, 0.42, 0.24)), jitter(rng, (0.90, 0.76, 0.55)))

def limestone(rng):
    g = n01(fbm(rng, (2, 6, 14), (0.3, 0.4, 0.3)))
    img = colorize(g, jitter(rng, (0.72, 0.70, 0.62)), jitter(rng, (0.95, 0.93, 0.85)))
    veins = np.exp(-(noise(rng, 7) / 0.10) ** 2)                # thin calcite veins / stylolites
    img *= (1 - 0.45 * veins)[..., None]
    img *= (1 - 0.35 * (noise(rng, 1.2) > 1.9))[..., None]     # small pits
    return img

def shale(rng):
    yy, xx = np.mgrid[0:S, 0:S]
    th = rng.uniform(-0.12, 0.12)
    u = yy * np.cos(th) + xx * np.sin(th) + 2.5 * noise(rng, 8)
    lam = 0.5 + 0.5 * np.sin(2 * np.pi * u / rng.uniform(3.0, 5.5))
    g = n01(0.6 * lam + 0.25 * noise(rng, 1.0) + 0.15 * noise(rng, 5))
    img = colorize(g, jitter(rng, (0.13, 0.13, 0.14), .02), jitter(rng, (0.42, 0.40, 0.38)))
    cr = np.exp(-(noise(rng, 3) / 0.06) ** 2)                   # fissility cracks
    return img * (1 - 0.5 * cr)[..., None]

def clay_soil(rng):
    g = n01(fbm(rng, (3, 8, 18), (0.4, 0.4, 0.2)) + 0.25 * noise(rng, 0.8))
    img = colorize(g, jitter(rng, (0.38, 0.22, 0.14)), jitter(rng, (0.62, 0.38, 0.25)))
    _, _, edge = voronoi(rng, rng.integers(18, 40))             # desiccation polygons
    crack = np.clip(1 - edge / 1.6, 0, 1)
    return img * (1 - 0.7 * crack)[..., None]

def sandy_soil(rng):
    g = n01(0.7 * noise(rng, 0.7) + 0.3 * noise(rng, 2.5))
    return colorize(g, jitter(rng, (0.55, 0.42, 0.25)), jitter(rng, (0.86, 0.72, 0.48)))

def gravel(rng):
    cell, dist, edge = voronoi(rng, rng.integers(55, 110))
    palette = [(0.50, 0.48, 0.45), (0.42, 0.38, 0.33), (0.58, 0.52, 0.44), (0.36, 0.36, 0.38), (0.55, 0.42, 0.34)]
    base = np.stack([np.array(palette[rng.integers(len(palette))]) * rng.uniform(0.7, 1.25)
                     for _ in range(cell.max() + 1)])
    img = base[cell]
    img *= (0.65 + 0.45 * (1 - np.clip(dist / 11, 0, 1)))[..., None]   # rounded shading
    img *= (0.25 + 0.75 * np.clip(edge / 2.2, 0, 1))[..., None]         # dark voids between clasts
    img *= (0.9 + 0.1 * noise(rng, 0.8))[..., None]
    return img

MATERIALS = {   # name -> (generator, material type)
    "Granite": (granite, "Rock"), "Basalt": (basalt, "Rock"), "Sandstone": (sandstone, "Rock"),
    "Limestone": (limestone, "Rock"), "Shale": (shale, "Rock"),
    "Clay": (clay_soil, "Soil"), "Sandy Soil": (sandy_soil, "Soil"), "Gravel Colluvium": (gravel, "Soil"),
}

def finish(rng, img, wet):
    yy, xx = np.mgrid[0:S, 0:S] / S
    a, b = rng.uniform(-0.25, 0.25, 2)
    light = 1 + a * (xx - .5) + b * (yy - .5)                   # illumination gradient
    img = img * light[..., None] * rng.uniform(0.85, 1.15)
    if wet:                                                     # wet material: darker, more saturated, less contrast
        m = img.mean(axis=2, keepdims=True)
        img = (m + (img - m) * 1.15) * 0.68
        img = 0.5 * img + 0.5 * ndi.gaussian_filter(img, (1.0, 1.0, 0))
    img = img + rng.normal(0, 0.012, img.shape)                 # sensor noise
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def main(n_per_class=40, seed=2026):
    rng = np.random.default_rng(seed)
    rows = []
    for name, (fn, mtype) in MATERIALS.items():
        slug = name.lower().replace(" ", "_")
        os.makedirs(os.path.join(IMG_DIR, slug), exist_ok=True)
        for k in range(n_per_class):
            wet = bool(rng.random() < 0.35)
            arr = finish(rng, fn(rng), wet)
            rel = f"images/{slug}/{slug}_{k + 1:03d}.jpg"
            Image.fromarray(arr).save(os.path.join(ROOT_DIR, "data", "raw", rel), quality=92)
            rows.append({"image_id": f"{slug}_{k + 1:03d}", "filepath": rel, "material": name,
                         "material_type": mtype, "moisture_state": "wet" if wet else "dry",
                         "source": "PROCEDURAL/SYNTHETIC"})
    pd.DataFrame(rows).to_csv(META_PATH, index=False)
    print(f"Generated {len(rows)} images in {len(MATERIALS)} classes -> {IMG_DIR}")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
