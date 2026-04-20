"""
Project: Agricultural Image Analysis for Crop Segmentation and Classification
Description: Advanced preprocessing pipeline for agricultural drone imagery. 
             Features global normalization, NaN skipping, and detailed real-time logging.
Author: Juan Pablo González Blandón
University: Universidad de Antioquia (UdeA)
"""

import cv2
import numpy as np
import os
import re
from pathlib import Path
from tqdm import tqdm

# --- PATH CONFIGURATION ---
BASE_PATH = Path(__file__).resolve().parent.parent / "data"
RGB_DIR = BASE_PATH / "raw_data" / "rgb-images"
NIR_DIR = BASE_PATH / "raw_data" / "multispectral-images" / "NIR"
RED_DIR = BASE_PATH / "raw_data" / "multispectral-images" / "RED"
REG_DIR = BASE_PATH / "raw_data" / "multispectral-images" / "REG"

OUTPUT_DIR = BASE_PATH / "prepro_dataset_classifier"

# --- HYPERPARAMETERS ---
TILE_SIZE = 224
OVERLAP = 32           
HEALTHY_THRESHOLD = 0.5
STRESSED_THRESHOLD = 0.3 
BRIGHTNESS_SHADOW = 25

def prepare_folders():
    """Creates the directory structure for the classified dataset."""
    categories = ['1_Crop_Healthy', '2_Crop_Stressed', '3_Potential_Soil_Path', '5_Shadow']
    for cat in categories:
        (OUTPUT_DIR / cat).mkdir(parents=True, exist_ok=True)

def normalize_image(image):
    """Normalizes an image/band to [0, 1] range based on its min/max values."""
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min == 0:
        return image.astype(np.float32)
    return (image.astype(np.float32) - img_min) / (img_max - img_min)

def process_dataset():
    prepare_folders()
    
    extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    
    # 1. READ FILES
    rgb_files = [f for f in os.listdir(RGB_DIR) if f.lower().endswith(extensions)]
    nir_files = [f for f in os.listdir(NIR_DIR) if f.lower().endswith(extensions)]
    red_files = [f for f in os.listdir(RED_DIR) if f.lower().endswith(extensions)]
    reg_files = [f for f in os.listdir(REG_DIR) if f.lower().endswith(extensions)]

    # 2. MAPPING LOGIC
    rgb_dict = {Path(f).stem[-4:]: f for f in rgb_files}
    red_dict = {re.search(r'(\d{4})_RED', f).group(1): f for f in red_files if re.search(r'(\d{4})_RED', f)}
    nir_dict = {re.search(r'(\d{4})_NIR', f).group(1): f for f in nir_files if re.search(r'(\d{4})_NIR', f)}
    reg_dict = {re.search(r'(\d{4})_REG', f).group(1): f for f in reg_files if re.search(r'(\d{4})_REG', f)}

    # 3. MATCHING
    common_indices = sorted(set(rgb_dict.keys()) & set(nir_dict.keys()) & set(red_dict.keys()) & set(reg_dict.keys()))
    
    print(f"--- Agricultural Preprocessing Pipeline ---")
    print(f"Output directory: {OUTPUT_DIR.resolve()}\n")

    for idx in tqdm(common_indices, desc="Processing Images"):
        # Load images
        img_rgb = cv2.imread(str(RGB_DIR / rgb_dict[idx]))
        img_nir = cv2.imread(str(NIR_DIR / nir_dict[idx]), cv2.IMREAD_UNCHANGED)
        img_red = cv2.imread(str(RED_DIR / red_dict[idx]), cv2.IMREAD_UNCHANGED)
        img_reg = cv2.imread(str(REG_DIR / reg_dict[idx]), cv2.IMREAD_UNCHANGED)

        if any(v is None for v in [img_rgb, img_nir, img_red, img_reg]):
            continue

        # --- GLOBAL NORMALIZATION ---
        img_nir_norm = normalize_image(img_nir)
        img_red_norm = normalize_image(img_red)
        img_reg_norm = normalize_image(img_reg)

        h, w, _ = img_rgb.shape
        step = TILE_SIZE - OVERLAP
        tile_count = 0

        # Sliding window
        for y in range(0, h - TILE_SIZE + 1, step):
            for x in range(0, w - TILE_SIZE + 1, step):
                
                # Extract tile
                tile_rgb = img_rgb[y:y+TILE_SIZE, x:x+TILE_SIZE]
                t_nir = img_nir_norm[y:y+TILE_SIZE, x:x+TILE_SIZE]
                t_red = img_red_norm[y:y+TILE_SIZE, x:x+TILE_SIZE]
                t_reg = img_reg_norm[y:y+TILE_SIZE, x:x+TILE_SIZE]

                # --- SAFE NDVI CALCULATION ---
                denom = t_nir + t_red
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi = (t_nir - t_red) / denom
                    ndvi_mean = np.nanmean(ndvi)
                
                # --- SKIP NAN DATA ---
                if np.isnan(ndvi_mean) or np.all(denom == 0):
                    continue

                tile_fn = f"{idx}_y{y}_x{x}.png"
                
                # --- CLASSIFICATION LOGIC ---
                if np.mean(tile_rgb) < BRIGHTNESS_SHADOW:
                    folder = '5_Shadow'
                elif ndvi_mean > HEALTHY_THRESHOLD:
                    folder = '1_Crop_Healthy'
                elif ndvi_mean > STRESSED_THRESHOLD or np.mean(t_reg) > np.mean(t_red) * 1.15:
                    folder = '2_Crop_Stressed'
                else:
                    folder = '3_Potential_Soil_Path'

                # --- SAVE AND DETAILED LOGGING ---
                save_path = OUTPUT_DIR / folder / tile_fn
                cv2.imwrite(str(save_path), tile_rgb)
                
                # This is the line you requested to see the math and the path:
                print(f"[SAVED] Index: {idx} | Tile: [{y},{x}] | NDVI: {ndvi_mean:.4f} -> {save_path.resolve()}")
                
                tile_count += 1
        
        print(f"[DONE] Index {idx} processed. Valid tiles saved: {tile_count}")

if __name__ == "__main__":
    process_dataset()
    print(f"\nPipeline complete. Review your classified tiles in: {OUTPUT_DIR}")