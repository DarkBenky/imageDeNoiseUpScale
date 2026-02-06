import asyncio
import aiohttp
import numpy as np
import json
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import os
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

def randomize_config_value(value, min_factor=0.5, max_factor=1.5):
    """Randomize a config value within a range (default ±50%)"""
    if isinstance(value, (int, float)) and value != 0:
        return value * np.random.uniform(min_factor, max_factor)
    return value

# SAVE_PATH = "/media/user/2TB/imageData"
# SAVE_PATH = "/media/user/2TB Clear/imageData"
SAVE_PATH = "/media/user/HDD 1TB/Data"
# SAVE_PATH = "/media/user/128GB"
SAVE_PERIOD = 1024
IMAGE_WIDTH_LOW_RES = 800
IMAGE_HEIGHT_LOW_RES = 600
IMAGE_WIDTH_HIGH_RES = 800
IMAGE_HEIGHT_HIGH_RES = 600

MAX_CONCURRENT_DOWNLOADS = 256
DOWNLOAD_TIMEOUT = 3
MAX_PROCESSING_WORKERS = int(multiprocessing.cpu_count() * 0.75)

MAX_NOISE_APPLICATIONS = 5

# Ray tracing noise configurations
# Ray tracing primarily produces Poisson noise from Monte Carlo sampling
# with varying intensity across the image (some regions need more samples)
RAY_TRACING_NOISE_CONFIGS = [
    # Low sample count - more noise (like 16-64 samples per pixel in ray tracer)
    {"type": "ray_tracing", "intensity": "very_low_samples", "base_scale": 0.15, "gaussian_layer": 0.02, "num_noisy_regions": 2},
    {"type": "ray_tracing", "intensity": "low_samples", "base_scale": 0.25, "gaussian_layer": 0.03, "num_noisy_regions": 3},
    {"type": "ray_tracing", "intensity": "medium_samples", "base_scale": 0.4, "gaussian_layer": 0.04, "num_noisy_regions": 3},
    {"type": "ray_tracing", "intensity": "high_noise", "base_scale": 0.5, "gaussian_layer": 0.05, "num_noisy_regions": 4},
    {"type": "ray_tracing", "intensity": "very_high_noise", "base_scale": 0.6, "gaussian_layer": 0.06, "num_noisy_regions": 4},
    {"type": "ray_tracing", "intensity": "extreme_noise", "base_scale": 0.8, "gaussian_layer": 0.08, "num_noisy_regions": 5},
    
    # With varying noise across color channels (common in ray tracing)
    {"type": "ray_tracing", "intensity": "low_samples_color_var", "base_scale": 0.25, "gaussian_layer": 0.03, "num_noisy_regions": 2, "color_variance": True},
    {"type": "ray_tracing", "intensity": "medium_samples_color_var", "base_scale": 0.4, "gaussian_layer": 0.04, "num_noisy_regions": 3, "color_variance": True},
    {"type": "ray_tracing", "intensity": "high_noise_color_var", "base_scale": 0.5, "gaussian_layer": 0.05, "num_noisy_regions": 4, "color_variance": True},
    {"type": "ray_tracing", "intensity": "extreme_noise_color_var", "base_scale": 0.7, "gaussian_layer": 0.07, "num_noisy_regions": 5, "color_variance": True},
    
    # Heavy localized noise (like difficult indirect lighting areas)
    {"type": "ray_tracing", "intensity": "difficult_regions_low", "base_scale": 0.2, "gaussian_layer": 0.02, "num_noisy_regions": 4, "difficult_scale": 0.6},
    {"type": "ray_tracing", "intensity": "difficult_regions_medium", "base_scale": 0.3, "gaussian_layer": 0.03, "num_noisy_regions": 5, "difficult_scale": 0.5},
    {"type": "ray_tracing", "intensity": "difficult_regions_high", "base_scale": 0.4, "gaussian_layer": 0.04, "num_noisy_regions": 6, "difficult_scale": 0.4},
    {"type": "ray_tracing", "intensity": "difficult_regions_extreme", "base_scale": 0.5, "gaussian_layer": 0.05, "num_noisy_regions": 7, "difficult_scale": 0.3},
    
    # Combined color variance + difficult regions
    {"type": "ray_tracing", "intensity": "realistic_low", "base_scale": 0.25, "gaussian_layer": 0.03, "num_noisy_regions": 3, "color_variance": True, "difficult_scale": 0.6},
    {"type": "ray_tracing", "intensity": "realistic_medium", "base_scale": 0.35, "gaussian_layer": 0.04, "num_noisy_regions": 4, "color_variance": True, "difficult_scale": 0.5},
    {"type": "ray_tracing", "intensity": "realistic_high", "base_scale": 0.45, "gaussian_layer": 0.05, "num_noisy_regions": 5, "color_variance": True, "difficult_scale": 0.4},
    {"type": "ray_tracing", "intensity": "realistic_extreme", "base_scale": 0.6, "gaussian_layer": 0.06, "num_noisy_regions": 6, "color_variance": True, "difficult_scale": 0.3},
    
    # Very fine grain (high sample count but still some noise)
    {"type": "ray_tracing", "intensity": "fine_grain_subtle", "base_scale": 0.05, "gaussian_layer": 0.01, "num_noisy_regions": 1},
    {"type": "ray_tracing", "intensity": "fine_grain_low", "base_scale": 0.1, "gaussian_layer": 0.015, "num_noisy_regions": 2},
    {"type": "ray_tracing", "intensity": "fine_grain_medium", "base_scale": 0.15, "gaussian_layer": 0.02, "num_noisy_regions": 2},
    
    # Ultra extreme noise (like 1-8 samples per pixel - very early ray tracer previews)
    {"type": "ray_tracing", "intensity": "ultra_extreme", "base_scale": 0.9, "gaussian_layer": 0.1, "num_noisy_regions": 6},
    {"type": "ray_tracing", "intensity": "ultra_extreme_color_var", "base_scale": 0.9, "gaussian_layer": 0.1, "num_noisy_regions": 7, "color_variance": True},
    {"type": "ray_tracing", "intensity": "ultra_extreme_difficult", "base_scale": 0.85, "gaussian_layer": 0.12, "num_noisy_regions": 8, "difficult_scale": 0.15},
    {"type": "ray_tracing", "intensity": "ultra_extreme_realistic", "base_scale": 0.8, "gaussian_layer": 0.1, "num_noisy_regions": 7, "color_variance": True, "difficult_scale": 0.2},
    
    # Insane noise levels (like 1-4 samples per pixel)
    {"type": "ray_tracing", "intensity": "insane", "base_scale": 0.95, "gaussian_layer": 0.15, "num_noisy_regions": 8},
    {"type": "ray_tracing", "intensity": "insane_color_var", "base_scale": 0.95, "gaussian_layer": 0.15, "num_noisy_regions": 9, "color_variance": True},
    {"type": "ray_tracing", "intensity": "insane_difficult", "base_scale": 0.9, "gaussian_layer": 0.18, "num_noisy_regions": 10, "difficult_scale": 0.1},
    {"type": "ray_tracing", "intensity": "insane_realistic", "base_scale": 0.9, "gaussian_layer": 0.15, "num_noisy_regions": 9, "color_variance": True, "difficult_scale": 0.12},
    
    # Maximum chaos (barely recognizable image - extreme low sample counts)
    {"type": "ray_tracing", "intensity": "maximum_chaos", "base_scale": 0.98, "gaussian_layer": 0.2, "num_noisy_regions": 12},
    {"type": "ray_tracing", "intensity": "maximum_chaos_color_var", "base_scale": 0.98, "gaussian_layer": 0.2, "num_noisy_regions": 12, "color_variance": True},
    {"type": "ray_tracing", "intensity": "maximum_chaos_realistic", "base_scale": 0.95, "gaussian_layer": 0.22, "num_noisy_regions": 15, "color_variance": True, "difficult_scale": 0.08},

    # Extreme localized noise (like caustics or very difficult lighting scenarios)
    {"type": "ray_tracing", "intensity": "extreme_localized", "base_scale": 0.4, "gaussian_layer": 0.05, "num_noisy_regions": 8, "difficult_scale": 0.25},
    {"type": "ray_tracing", "intensity": "extreme_localized_color_var", "base_scale": 0.4, "gaussian_layer": 0.05, "num_noisy_regions": 8, "color_variance": True, "difficult_scale": 0.25},
    {"type": "ray_tracing", "intensity": "ultra_localized", "base_scale": 0.5, "gaussian_layer": 0.06, "num_noisy_regions": 10, "difficult_scale": 0.2},
    {"type": "ray_tracing", "intensity": "ultra_localized_realistic", "base_scale": 0.5, "gaussian_layer": 0.06, "num_noisy_regions": 12, "color_variance": True, "difficult_scale": 0.18},
    {"type": "ray_tracing", "intensity": "insane_localized", "base_scale": 0.7, "gaussian_layer": 0.08, "num_noisy_regions": 14, "difficult_scale": 0.15},
    {"type": "ray_tracing", "intensity": "insane_localized_realistic", "base_scale": 0.7, "gaussian_layer": 0.08, "num_noisy_regions": 16, "color_variance": True, "difficult_scale": 0.12},
    
    # Beyond maximum - absolute worst case scenarios
    {"type": "ray_tracing", "intensity": "apocalyptic", "base_scale": 0.99, "gaussian_layer": 0.25, "num_noisy_regions": 15},
    {"type": "ray_tracing", "intensity": "apocalyptic_color_var", "base_scale": 0.99, "gaussian_layer": 0.25, "num_noisy_regions": 16, "color_variance": True},
    {"type": "ray_tracing", "intensity": "apocalyptic_realistic", "base_scale": 0.98, "gaussian_layer": 0.28, "num_noisy_regions": 18, "color_variance": True, "difficult_scale": 0.05},
    
    # Pure chaos - nearly impossible to denoise
    {"type": "ray_tracing", "intensity": "absolute_chaos", "base_scale": 0.995, "gaussian_layer": 0.3, "num_noisy_regions": 20},
    {"type": "ray_tracing", "intensity": "absolute_chaos_color_var", "base_scale": 0.995, "gaussian_layer": 0.3, "num_noisy_regions": 20, "color_variance": True},
    {"type": "ray_tracing", "intensity": "absolute_chaos_realistic", "base_scale": 0.99, "gaussian_layer": 0.35, "num_noisy_regions": 25, "color_variance": True, "difficult_scale": 0.03},
    
    # Mixed intensity levels (varied noise across different noise levels)
    {"type": "ray_tracing", "intensity": "mixed_low_high", "base_scale": 0.3, "gaussian_layer": 0.04, "num_noisy_regions": 6, "difficult_scale": 0.4},
    {"type": "ray_tracing", "intensity": "mixed_medium_extreme", "base_scale": 0.5, "gaussian_layer": 0.06, "num_noisy_regions": 8, "difficult_scale": 0.25},
    {"type": "ray_tracing", "intensity": "mixed_high_insane", "base_scale": 0.7, "gaussian_layer": 0.09, "num_noisy_regions": 10, "difficult_scale": 0.15},
    {"type": "ray_tracing", "intensity": "mixed_realistic_low", "base_scale": 0.35, "gaussian_layer": 0.045, "num_noisy_regions": 7, "color_variance": True, "difficult_scale": 0.35},
    {"type": "ray_tracing", "intensity": "mixed_realistic_high", "base_scale": 0.65, "gaussian_layer": 0.075, "num_noisy_regions": 9, "color_variance": True, "difficult_scale": 0.2},
    
    # Heavy localized with low base (mostly clean but very noisy regions)
    {"type": "ray_tracing", "intensity": "spotty_low", "base_scale": 0.15, "gaussian_layer": 0.02, "num_noisy_regions": 5, "difficult_scale": 0.5},
    {"type": "ray_tracing", "intensity": "spotty_medium", "base_scale": 0.2, "gaussian_layer": 0.025, "num_noisy_regions": 7, "difficult_scale": 0.4},
    {"type": "ray_tracing", "intensity": "spotty_high", "base_scale": 0.25, "gaussian_layer": 0.03, "num_noisy_regions": 9, "difficult_scale": 0.3},
    {"type": "ray_tracing", "intensity": "spotty_extreme", "base_scale": 0.3, "gaussian_layer": 0.04, "num_noisy_regions": 12, "difficult_scale": 0.2},
    {"type": "ray_tracing", "intensity": "spotty_realistic_low", "base_scale": 0.18, "gaussian_layer": 0.025, "num_noisy_regions": 6, "color_variance": True, "difficult_scale": 0.45},
    {"type": "ray_tracing", "intensity": "spotty_realistic_high", "base_scale": 0.28, "gaussian_layer": 0.035, "num_noisy_regions": 10, "color_variance": True, "difficult_scale": 0.25},
    
    # Gradient noise (simulating progressive rendering with some areas converged)
    {"type": "ray_tracing", "intensity": "progressive_early", "base_scale": 0.6, "gaussian_layer": 0.07, "num_noisy_regions": 4, "difficult_scale": 0.35},
    {"type": "ray_tracing", "intensity": "progressive_mid", "base_scale": 0.45, "gaussian_layer": 0.05, "num_noisy_regions": 5, "difficult_scale": 0.45},
    {"type": "ray_tracing", "intensity": "progressive_realistic", "base_scale": 0.5, "gaussian_layer": 0.055, "num_noisy_regions": 6, "color_variance": True, "difficult_scale": 0.4},
    
    # Extreme variations in every parameter
    {"type": "ray_tracing", "intensity": "wildcard_1", "base_scale": 0.55, "gaussian_layer": 0.12, "num_noisy_regions": 9, "color_variance": True, "difficult_scale": 0.28},
    {"type": "ray_tracing", "intensity": "wildcard_2", "base_scale": 0.75, "gaussian_layer": 0.06, "num_noisy_regions": 11, "difficult_scale": 0.22},
    {"type": "ray_tracing", "intensity": "wildcard_3", "base_scale": 0.42, "gaussian_layer": 0.15, "num_noisy_regions": 13, "color_variance": True, "difficult_scale": 0.32},
    {"type": "ray_tracing", "intensity": "wildcard_4", "base_scale": 0.88, "gaussian_layer": 0.11, "num_noisy_regions": 7, "color_variance": True, "difficult_scale": 0.16},
    {"type": "ray_tracing", "intensity": "wildcard_5", "base_scale": 0.33, "gaussian_layer": 0.08, "num_noisy_regions": 15, "difficult_scale": 0.24},
    
    # Sample splatting noise (photon mapping / scattered sample reconstruction)
    # Fine detail versions (high density, low displacement)
    {"type": "sample_splatting", "intensity": "ultra_fine", "sample_density": 0.95, "max_displacement": 1, "base_darkness": 0.05, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "very_fine", "sample_density": 0.92, "max_displacement": 2, "base_darkness": 0.08, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "fine", "sample_density": 0.87, "max_displacement": 2, "base_darkness": 0.1, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "high_detail", "sample_density": 0.83, "max_displacement": 3, "base_darkness": 0.12, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "medium_detail", "sample_density": 0.78, "max_displacement": 4, "base_darkness": 0.15, "blend_radius": 0},
    
    # Medium quality versions
    {"type": "sample_splatting", "intensity": "very_sparse", "sample_density": 0.45, "max_displacement": 8, "base_darkness": 0.3, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "sparse", "sample_density": 0.63, "max_displacement": 6, "base_darkness": 0.25, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "medium_sparse", "sample_density": 0.85, "max_displacement": 5, "base_darkness": 0.2, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "moderate", "sample_density": 0.67, "max_displacement": 4, "base_darkness": 0.15, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "dense", "sample_density": 0.88, "max_displacement": 3, "base_darkness": 0.1, "blend_radius": 0},
    
    # High displacement versions (scattered samples)
    {"type": "sample_splatting", "intensity": "high_displacement", "sample_density": 0.5, "max_displacement": 12, "base_darkness": 0.2, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "extreme_displacement", "sample_density": 0.82, "max_displacement": 20, "base_darkness": 0.25, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "chaotic_displacement", "sample_density": 0.39, "max_displacement": 30, "base_darkness": 0.3, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "insane_displacement", "sample_density": 0.96, "max_displacement": 40, "base_darkness": 0.35, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "apocalyptic_displacement", "sample_density": 0.7, "max_displacement": 50, "base_darkness": 0.4, "blend_radius": 0},
    
    # Ultra sparse versions (very challenging)
    {"type": "sample_splatting", "intensity": "ultra_sparse_dark", "sample_density": 0.33, "max_displacement": 10, "base_darkness": 0.5, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "ultra_sparse_scattered", "sample_density": 0.52, "max_displacement": 15, "base_darkness": 0.4, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "mega_sparse", "sample_density": 0.76, "max_displacement": 12, "base_darkness": 0.6, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "mega_sparse_scattered", "sample_density": 0.64, "max_displacement": 18, "base_darkness": 0.65, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "nearly_black", "sample_density": 0.73, "max_displacement": 15, "base_darkness": 0.75, "blend_radius": 0},
    
    # Photon-like versions (sharp samples, no blur)
    {"type": "sample_splatting", "intensity": "photon_ultra_fine", "sample_density": 0.92, "max_displacement": 1, "base_darkness": 0.03, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "photon_very_fine", "sample_density": 0.88, "max_displacement": 2, "base_darkness": 0.05, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "photon_fine", "sample_density": 0.82, "max_displacement": 2, "base_darkness": 0.08, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "photon_high_detail", "sample_density": 0.75, "max_displacement": 3, "base_darkness": 0.1, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "photon_medium_detail", "sample_density": 0.65, "max_displacement": 4, "base_darkness": 0.12, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "photon_like_low", "sample_density": 0.75, "max_displacement": 7, "base_darkness": 0.35, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "photon_like_medium", "sample_density": 0.95, "max_displacement": 5, "base_darkness": 0.25, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "photon_like_high", "sample_density": 0.59, "max_displacement": 4, "base_darkness": 0.18, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "photon_moderate", "sample_density": 0.74, "max_displacement": 5, "base_darkness": 0.15, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "photon_scattered", "sample_density": 0.94, "max_displacement": 8, "base_darkness": 0.2, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "photon_insane_displacement", "sample_density": 0.68, "max_displacement": 12, "base_darkness": 0.4, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "photon_apocalyptic_displacement", "sample_density": 0.85, "max_displacement": 11, "base_darkness": 0.5, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "photon_nearly_black", "sample_density": 0.97, "max_displacement": 9, "base_darkness": 0.7, "blend_radius": 0},
    
    # Mixed challenging versions (low density + high displacement)
    {"type": "sample_splatting", "intensity": "chaos_low", "sample_density": 0.55, "max_displacement": 25, "base_darkness": 0.45, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "chaos_medium", "sample_density": 0.65, "max_displacement": 35, "base_darkness": 0.4, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "chaos_high", "sample_density": 0.45, "max_displacement": 45, "base_darkness": 0.35, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "chaos_extreme", "sample_density": 0.35, "max_displacement": 55, "base_darkness": 0.5, "blend_radius": 0},
    
    # Varying darkness levels with moderate parameters
    {"type": "sample_splatting", "intensity": "dark_moderate", "sample_density": 0.88, "max_displacement": 8, "base_darkness": 0.55, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "very_dark", "sample_density": 0.80, "max_displacement": 10, "base_darkness": 0.7, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "extremely_dark", "sample_density": 0.75, "max_displacement": 12, "base_darkness": 0.8, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "pitch_black", "sample_density": 0.75, "max_displacement": 15, "base_darkness": 0.9, "blend_radius": 0},
    
    # No blending versions (harsh samples)
    {"type": "sample_splatting", "intensity": "harsh_sparse", "sample_density": 0.60, "max_displacement": 10, "base_darkness": 0.4, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "harsh_moderate", "sample_density": 0.55, "max_displacement": 8, "base_darkness": 0.3, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "harsh_dense", "sample_density": 0.5, "max_displacement": 6, "base_darkness": 0.2, "blend_radius": 0},
    
    # Sharp sample variations (no blur like real ray tracing)
    {"type": "sample_splatting", "intensity": "sharp_sparse", "sample_density": 0.95, "max_displacement": 10, "base_darkness": 0.35, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "sharp_moderate", "sample_density": 0.92, "max_displacement": 8, "base_darkness": 0.25, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "sharp_dense", "sample_density": 0.88, "max_displacement": 5, "base_darkness": 0.15, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "sharp_extreme", "sample_density": 0.82, "max_displacement": 15, "base_darkness": 0.45, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "sharp_chaotic", "sample_density": 0.78, "max_displacement": 20, "base_darkness": 0.5, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "sharp_insane", "sample_density": 0.70, "max_displacement": 25, "base_darkness": 0.6, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "sharp_apocalyptic", "sample_density": 0.92, "max_displacement": 30, "base_darkness": 0.7, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "sharp_nightmare", "sample_density": 0.89, "max_displacement": 35, "base_darkness": 0.8, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "sharp_abyss", "sample_density": 0.91, "max_displacement": 40, "base_darkness": 0.9, "blend_radius": 0},
    
    # Extreme combinations (worst case scenarios)
    {"type": "sample_splatting", "intensity": "nightmare_1", "sample_density": 0.75, "max_displacement": 40, "base_darkness": 0.7, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "nightmare_2", "sample_density": 0.68, "max_displacement": 50, "base_darkness": 0.75, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "nightmare_3", "sample_density": 0.92, "max_displacement": 35, "base_darkness": 0.65, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "nightmare_4", "sample_density": 0.48, "max_displacement": 45, "base_darkness": 0.8, "blend_radius": 0},
    
    # Barely visible versions
    {"type": "sample_splatting", "intensity": "ghost_1", "sample_density": 0.44, "max_displacement": 20, "base_darkness": 0.85, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "ghost_2", "sample_density": 0.39, "max_displacement": 25, "base_darkness": 0.9, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "ghost_3", "sample_density": 0.29, "max_displacement": 30, "base_darkness": 0.92, "blend_radius": 0},
    
    # Random challenging mixes
    {"type": "sample_splatting", "intensity": "random_hard_1", "sample_density": 0.32, "max_displacement": 22, "base_darkness": 0.48, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_2", "sample_density": 0.26, "max_displacement": 28, "base_darkness": 0.52, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_3", "sample_density": 0.23, "max_displacement": 32, "base_darkness": 0.58, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_4", "sample_density": 0.37, "max_displacement": 18, "base_darkness": 0.42, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_5", "sample_density": 0.45, "max_displacement": 38, "base_darkness": 0.62, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_6", "sample_density": 0.78, "max_displacement": 48, "base_darkness": 0.54, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_7", "sample_density": 0.92, "max_displacement": 64, "base_darkness": 0.47, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_8", "sample_density": 0.51, "max_displacement": 73, "base_darkness": 0.51, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_9", "sample_density": 0.61, "max_displacement": 55, "base_darkness": 0.49, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_10", "sample_density": 0.93, "max_displacement": 60, "base_darkness": 0.53, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_11", "sample_density": 0.39, "max_displacement": 70, "base_darkness": 0.46, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_12", "sample_density": 0.72, "max_displacement": 80, "base_darkness": 0.55, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_13", "sample_density": 0.84, "max_displacement": 90, "base_darkness": 0.6, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_14", "sample_density": 0.66, "max_displacement": 85, "base_darkness": 0.58, "blend_radius": 0},
    {"type": "sample_splatting", "intensity": "random_hard_15", "sample_density": 0.58, "max_displacement": 95, "base_darkness": 0.62, "blend_radius": 0},
    
    {"type": "poisson", "intensity": "low", "scale": 30.0, "whole_image": True},
    {"type": "poisson", "intensity": "medium", "scale": 15.0, "whole_image": True},
    {"type": "poisson", "intensity": "high", "scale": 7.0, "whole_image": True},
    {"type": "poisson", "intensity": "localized_low", "scale": 25.0, "whole_image": False, "coverage": 0.3},
    {"type": "poisson", "intensity": "localized_medium", "scale": 12.0, "whole_image": False, "coverage": 0.4},
    {"type": "poisson", "intensity": "localized_high", "scale": 5.0, "whole_image": False, "coverage": 0.5},

    {"type": "speckle", "intensity": "low", "sigma": 0.1, "whole_image": True},
    {"type": "speckle", "intensity": "medium", "sigma": 0.2, "whole_image": True},
    {"type": "speckle", "intensity": "high", "sigma": 0.3, "whole_image": True},
    {"type": "speckle", "intensity": "localized_low", "sigma": 0.15, "whole_image": False, "coverage": 0.3},
    {"type": "speckle", "intensity": "localized_medium", "sigma": 0.25, "whole_image": False, "coverage": 0.4},
    {"type": "speckle", "intensity": "localized_high", "sigma": 0.35, "whole_image": False, "coverage": 0.5},

    {"type": "salt_pepper", "intensity": "low", "amount": 0.01, "whole_image": True},
    {"type": "salt_pepper", "intensity": "medium", "amount": 0.03, "whole_image": True},
    {"type": "salt_pepper", "intensity": "high", "amount": 0.05, "whole_image": True},
    {"type": "salt_pepper", "intensity": "localized_low", "amount": 0.02, "whole_image": False, "coverage": 0.3},
    {"type": "salt_pepper", "intensity": "localized_medium", "amount": 0.04, "whole_image": False, "coverage": 0.4},
    {"type": "salt_pepper", "intensity": "localized_high", "amount": 0.06, "whole_image": False, "coverage": 0.5},

    {"type": "gaussian", "intensity": "low", "sigma": 10, "whole_image": True},
    {"type": "gaussian", "intensity": "medium", "sigma": 25, "whole_image": True},
    {"type": "gaussian", "intensity": "high", "sigma": 50, "whole_image": True},
    {"type": "gaussian", "intensity": "localized_low", "sigma": 15, "whole_image": False, "coverage": 0.3},
    {"type": "gaussian", "intensity": "localized_medium", "sigma": 30, "whole_image": False, "coverage": 0.4},
    {"type": "gaussian", "intensity": "localized_high", "sigma": 60, "whole_image": False, "coverage": 0.5},
]


def add_gaussian_noise(image, sigma, whole_image=True, coverage=1.0):
    """Add Gaussian noise to image"""
    noisy = image.copy().astype(np.float32) / 255.0
    
    if whole_image:
        noise = np.random.normal(0, sigma, image.shape)
        noisy += noise
    else:
        h, w = image.shape[:2]
        region_h, region_w = int(h * np.sqrt(coverage)), int(w * np.sqrt(coverage))
        y = np.random.randint(0, max(1, h - region_h))
        x = np.random.randint(0, max(1, w - region_w))
        noise = np.random.normal(0, sigma, (region_h, region_w, image.shape[2]))
        noisy[y:y+region_h, x:x+region_w] += noise
    
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount, whole_image=True, coverage=1.0):
    """Add salt and pepper noise"""
    noisy = image.copy()
    
    if whole_image:
        mask = np.random.choice([0, 1, 2], size=image.shape[:2], p=[1-amount, amount/2, amount/2])
        noisy[mask == 1] = 255
        noisy[mask == 2] = 0
    else:
        h, w = image.shape[:2]
        region_h, region_w = int(h * np.sqrt(coverage)), int(w * np.sqrt(coverage))
        y = np.random.randint(0, max(1, h - region_h))
        x = np.random.randint(0, max(1, w - region_w))
        mask = np.random.choice([0, 1, 2], size=(region_h, region_w), p=[1-amount, amount/2, amount/2])
        noisy[y:y+region_h, x:x+region_w][mask == 1] = 255
        noisy[y:y+region_h, x:x+region_w][mask == 2] = 0
    
    return noisy


def add_speckle_noise(image, sigma, whole_image=True, coverage=1.0):
    """Add speckle (multiplicative) noise"""
    noisy = image.copy().astype(np.float32) / 255.0
    
    if whole_image:
        noise = np.random.normal(1, sigma, image.shape)
        noisy *= noise
    else:
        h, w = image.shape[:2]
        region_h, region_w = int(h * np.sqrt(coverage)), int(w * np.sqrt(coverage))
        y = np.random.randint(0, max(1, h - region_h))
        x = np.random.randint(0, max(1, w - region_w))
        noise = np.random.normal(1, sigma, (region_h, region_w, image.shape[2]))
        noisy[y:y+region_h, x:x+region_w] *= noise
    
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)


def add_poisson_noise(image, whole_image=True, scale=1.0, coverage=1.0):
    """Add Poisson noise
    
    Args:
        image: Input image
        whole_image: Apply to whole image or localized region
        scale: Scale factor for controlling noise intensity (lower = more noise)
        coverage: Coverage for localized noise
    """
    noisy = image.copy().astype(np.float32)
    
    if whole_image:
        scaled = noisy * scale
        noisy = np.random.poisson(scaled) / scale
    else:
        h, w = image.shape[:2]
        region_h, region_w = int(h * np.sqrt(coverage)), int(w * np.sqrt(coverage))
        y = np.random.randint(0, max(1, h - region_h))
        x = np.random.randint(0, max(1, w - region_w))
        
        scaled = noisy[y:y+region_h, x:x+region_w] * scale
        noisy[y:y+region_h, x:x+region_w] = np.random.poisson(scaled) / scale
    
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_sample_splatting_noise(image, config):
    """Apply sample splatting noise (simulates photon mapping / low sample count path tracing)
    
    Creates noise by:
    1. Starting with black/dark canvas
    2. Sampling colors from original image
    3. Placing samples at nearby positions (with random displacement)
    4. Blending samples to create final noisy image
    
    This simulates ray tracers with very low sample counts where pixel colors
    are reconstructed from sparse, scattered samples.
    """
    h, w = image.shape[:2]
    
    sample_density = randomize_config_value(config.get("sample_density", 0.3))
    max_displacement = randomize_config_value(config.get("max_displacement", 5))
    base_darkness = randomize_config_value(config.get("base_darkness", 0.2))
    blend_radius = int(randomize_config_value(config.get("blend_radius", 1)))
    
    canvas = np.ones_like(image, dtype=np.float32) * (image.mean() * base_darkness)
    sample_counts = np.zeros((h, w), dtype=np.float32)
    
    num_samples = int(h * w * sample_density)
    
    for _ in range(num_samples):
        src_y = np.random.randint(0, h)
        src_x = np.random.randint(0, w)
        
        color = image[src_y, src_x].astype(np.float32)
        
        disp_y = int(np.random.normal(0, max_displacement))
        disp_x = int(np.random.normal(0, max_displacement))
        
        dst_y = np.clip(src_y + disp_y, 0, h - 1)
        dst_x = np.clip(src_x + disp_x, 0, w - 1)
        
        for dy in range(-blend_radius, blend_radius + 1):
            for dx in range(-blend_radius, blend_radius + 1):
                py = np.clip(dst_y + dy, 0, h - 1)
                px = np.clip(dst_x + dx, 0, w - 1)
                
                weight = 1.0 / (1.0 + abs(dy) + abs(dx))
                canvas[py, px] += color * weight
                sample_counts[py, px] += weight
    
    mask = sample_counts > 0
    canvas[mask] /= sample_counts[mask, np.newaxis]
    
    no_sample_pixels = ~mask
    if np.any(no_sample_pixels):
        canvas[no_sample_pixels] = (image[no_sample_pixels].astype(np.float32) * base_darkness + 
                                     np.random.normal(0, 15, (np.sum(no_sample_pixels), 3)))
    
    return np.clip(canvas, 0, 255).astype(np.uint8)


def apply_ray_tracing_noise(image, config):
    """Apply ray tracing-style noise (Poisson base + Gaussian + localized variations)
    
    Ray tracing noise characteristics:
    - Poisson noise as base (from Monte Carlo sampling)
    - Gaussian noise layer (from variance in light path integration)  
    - Localized regions with higher noise (difficult indirect lighting, caustics, etc.)
    - Optional per-channel variance
    """
    noisy = image.copy().astype(np.float32)
    h, w = image.shape[:2]
    
    base_scale = randomize_config_value(config.get("base_scale", 0.3))
    if config.get("color_variance", False):
        for c in range(3):
            channel_scale = base_scale * np.random.uniform(0.8, 1.2)
            scaled = noisy[:, :, c] * channel_scale
            noisy[:, :, c] = np.random.poisson(scaled) / channel_scale
    else:
        scaled = noisy * base_scale
        noisy = np.random.poisson(scaled) / base_scale
    
    gaussian_sigma = randomize_config_value(config.get("gaussian_layer", 0.03))
    gaussian_noise = np.random.normal(0, gaussian_sigma, image.shape)
    noisy = noisy + gaussian_noise * 255.0
    
    num_noisy_regions = int(randomize_config_value(config.get("num_noisy_regions", 3)))
    difficult_scale = config.get("difficult_scale", None)
    if difficult_scale is not None:
        difficult_scale = randomize_config_value(difficult_scale)
    
    for _ in range(num_noisy_regions):
        # Random region size and position
        region_coverage = np.random.uniform(0.05, 0.25)
        region_h = int(h * np.random.uniform(0.15, 0.4))
        region_w = int(w * np.random.uniform(0.15, 0.4))
        
        y = np.random.randint(0, max(1, h - region_h))
        x = np.random.randint(0, max(1, w - region_w))
        
        # Apply extra Poisson noise to this region
        if difficult_scale is not None:
            region = noisy[y:y+region_h, x:x+region_w]
            scaled_region = region * difficult_scale
            noisy[y:y+region_h, x:x+region_w] = np.random.poisson(np.clip(scaled_region, 0, None)) / difficult_scale
        
        # Add extra Gaussian noise to region
        extra_gaussian = np.random.normal(0, gaussian_sigma * 1.5, (region_h, region_w, 3))
        noisy[y:y+region_h, x:x+region_w] += extra_gaussian * 255.0
    
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_noise(image, config, num_applications=1):
    """Apply noise based on configuration
    
    Args:
        image: Input image as numpy array
        config: Noise configuration dict
        num_applications: Number of times to apply the noise (for layered effect)
    """
    noise_type = config["type"]
    
    # Handle ray tracing noise separately
    if noise_type == "ray_tracing":
        return apply_ray_tracing_noise(image, config)
    
    # Handle sample splatting noise
    if noise_type == "sample_splatting":
        img = apply_sample_splatting_noise(image, config)
        
        ray_tracing_configs = [c for c in RAY_TRACING_NOISE_CONFIGS if c.get("type") == "ray_tracing"]
        if ray_tracing_configs:
            extra_config = random.choice(ray_tracing_configs)
            img = apply_noise(img, extra_config, num_applications=1)
        
        return img

      
    whole_image = config.get("whole_image", True)
    coverage = config.get("coverage", 1.0)
    
    if config.get("multi_apply", False):
        num_applications = config.get("applications", num_applications)
    
    if noise_type == "hybrid":
        base_noise = config["base_noise"]
        noisy = image.copy()
        
        if base_noise == "gaussian":
            whole_sigma = randomize_config_value(config["whole_sigma"])
            local_sigma = randomize_config_value(config["local_sigma"])
            noisy = add_gaussian_noise(noisy, whole_sigma, whole_image=True)
            noisy = add_gaussian_noise(noisy, local_sigma, whole_image=False, coverage=coverage)
        elif base_noise == "salt_pepper":
            whole_amount = randomize_config_value(config["whole_amount"])
            local_amount = randomize_config_value(config["local_amount"])
            noisy = add_salt_pepper_noise(noisy, whole_amount, whole_image=True)
            noisy = add_salt_pepper_noise(noisy, local_amount, whole_image=False, coverage=coverage)
        elif base_noise == "speckle":
            whole_sigma = randomize_config_value(config["whole_sigma"])
            local_sigma = randomize_config_value(config["local_sigma"])
            noisy = add_speckle_noise(noisy, whole_sigma, whole_image=True)
            noisy = add_speckle_noise(noisy, local_sigma, whole_image=False, coverage=coverage)
        elif base_noise == "poisson":
            whole_scale = randomize_config_value(config["whole_scale"])
            local_scale = randomize_config_value(config["local_scale"])
            noisy = add_poisson_noise(noisy, whole_image=True, scale=whole_scale)
            noisy = add_poisson_noise(noisy, whole_image=False, scale=local_scale, coverage=coverage)
        
        return noisy
    
    noisy = image.copy()
    for _ in range(num_applications):
        if noise_type == "gaussian":
            sigma = randomize_config_value(config["sigma"])
            noisy = add_gaussian_noise(noisy, sigma, whole_image, coverage)
        elif noise_type == "salt_pepper":
            amount = randomize_config_value(config["amount"])
            noisy = add_salt_pepper_noise(noisy, amount, whole_image, coverage)
        elif noise_type == "speckle":
            sigma = randomize_config_value(config["sigma"])
            noisy = add_speckle_noise(noisy, sigma, whole_image, coverage)
        elif noise_type == "poisson":
            scale = randomize_config_value(config.get("scale", 1.0))
            noisy = add_poisson_noise(noisy, whole_image, scale, coverage)
    
    return noisy


def crop_center(image, target_width, target_height):
    """Crop image from center to target size"""
    width, height = image.size
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    return image.crop((left, top, right, bottom))


def process_image_sync(image_data, image_index, save_path, configs):
    """Process a single image synchronously (for parallel execution)"""
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        width, height = image.size
        
        if width < IMAGE_WIDTH_HIGH_RES or height < IMAGE_HEIGHT_HIGH_RES:
            return False
        
        if width == IMAGE_WIDTH_HIGH_RES and height == IMAGE_HEIGHT_HIGH_RES:
            return False
        
        high_res = crop_center(image, IMAGE_WIDTH_HIGH_RES, IMAGE_HEIGHT_HIGH_RES)
        
        image_folder = Path(save_path) / f"image_{image_index:08d}"
        image_folder.mkdir(parents=True, exist_ok=True)
        
        high_res_resized = high_res.resize((IMAGE_WIDTH_LOW_RES, IMAGE_HEIGHT_LOW_RES), Image.Resampling.LANCZOS)
        high_res_resized.save(image_folder / "high_res.png", quality=100, compress_level=1)
        high_res_luminance = high_res_resized.convert("L")
        high_res_luminance.save(image_folder / "high_res_luminance.png", quality=100, compress_level=1)
        
        low_res = high_res.resize((IMAGE_WIDTH_LOW_RES, IMAGE_HEIGHT_LOW_RES), Image.Resampling.LANCZOS)
        
        low_res_np = np.array(low_res)
        
        noise_config = random.choice(configs)
        
        if noise_config.get("type") == "sample_splatting":
            num_applications = 1
        elif not noise_config.get("multi_apply", False):
            num_applications = random.randint(1, MAX_NOISE_APPLICATIONS)
        else:
            num_applications = 1
        
        noisy_low_res = apply_noise(low_res_np, noise_config, num_applications)
        
        low_res_noisy = Image.fromarray(noisy_low_res)
        low_res_noisy.save(image_folder / "low_res.png", quality=70, compress_level=1)
        low_res_luminance = low_res_noisy.convert("L")
        low_res_luminance.save(image_folder / "low_res_luminance.png", quality=70, compress_level=1)
        
        metadata = {
            "noise_config": noise_config,
            "num_applications": num_applications if not noise_config.get("multi_apply", False) else noise_config.get("applications", 1),
            "high_res_size": [IMAGE_WIDTH_HIGH_RES, IMAGE_HEIGHT_HIGH_RES],
            "low_res_size": [IMAGE_WIDTH_LOW_RES, IMAGE_HEIGHT_LOW_RES]
        }
        metadata_path = image_folder / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return True
        
    except Exception as e:
        return False


async def process_image(image_data, image_index, executor):
    """Process image asynchronously using executor"""
    if image_data is None:
        return False
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, 
        process_image_sync, 
        image_data, 
        image_index, 
        SAVE_PATH, 
        RAY_TRACING_NOISE_CONFIGS
    )


async def download_image(session, url, semaphore):
    """Download a single image asynchronously"""
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT), ssl=False) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    return None
        except:
            return None


async def download_batch(urls):
    """Download multiple images concurrently"""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_image(session, url, semaphore) for url in urls]
        return await asyncio.gather(*tasks)


def save_index(index):
    """Save current index to file"""
    with open("imageIndex.json", "w") as f:
        json.dump({"image_index": index}, f)


async def process_dataset():
    """Main processing function with parallel async downloads and processing"""
    current_index = json.load(open("imageIndex.json"))["image_index"]
    ds = load_dataset("laion/relaion-high-resolution", split="train", streaming=True)
    
    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
    
    batch_urls = []
    batch_indices = []
    processed_count = 0
    
    print(f"Starting from index {current_index}")
    print(f"Using {MAX_PROCESSING_WORKERS} workers for parallel processing")
    
    with ProcessPoolExecutor(max_workers=MAX_PROCESSING_WORKERS) as executor:
        for item in iter(ds):
            batch_urls.append(item["URL"])
            batch_indices.append(current_index)
            current_index += 1
            
            if len(batch_urls) >= MAX_CONCURRENT_DOWNLOADS:
                image_data_list = await download_batch(batch_urls)
                
                tasks = [
                    process_image(image_data, idx, executor)
                    for image_data, idx in zip(image_data_list, batch_indices)
                ]
                results = await asyncio.gather(*tasks)
                
                processed_count += sum(results)
                
                if current_index % SAVE_PERIOD == 0:
                    save_index(current_index)
                    print(f"Progress: {current_index} downloaded, {processed_count} processed successfully")
                
                batch_urls = []
                batch_indices = []
        
        if batch_urls:
            image_data_list = await download_batch(batch_urls)
            tasks = [
                process_image(image_data, idx, executor)
                for image_data, idx in zip(image_data_list, batch_indices)
            ]
            results = await asyncio.gather(*tasks)
            processed_count += sum(results)
    
    save_index(current_index)
    print(f"Processing complete! Total: {current_index} downloaded, {processed_count} processed successfully")


class ImageDenoiseDataset(Dataset):
    """PyTorch Dataset for image denoising and upscaling"""
    
    def __init__(self, data_path, transform=None, load_metadata=False):
        """
        Args:
            data_path: Path to the imageData folder containing image_XXXXXXXX subfolders
            transform: Optional torchvision transforms to apply to both images
            load_metadata: Whether to return metadata along with images
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.load_metadata = load_metadata
        
        self.image_folders = sorted([d for d in self.data_path.iterdir() 
                                     if d.is_dir() and d.name.startswith('image_')])
        
        self.valid_folders = []
        for folder in self.image_folders:
            high_res_path = folder / "high_res.png"
            low_res_path = folder / "low_res.png"
            if high_res_path.exists() and low_res_path.exists():
                self.valid_folders.append(folder)
        
        print(f"Found {len(self.valid_folders)} valid image pairs in {data_path}")
    
    def __len__(self):
        return len(self.valid_folders)
    
    def __getitem__(self, idx):
        """
        Returns:
            low_res: Tensor of noisy low-resolution image (input)
            high_res: Tensor of clean high-resolution image (target)
            metadata: Optional dict with noise configuration used
        """
        max_retries = 10
        for retry in range(max_retries):
            try:
                actual_idx = (idx + retry) % len(self.valid_folders)
                folder = self.valid_folders[actual_idx]
                
                high_res_path = folder / "high_res.png"
                low_res_path = folder / "low_res.png"
                
                high_res = Image.open(high_res_path).convert('RGB')
                low_res = Image.open(low_res_path).convert('RGB')
                
                if self.transform:
                    high_res = self.transform(high_res)
                    low_res = self.transform(low_res)
                else:
                    to_tensor = transforms.ToTensor()
                    high_res = to_tensor(high_res)
                    low_res = to_tensor(low_res)
                
                if self.load_metadata:
                    metadata_path = folder / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        return low_res, high_res, metadata
                    else:
                        return low_res, high_res, {}
                
                return low_res, high_res
            
            except (OSError, IOError, ValueError) as e:
                if retry < max_retries - 1:
                    continue
                else:
                    raise RuntimeError(f"Failed to load image after {max_retries} retries. Last error: {e}")


def create_data_loaders(data_path, batch_size=16, train_split=0.9, num_workers=4, 
                       shuffle=True, transform=None):
    """
    Create train and validation data loaders
    
    Args:
        data_path: Path to the imageData folder
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training (rest for validation)
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the training data
        transform: Optional torchvision transforms
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    full_dataset = ImageDenoiseDataset(data_path, transform=transform)
    
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    asyncio.run(process_dataset())
            