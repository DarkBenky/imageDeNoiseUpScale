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

SAVE_PATH = "/media/user/f7a503ec-b25c-41a5-9baa-d350714f613a/imageData"
SAVE_PERIOD = 256
IMAGE_WIDTH_LOW_RES = 800
IMAGE_HEIGHT_LOW_RES = 600
IMAGE_WIDTH_HIGH_RES = int(800 * 1.5)
IMAGE_HEIGHT_HIGH_RES = int(600 * 1.5)

MAX_CONCURRENT_DOWNLOADS = 256
DOWNLOAD_TIMEOUT = 3
MAX_PROCESSING_WORKERS = multiprocessing.cpu_count() * 2

MAX_NOISE_APPLICATIONS = 5

NOISE_CONFIGS = [
    {"type": "gaussian", "intensity": "very_low", "whole_image": True, "sigma": 0.02},
    {"type": "gaussian", "intensity": "low", "whole_image": True, "sigma": 0.04},
    {"type": "gaussian", "intensity": "medium", "whole_image": True, "sigma": 0.08},
    {"type": "gaussian", "intensity": "high", "whole_image": True, "sigma": 0.15},
    {"type": "gaussian", "intensity": "very_high", "whole_image": True, "sigma": 0.25},
    {"type": "gaussian", "intensity": "extreme", "whole_image": True, "sigma": 0.4},
    {"type": "gaussian", "intensity": "ultra_extreme", "whole_image": True, "sigma": 0.6},
    {"type": "gaussian", "intensity": "high_localized", "whole_image": False, "sigma": 0.25, "coverage": 0.3},
    {"type": "gaussian", "intensity": "extreme_localized", "whole_image": False, "sigma": 0.5, "coverage": 0.2},
    
    {"type": "salt_pepper", "intensity": "very_low", "whole_image": True, "amount": 0.02},
    {"type": "salt_pepper", "intensity": "low", "whole_image": True, "amount": 0.05},
    {"type": "salt_pepper", "intensity": "medium", "whole_image": True, "amount": 0.1},
    {"type": "salt_pepper", "intensity": "high", "whole_image": True, "amount": 0.2},
    {"type": "salt_pepper", "intensity": "very_high", "whole_image": True, "amount": 0.3},
    {"type": "salt_pepper", "intensity": "extreme", "whole_image": True, "amount": 0.4},
    {"type": "salt_pepper", "intensity": "high_localized", "whole_image": False, "amount": 0.25, "coverage": 0.2},
    {"type": "salt_pepper", "intensity": "extreme_localized", "whole_image": False, "amount": 0.5, "coverage": 0.15},
    
    {"type": "speckle", "intensity": "very_low", "whole_image": True, "sigma": 0.05},
    {"type": "speckle", "intensity": "low", "whole_image": True, "sigma": 0.1},
    {"type": "speckle", "intensity": "medium", "whole_image": True, "sigma": 0.2},
    {"type": "speckle", "intensity": "high", "whole_image": True, "sigma": 0.3},
    {"type": "speckle", "intensity": "very_high", "whole_image": True, "sigma": 0.45},
    {"type": "speckle", "intensity": "extreme", "whole_image": True, "sigma": 0.6},
    {"type": "speckle", "intensity": "high_localized", "whole_image": False, "sigma": 0.35, "coverage": 0.25},
    {"type": "speckle", "intensity": "extreme_localized", "whole_image": False, "sigma": 0.5, "coverage": 0.2},
    
    {"type": "poisson", "intensity": "very_low", "whole_image": True, "scale": 1.0},
    {"type": "poisson", "intensity": "low", "whole_image": True, "scale": 0.8},
    {"type": "poisson", "intensity": "medium", "whole_image": True, "scale": 0.6},
    {"type": "poisson", "intensity": "high", "whole_image": True, "scale": 0.4},
    {"type": "poisson", "intensity": "very_high", "whole_image": True, "scale": 0.2},
    {"type": "poisson", "intensity": "extreme", "whole_image": True, "scale": 0.1},
    {"type": "poisson", "intensity": "low_localized", "whole_image": False, "scale": 0.6, "coverage": 0.3},
    {"type": "poisson", "intensity": "high_localized", "whole_image": False, "scale": 0.3, "coverage": 0.2},
    
    {"type": "gaussian", "intensity": "combined_low", "whole_image": True, "sigma": 0.05, "multi_apply": True, "applications": 2},
    {"type": "gaussian", "intensity": "combined_medium", "whole_image": True, "sigma": 0.08, "multi_apply": True, "applications": 3},
    {"type": "gaussian", "intensity": "combined_high", "whole_image": True, "sigma": 0.1, "multi_apply": True, "applications": 4},
    {"type": "speckle", "intensity": "combined_low", "whole_image": True, "sigma": 0.08, "multi_apply": True, "applications": 2},
    {"type": "speckle", "intensity": "combined_medium", "whole_image": True, "sigma": 0.12, "multi_apply": True, "applications": 3},
    {"type": "salt_pepper", "intensity": "combined_low", "whole_image": True, "amount": 0.05, "multi_apply": True, "applications": 2},
    {"type": "salt_pepper", "intensity": "combined_medium", "whole_image": True, "amount": 0.08, "multi_apply": True, "applications": 2},
    {"type": "poisson", "intensity": "combined_medium", "whole_image": True, "scale": 0.5, "multi_apply": True, "applications": 2},
    {"type": "poisson", "intensity": "combined_high", "whole_image": True, "scale": 0.3, "multi_apply": True, "applications": 3},

    {"type": "hybrid", "base_noise": "gaussian", "intensity": "whole_plus_local_low", "whole_sigma": 0.03, "local_sigma": 0.15, "coverage": 0.25},
    {"type": "hybrid", "base_noise": "gaussian", "intensity": "whole_plus_local_medium", "whole_sigma": 0.05, "local_sigma": 0.25, "coverage": 0.3},
    {"type": "hybrid", "base_noise": "gaussian", "intensity": "whole_plus_local_high", "whole_sigma": 0.08, "local_sigma": 0.4, "coverage": 0.2},
    {"type": "hybrid", "base_noise": "gaussian", "intensity": "whole_plus_local_extreme", "whole_sigma": 0.1, "local_sigma": 0.6, "coverage": 0.15},
    
    {"type": "hybrid", "base_noise": "salt_pepper", "intensity": "whole_plus_local_low", "whole_amount": 0.03, "local_amount": 0.15, "coverage": 0.25},
    {"type": "hybrid", "base_noise": "salt_pepper", "intensity": "whole_plus_local_medium", "whole_amount": 0.05, "local_amount": 0.25, "coverage": 0.3},
    {"type": "hybrid", "base_noise": "salt_pepper", "intensity": "whole_plus_local_high", "whole_amount": 0.08, "local_amount": 0.4, "coverage": 0.2},
    
    {"type": "hybrid", "base_noise": "speckle", "intensity": "whole_plus_local_low", "whole_sigma": 0.05, "local_sigma": 0.2, "coverage": 0.3},
    {"type": "hybrid", "base_noise": "speckle", "intensity": "whole_plus_local_medium", "whole_sigma": 0.1, "local_sigma": 0.3, "coverage": 0.25},
    {"type": "hybrid", "base_noise": "speckle", "intensity": "whole_plus_local_high", "whole_sigma": 0.15, "local_sigma": 0.5, "coverage": 0.2},
    
    {"type": "hybrid", "base_noise": "poisson", "intensity": "whole_plus_local_low", "whole_scale": 0.8, "local_scale": 0.4, "coverage": 0.3},
    {"type": "hybrid", "base_noise": "poisson", "intensity": "whole_plus_local_medium", "whole_scale": 0.6, "local_scale": 0.2, "coverage": 0.25},
    {"type": "hybrid", "base_noise": "poisson", "intensity": "whole_plus_local_high", "whole_scale": 0.4, "local_scale": 0.1, "coverage": 0.2},
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


def apply_noise(image, config, num_applications=1):
    """Apply noise based on configuration
    
    Args:
        image: Input image as numpy array
        config: Noise configuration dict
        num_applications: Number of times to apply the noise (for layered effect)
    """
    noise_type = config["type"]
    whole_image = config.get("whole_image", True)
    coverage = config.get("coverage", 1.0)
    
    if config.get("multi_apply", False):
        num_applications = config.get("applications", num_applications)
    
    if noise_type == "hybrid":
        base_noise = config["base_noise"]
        noisy = image.copy()
        
        if base_noise == "gaussian":
            noisy = add_gaussian_noise(noisy, config["whole_sigma"], whole_image=True)
            noisy = add_gaussian_noise(noisy, config["local_sigma"], whole_image=False, coverage=coverage)
        elif base_noise == "salt_pepper":
            noisy = add_salt_pepper_noise(noisy, config["whole_amount"], whole_image=True)
            noisy = add_salt_pepper_noise(noisy, config["local_amount"], whole_image=False, coverage=coverage)
        elif base_noise == "speckle":
            noisy = add_speckle_noise(noisy, config["whole_sigma"], whole_image=True)
            noisy = add_speckle_noise(noisy, config["local_sigma"], whole_image=False, coverage=coverage)
        elif base_noise == "poisson":
            noisy = add_poisson_noise(noisy, whole_image=True, scale=config["whole_scale"])
            noisy = add_poisson_noise(noisy, whole_image=False, scale=config["local_scale"], coverage=coverage)
        
        return noisy
    
    noisy = image.copy()
    for _ in range(num_applications):
        if noise_type == "gaussian":
            noisy = add_gaussian_noise(noisy, config["sigma"], whole_image, coverage)
        elif noise_type == "salt_pepper":
            noisy = add_salt_pepper_noise(noisy, config["amount"], whole_image, coverage)
        elif noise_type == "speckle":
            noisy = add_speckle_noise(noisy, config["sigma"], whole_image, coverage)
        elif noise_type == "poisson":
            scale = config.get("scale", 1.0)
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
        
        high_res_path = image_folder / "high_res.png"
        high_res.save(high_res_path, quality=95, compress_level=1)
        
        low_res = high_res.resize((IMAGE_WIDTH_LOW_RES, IMAGE_HEIGHT_LOW_RES), Image.Resampling.LANCZOS)
        
        low_res_np = np.array(low_res)
        
        noise_config = random.choice(configs)
        
        if not noise_config.get("multi_apply", False):
            num_applications = random.randint(1, MAX_NOISE_APPLICATIONS)
        else:
            num_applications = 1
        
        noisy_low_res = apply_noise(low_res_np, noise_config, num_applications)
        
        low_res_noisy = Image.fromarray(noisy_low_res)
        low_res_path = image_folder / "low_res.png"
        low_res_noisy.save(low_res_path, quality=95, compress_level=1)
        
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
        NOISE_CONFIGS
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
    
    with ThreadPoolExecutor(max_workers=MAX_PROCESSING_WORKERS) as executor:
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
        folder = self.valid_folders[idx]
        
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
            