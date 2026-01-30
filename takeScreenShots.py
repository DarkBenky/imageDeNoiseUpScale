import time
from datetime import datetime
from pathlib import Path
from mss import mss
from PIL import Image, ImageFilter
import random
import string
import json
import numpy as np

SAVE_DIRECTORY = "/home/user/Desktop/screenshots"
SCREENSHOT_INTERVAL = 1

def take_screenshots():
    save_path = Path(SAVE_DIRECTORY)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting screenshot capture every {SCREENSHOT_INTERVAL} seconds")
    print(f"Saving to: {SAVE_DIRECTORY}")
    print("Press Ctrl+C to stop\n")
    
    screenshot_count = 0
    
    try:
        with mss() as sct:
            while True:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                for monitor_num, monitor in enumerate(sct.monitors[1:], start=1):
                    screenshot = sct.grab(monitor)
                    
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                    
                    filename = f"monitor{monitor_num}_{timestamp}.png"
                    filepath = save_path / filename
                    
                    img.save(filepath)
                    print(f"Saved: {filename}")
                
                screenshot_count += 1
                print(f"Capture #{screenshot_count} complete\n")
                
                time.sleep(SCREENSHOT_INTERVAL)
                
    except KeyboardInterrupt:
        print(f"\nStopped. Total captures: {screenshot_count}")

DAMP_PATH = "/media/user/2TB Clear/imageData"
IMAGE_WIDTH_LOW_RES = 800
IMAGE_HEIGHT_LOW_RES = 600
IMAGE_WIDTH_HIGH_RES = IMAGE_HEIGHT_LOW_RES
IMAGE_HEIGHT_HIGH_RES = IMAGE_WIDTH_LOW_RES

def pixel_jitter(img, max_shift=1):
    arr = np.asarray(img)
    h, w, _ = arr.shape

    dx = np.random.randint(-max_shift, max_shift + 1, size=(h, w))
    dy = np.random.randint(-max_shift, max_shift + 1, size=(h, w))

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xx = np.clip(xx + dx, 0, w - 1)
    yy = np.clip(yy + dy, 0, h - 1)

    return Image.fromarray(arr[yy, xx])

def ray_tracing_noise(img, strength=0.15, chroma=0.6):
    arr = np.asarray(img).astype(np.float32) / 255.0

    # luminance (used to scale noise like Monte Carlo variance)
    lum = (
        0.2126 * arr[..., 0] +
        0.7152 * arr[..., 1] +
        0.0722 * arr[..., 2]
    )

    # noise grows with brightness (key ray tracing trait)
    noise = np.random.normal(0, strength, arr.shape)
    noise *= lum[..., None]

    # extra chromatic speckle
    chroma_noise = np.random.normal(0, strength * chroma, arr.shape)

    out = arr + noise + chroma_noise
    out = np.clip(out, 0.0, 1.0)

    return Image.fromarray((out * 255).astype(np.uint8))

def dump_screenshots():
    src_path = Path(SAVE_DIRECTORY)
    dst_path = Path(DAMP_PATH)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    screenshots = sorted(src_path.glob("*.png"))
    
    successful = 0
    failed = 0
    failed_files = []
    
    for i, screenshot in enumerate(screenshots, start=1):
        try:
            with Image.open(screenshot) as img:
                high_res_img = img.resize((IMAGE_WIDTH_LOW_RES, IMAGE_HEIGHT_LOW_RES), Image.LANCZOS)
                high_res_luminance = high_res_img.convert("L")
                
                downScale_factor = random.uniform(0.5, 1.0)
                low_res_down = img.resize((int(IMAGE_WIDTH_LOW_RES * downScale_factor), int(IMAGE_HEIGHT_LOW_RES * downScale_factor)), Image.LANCZOS)
                low_res_img = low_res_down.resize((IMAGE_WIDTH_LOW_RES, IMAGE_HEIGHT_LOW_RES), Image.LANCZOS)
                
                for _ in range(random.randint(0, 3)):
                    modifications = ["quantize", "noise", "blur", 'compression', 'rayTracingNoise', 'pixelJitter']
                    option = random.choice(modifications)
                    if option == "quantize":
                        low_res_img = low_res_img.quantize(colors=random.randint(8, 256), method=Image.FASTOCTREE).convert("RGB")
                    elif option == "noise":
                        sigma = random.uniform(5, 30)
                        noise = Image.effect_noise(low_res_img.size, sigma)
                        low_res_img = Image.blend(low_res_img, noise.convert("RGB"), 0.5)
                    elif option == "blur":
                        radius = random.uniform(1, 5)
                        low_res_img = low_res_img.filter(ImageFilter.GaussianBlur(radius))
                    elif option == "compression":
                        from io import BytesIO
                        buffer = BytesIO()
                        quality = random.randint(10, 50)
                        low_res_img.save(buffer, format="JPEG", quality=quality)
                        buffer.seek(0)
                        low_res_img = Image.open(buffer).convert("RGB")
                    elif option == "rayTracingNoise":
                        strength = random.uniform(0.1, 0.3)
                        chroma = random.uniform(0.4, 0.8)
                        low_res_img = ray_tracing_noise(low_res_img, strength=strength, chroma=chroma)
                    elif option == "pixelJitter":
                        offset = random.randint(1, 12)
                        low_res_img = pixel_jitter(low_res_img, max_shift=offset)

                low_res_luminance = low_res_img.convert("L")
                
                random_string = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6))

                folder_name = f"image_screenShot_{random_string}_{i:08d}"
                folder_path = dst_path / folder_name
                folder_path.mkdir(parents=True, exist_ok=True)
                
                low_res_img.save(folder_path / "low_res.png")
                low_res_luminance.save(folder_path / "low_res_luminance.png")
                high_res_img.save(folder_path / "high_res.png")
                high_res_luminance.save(folder_path / "high_res_luminance.png")
                
                metadata = {
                    "noise_config": {"type": "screenshot", "source": "screen_capture", "modification": option},
                    "num_applications": 0,
                    "high_res_size": [IMAGE_WIDTH_LOW_RES, IMAGE_HEIGHT_LOW_RES],
                    "low_res_size": [IMAGE_WIDTH_LOW_RES, IMAGE_HEIGHT_LOW_RES]
                }
                with open(folder_path / "metadata.json", 'w') as f:
                    json.dump(metadata, f)
                
                screenshot.unlink()
                successful += 1
        
        except (OSError, IOError) as e:
            print(f"Skipping corrupted file: {screenshot.name} - {e}")
            failed += 1
            failed_files.append(screenshot.name)
            try:
                screenshot.unlink()
            except:
                pass
            continue
        
        if i % 50 == 0:
            print(f"Processed {i}/{len(screenshots)} - Success: {successful}, Failed: {failed}")
    
    print(f"\nDumping completed!")
    print(f"Total processed: {len(screenshots)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if failed_files:
        print(f"Failed files: {', '.join(failed_files[:10])}")
        if len(failed_files) > 10:
            print(f"... and {len(failed_files) - 10} more")

def process_for_training(image_dir_paths:  list[Path], save_path: Path):
    for image_dir in image_dir_paths:
        try:
            metadata_path = image_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            noise_type = metadata.get("noise_config", {}).get("type", "")
            
            low_res_path = image_dir / "low_res.png"
            high_res_path = image_dir / "high_res.png"
            
            if not low_res_path.exists() or not high_res_path.exists():
                continue
            
            random_string = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=8))
            output_dir = save_path / f"sample_{random_string}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if noise_type == "screenshot":
                with Image.open(low_res_path) as img:
                    low_res_down = img.resize((400, 300), Image.LANCZOS)
                    low_res_up = low_res_down.resize((800, 600), Image.LANCZOS)
                    low_res_up.save(output_dir / "low_res.png")
                    
                    luminance = low_res_up.convert("L")
                    luminance.save(output_dir / "low_res_luminance.png")
                
                with Image.open(high_res_path) as img:
                    high_res_down = img.resize((800, 600), Image.LANCZOS)
                    high_res_down.save(output_dir / "high_res.png")
                    
                    luminance = high_res_down.convert("L")
                    luminance.save(output_dir / "high_res_luminance.png")
            
            else:
                with Image.open(low_res_path) as img:
                    low_res_img = img.resize((800, 600), Image.LANCZOS)
                    low_res_img.save(output_dir / "low_res.png")
                    
                    luminance = low_res_img.convert("L")
                    luminance.save(output_dir / "low_res_luminance.png")
                
                with Image.open(high_res_path) as img:
                    high_res_img = img.resize((800, 600), Image.LANCZOS)
                    high_res_img.save(output_dir / "high_res.png")
                    
                    luminance = high_res_img.convert("L")
                    luminance.save(output_dir / "high_res_luminance.png")
        
        except Exception:
            continue

def process_for_training_multiprocessing(image_dir_paths:  list[Path], save_path: Path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    checkpoint_file = Path("processing_checkpoint.json")
    processed_dirs = set()
    
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            processed_dirs = set(checkpoint_data.get("processed_dirs", []))
        print(f"Resuming: Found {len(processed_dirs)} already processed directories")
    
    dirs_to_process = [d for d in image_dir_paths if d.name not in processed_dirs]
    print(f"Total directories: {len(image_dir_paths)}")
    print(f"Already processed: {len(processed_dirs)}")
    print(f"Remaining to process: {len(dirs_to_process)}")

    max_workers = max(1, int(multiprocessing.cpu_count() * 0.5))
    print(f"Using {max_workers} workers for processing.\n")

    batch_size = 8196
    total_dirs = len(dirs_to_process)
    
    for start_idx in range(0, total_dirs, batch_size):
        end_idx = min(start_idx + batch_size, total_dirs)
        batch_dirs = dirs_to_process[start_idx:end_idx]
        print(f"Processing batch {start_idx // batch_size + 1}: Directories {start_idx} to {end_idx - 1} of {total_dirs}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for image_dir in batch_dirs:
                future = executor.submit(process_for_training, [image_dir], save_path)
                futures[future] = image_dir
            
            for future in futures:
                try:
                    future.result()
                    processed_dirs.add(futures[future].name)
                except Exception as e:
                    print(f"Error processing directory {futures[future].name}: {e}")
        
        with open(checkpoint_file, 'w') as f:
            json.dump({"processed_dirs": list(processed_dirs)}, f)
        print(f"Checkpoint saved: {len(processed_dirs)} directories processed\n")
    
    print(f"\nProcessing complete! Total processed: {len(processed_dirs)}")

if __name__ == "__main__":
    dump_screenshots()
    
    # source_path = Path("/media/user/2TB/imageData")
    # save_path = Path("/media/user/2TB Clear/imageData")
    # save_path.mkdir(parents=True, exist_ok=True)
    
    # all_dirs = sorted([d for d in source_path.iterdir() if d.is_dir() and d.name.startswith('image_')])
    # print(f"Found {len(all_dirs)} image directories to process")
    
    # process_for_training_multiprocessing(
    #     image_dir_paths=all_dirs,
    #     save_path=save_path
    # )
    # take_screenshots()

