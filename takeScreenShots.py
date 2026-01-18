import time
from datetime import datetime
from pathlib import Path
from mss import mss
from PIL import Image
import random
import string
import json

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

DAMP_PATH = "/media/user/2TB/imageData"
IMAGE_WIDTH_LOW_RES = 800
IMAGE_HEIGHT_LOW_RES = 600
IMAGE_WIDTH_HIGH_RES = int(800 * 1.5)
IMAGE_HEIGHT_HIGH_RES = int(600 * 1.5)

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
                low_res_img = img.resize((IMAGE_WIDTH_LOW_RES, IMAGE_HEIGHT_LOW_RES), Image.LANCZOS)
                high_res_img = img.resize((IMAGE_WIDTH_HIGH_RES, IMAGE_HEIGHT_HIGH_RES), Image.LANCZOS)
                random_string = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=6))

                folder_name = f"image_screenShot_{random_string}_{i:08d}"
                folder_path = dst_path / folder_name
                folder_path.mkdir(parents=True, exist_ok=True)
                low_res_img.save(folder_path / "low_res.png")
                high_res_img.save(folder_path / "high_res.png")
                
                metadata = {
                    "noise_config": {"type": "screenshot", "source": "screen_capture"},
                    "num_applications": 0,
                    "high_res_size": [IMAGE_WIDTH_HIGH_RES, IMAGE_HEIGHT_HIGH_RES],
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

      

if __name__ == "__main__":
    dump_screenshots()
    # take_screenshots()
