import re
import time
import numpy as np
from PIL import Image
import pytesseract
from mss import mss
from obsws_python import ReqClient, error
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# OBS WebSocket configuration
HOST = "localhost"
PORT = 4455
PASSWORD = "P4o2I0ozqbtCvwKc"
SOURCE_NAME = "Presentation"
FILTER_NAME_PREFIX = "BlurAPIKey_"
FILTER_KIND = "obs_composite_blur"  # Replace with the correct filterKind after verification

# Regex patterns for API keys
API_KEY_PATTERNS = {
    "OpenAI_API_Key": r'sk-proj-[a-zA-Z0-9_-]{50,150}',
    "Gemini_API_Key": r'AI[a-zA-Z0-9_-]{30,40}',
    "HF_Bearer": r'hf_[a-zA-Z0-9]{30,40}'
}

# OCR and blur settings
CHECK_INTERVAL = 0.1
BLUR_RADIUS = 20
BLUR_MARGIN = 20
BLUR_ALGORITHM = "pixelate"  # Options: gaussian, box, bilateral, anisotropic, pixelate

# OBS WebSocket connection
ws = ReqClient(host=HOST, port=PORT, password=PASSWORD)

def capture_screen():
    with mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        return img

def detect_api_keys(image):
    try:
        img = image.convert('L')
        ocr_result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        text = ocr_result['text']
        left = ocr_result['left']
        top = ocr_result['top']
        width = ocr_result['width']
        height = ocr_result['height']

        detected_regions = []
        for key_name, pattern in API_KEY_PATTERNS.items():
            patterns = [
                pattern,
                rf'{key_name}\s*=\s*["\']?{pattern}["\']?'
            ]
            for pat in patterns:
                for i, word in enumerate(text):
                    if re.search(pat, word, re.IGNORECASE):
                        logging.info(f"Detected {key_name} at position {left[i]},{top[i]}")
                        detected_regions.append({
                            'key_name': key_name,
                            'left': max(0, left[i] - BLUR_MARGIN),
                            'top': max(0, top[i] - BLUR_MARGIN),
                            'width': max(1, width[i] + 2 * BLUR_MARGIN),
                            'height': max(1, height[i] + 2 * BLUR_MARGIN)
                        })
                    combined_text = ' '.join(text[max(0, i-3):i+4])
                    if re.search(pat, combined_text, re.IGNORECASE):
                        logging.info(f"Detected {key_name} in combined text at {left[i]},{top[i]}")
                        detected_regions.append({
                            'key_name': key_name,
                            'left': max(0, left[i] - BLUR_MARGIN),
                            'top': max(0, top[i] - BLUR_MARGIN),
                            'width': max(1, width[i] + 2 * BLUR_MARGIN),
                            'height': max(1, height[i] + 2 * BLUR_MARGIN)
                        })
        return detected_regions
    except Exception as e:
        logging.error(f"OCR error: {e}")
        return []

def apply_blur_filter(region, filter_name):
    try:
        if region['width'] <= 0 or region['height'] <= 0:
            logging.warning(f"Invalid region for {filter_name}: {region}")
            return

        try:
            ws.get_source_filter(SOURCE_NAME, filter_name)
            filter_exists = True
            logging.debug(f"Filter {filter_name} exists")
        except error.OBSSDKRequestError as e:
            filter_exists = False
            logging.debug(f"Filter {filter_name} does not exist: {e}")

        if not filter_exists:
            settings = {
                "blur_algorithm": BLUR_ALGORITHM,
                "blur_radius": BLUR_RADIUS,
                "mask_enabled": True,
                "mask_left": region['left'],
                "mask_top": region['top'],
                "mask_width": region['width'],
                "mask_height": region['height'],
                "opacity": 100.0
            }
            logging.debug(f"Creating filter {filter_name} with settings: {settings}")
            ws.create_source_filter(SOURCE_NAME, filter_name, FILTER_KIND, settings)
            logging.info(f"Applied blur filter {filter_name} at {region}")
        else:
            settings = {
                "mask_left": region['left'],
                "mask_top": region['top'],
                "mask_width": region['width'],
                "mask_height": region['height'],
                "blur_radius": BLUR_RADIUS,
                "opacity": 100.0
            }
            logging.debug(f"Updating filter {filter_name} with settings: {settings}")
            ws.set_source_filter_settings(SOURCE_NAME, filter_name, settings)
            logging.info(f"Updated blur filter {filter_name} at {region}")
    except error.OBSSDKRequestError as e:
        logging.error(f"Error applying blur filter {filter_name}: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Unexpected error applying blur filter {filter_name}: {e}", exc_info=True)

def remove_blur_filter(filter_name):
    try:
        ws.remove_source_filter(SOURCE_NAME, filter_name)
        logging.info(f"Removed blur filter {filter_name}")
    except error.OBSSDKRequestError as e:
        if "filter not found" not in str(e).lower():
            logging.error(f"Error removing blur filter {filter_name}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error removing blur filter {filter_name}: {e}")

def main():
    active_filters = {}
    try:
        while True:
            img = capture_screen()
            detected_regions = detect_api_keys(img)
            current_keys = {region['key_name'] for region in detected_regions}
            for key_name in list(active_filters.keys()):
                if key_name not in current_keys:
                    remove_blur_filter(active_filters[key_name])
                    del active_filters[key_name]
            for region in detected_regions:
                key_name = region['key_name']
                filter_name = f"{FILTER_NAME_PREFIX}{key_name}"
                apply_blur_filter(region, filter_name)
                active_filters[key_name] = filter_name
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        logging.info("Script terminated by user")
    finally:
        for filter_name in active_filters.values():
            remove_blur_filter(filter_name)

if __name__ == "__main__":
    main()