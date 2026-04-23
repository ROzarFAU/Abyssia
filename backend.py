"""
Austin Pritchard

Backend - Underwater Scene Analysis
-------------------------------------------
Combines CLIP (zero-shot classification) and Qwen2.5-VL (caption generation)
to analyze underwater images. In addition, it uses a base CLIP model to detect
if the image appears to be underwater.

Usage (standalone):
    python backend.py path/to/your/image.jpg

Usage (frontend import through Python):
    from backend import analyze_image
    result = analyze_image("path/to/image.jpg")
    print(result["caption"])
    print(result["top_label"])
    print(result["labels"])

Setup:
    pip install onnxruntime transformers torch torchvision qwen-vl-utils pillow numpy (run this command on the terminal/command prompt to install the required libraries)
    Place the underwater_clip_optimized folder in the same directory as this file.
"""

import os
import sys
import numpy as np
import torch
import onnxruntime as ort
from PIL import Image
from transformers import (
    CLIPProcessor,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

# Configuration
# --------------- 

CLIP_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "underwater_clip_optimized")
CLIP_MODEL_FILE = "model_quantized.onnx"
QWEN_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
BASE_CLIP_MODEL = "openai/clip-vit-base-patch32"  # general-purpose CLIP for underwater detection

UNDERWATER_LABELS = [
    "coral",
    "fish",
    "shipwreck",
    "diver",
    "jellyfish",
    "sea turtle",
    "shark",
    "seaweed",
    "stingray",
    "octopus",
]



CAPTION_PROMPT = (
    "You are an expert marine biologist. Describe what you see in this underwater "
    "image in one or two clear sentences. Focus on the marine life, environment, "
    "and any notable features visible in the scene."
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGE_SIZE = 640

# Model Loading
# --------------

print("Loading CLIP model...")
clip_onnx_path = os.path.join(CLIP_MODEL_DIR, CLIP_MODEL_FILE)
clip_session = ort.InferenceSession(clip_onnx_path)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_DIR, local_files_only=True)
print("CLIP model loaded.")

print("Loading base CLIP model for underwater detection...")
from transformers import CLIPModel, CLIPProcessor as CLIPProc
base_clip_model = CLIPModel.from_pretrained(BASE_CLIP_MODEL)
base_clip_processor = CLIPProc.from_pretrained(BASE_CLIP_MODEL)
base_clip_model.eval()
print("Base CLIP model loaded.")

print("Loading Qwen2.5-VL model (this may take a moment)...")
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_PATH,
    torch_dtype=torch.bfloat16,
).to(DEVICE)
qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
print(f"Qwen model loaded. Running on: {DEVICE}")

# Core Functions
# ----------------

def analyze_image(image_path: str) -> dict:
    """
    Main function. Takes an image file path and returns a dict with:
        - caption    : natural-language description of the scene
        - top_label  : the most likely underwater category
        - labels     : all categories with confidence scores, sorted highest first

    Args:
        image_path: path to a JPEG or PNG image file

    Returns:
        {
            "caption": "A coral reef with several fish...",
            "top_label": "coral",
            "labels": [
                {"label": "coral", "confidence": "94.21%"},
                {"label": "fish",  "confidence": "3.10%"},
                ...
            ]
        }
    """
    # Validate file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No file found at: {image_path}")

    # Validate file extension
    allowed_extensions = {".jpg", ".jpeg", ".png", ".avif"}
    ext = os.path.splitext(image_path)[-1].lower()
    if ext not in allowed_extensions:
        raise ValueError(f"Invalid file type '{ext}'. Only JPEG, JPG, AVIF and PNG are accepted.")

    image = Image.open(image_path).convert("RGB")
    image.thumbnail([MAX_IMAGE_SIZE, MAX_IMAGE_SIZE], Image.Resampling.LANCZOS)

    is_underwater = _is_underwater(image)
    caption = _caption(image)
    labels = _classify(image) if is_underwater else []

    return {
        "caption": caption,
        "top_label": labels[0]["label"] if labels else "N/A",
        "labels": labels,
        "is_underwater": is_underwater,
    }


def _is_underwater(image: Image.Image) -> bool:
    """
    Uses the base OpenAI CLIP model (trained on broad image-text pairs)
    to determine if an image is underwater. Compares underwater vs
    non-underwater descriptions and returns True if underwater wins.
    """
    underwater_texts = [
        "an underwater photo",
        "a photo taken underwater",
        "marine life underwater",
        "an underwater ocean scene",
    ]
    non_underwater_texts = [
        "a photo taken above water",
        "an indoor photo",
        "a photo of an object",
        "a photo taken on land",
    ]

    all_texts = underwater_texts + non_underwater_texts

    inputs = base_clip_processor(
        text=all_texts,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = base_clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    underwater_score = float(probs[:len(underwater_texts)].sum())
    non_underwater_score = float(probs[len(underwater_texts):].sum())

    return underwater_score > non_underwater_score


def _classify(image: Image.Image) -> list:
    """
    Run CLIP classification using the ONNX session directly.
    Returns labels sorted by confidence descending.
    """
    inputs = clip_processor(
        text=UNDERWATER_LABELS,
        images=image,
        return_tensors="pt",  
        padding=True,
    )

    inputs = {k: v.cpu().numpy() for k, v in inputs.items()}

    # Only pass inputs the ONNX model actually expects
    expected_inputs = {inp.name for inp in clip_session.get_inputs()}
    ort_inputs = {k: v for k, v in inputs.items() if k in expected_inputs}

    ort_outputs = clip_session.run(None, ort_inputs)

    # ort_outputs[0] = logits_per_image shape (1, num_labels)
    logits = ort_outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    results = [
        {"label": label, "confidence": f"{prob:.2%}"}
        for label, prob in zip(UNDERWATER_LABELS, probs)
    ]
    results.sort(key=lambda x: float(x["confidence"].strip("%")), reverse=True)
    return results


def _caption(image: Image.Image) -> str:
    """Run Qwen2.5-VL and return a natural-language caption for the image."""
    messages = [
        {"role": "system", "content": "You are a helpful marine biology assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image", "image": image},
            ],
        },
    ]

    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = qwen_processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    output_ids = qwen_model.generate(**inputs, max_new_tokens=150)
    generated_ids = [
        output_ids[i][len(inputs.input_ids[i]):]
        for i in range(len(inputs.input_ids))
    ]
    caption = qwen_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    return caption.strip()


# Standalone Entry Point
# ---------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backend.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"\nAnalyzing: {image_path}\n")

    result = analyze_image(image_path)

    print(f"Caption:   {result['caption']}")
    print(f"Top Label: {result['top_label']}")
    print("\nAll Labels:")
    for item in result["labels"]:
        print(f"  {item['label']}: {item['confidence']}")