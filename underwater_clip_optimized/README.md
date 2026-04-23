\# 🌊 Underwater CLIP - Optimized Edge Deployment Bundle



\*\*Project Phase:\*\* 4.0 (Optimization \& Latency Benchmarking)  

\*\*Hand-off From:\*\* Cosme Ray Penney  

\*\*Hand-off To:\*\* Austin (Web Integration Lead)  



\## 🚀 Overview

This folder contains the final, high-performance version of our domain-adapted CLIP model. I have converted the original PyTorch weights into an \*\*8-bit Quantized ONNX\*\* format. This version is specifically tuned to run fast on standard CPUs, making it ideal for our web-based dashboard and edge deployment.



\## 📊 Performance Benchmarks

These metrics were captured on an ASUS Zephyrus G16 (Intel i9) to verify deployment readiness:



\* \*\*Accuracy:\*\* 99.21% (Verified against Isaac Sim underwater shipwreck data)

\* \*\*Model Size:\*\* ~145 MB (Reduced from 590 MB — \*\*75% Compression\*\*)

\* \*\*Latency:\*\* 15.38 ms per image

\* \*\*Speedup:\*\* \*\*3.90x Faster\*\* than the original PyTorch model



\## 📂 Included Files

\* `model\_quantized.onnx`: The core 8-bit quantized neural network.

\* `preprocessor\_config.json`: Image normalization and resizing rules (Fixed for ONNX).

\* `config.json`: Master model architecture blueprint.

\* `tokenizer.json` \& `tokenizer\_config.json`: Text-to-vector mapping files.



\## 🛠 Integration (How to use this, Austin)

You will need the `optimum` and `onnxruntime` libraries to run this. They are much lighter than a full Torch installation.



\### 1. Install Dependencies

```bash

pip install onnxruntime optimum transformers

# Use the ORTModel class from the Optimum library. It mimics the Hugging Face API but uses the fast ONNX backend.

from optimum.onnxruntime import ORTModelForZeroShotImageClassification

from transformers import CLIPProcessor



\# Path to this folder

model\_dir = "./underwater\_clip\_optimized"



\# Load the optimized brain

model = ORTModelForZeroShotImageClassification.from\_pretrained(

&nbsp;   model\_dir, 

&nbsp;   file\_name="model\_quantized.onnx"

)



\# Load the processor

processor = CLIPProcessor.from\_pretrained(model\_dir)



\# Now you can use 'model' and 'processor' in your inference pipeline!

