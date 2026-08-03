# TensorFlow Lite Micro for ESP32 with Custom Examples

This repository provides a localized and optimized port of **TensorFlow Lite Micro (TFLM)** specifically tailored for **ESP32** microcontrollers. It includes out-of-the-box custom TinyML project examples to jumpstart your embedded machine learning journey on ESP32 hardware.

## 🚀 Features
* **ESP32 Optimized:** Configured to run efficiently on ESP32 development boards.
* **Custom Examples Included:** Pre-bundled, ready-to-flash TinyML deployment examples (including model conversion and inference pipelines).
* **Simplified Structure:** Clean codebase that eliminates library clutter, making it easy to integrate into your own projects.
* **Quantization Support:** Full support for `int8` quantized models to maximize performance and minimize memory (`tensor_arena`) footprint.

## 🛠️ Getting Started
1. Clone this repository to your local machine or Arduino/ESP-IDF components directory.
2. Open one of the provided example projects.
3. Replace the `model.h` file with your own trained TensorFlow Lite model array if needed.
4. Compile and flash it directly to your ESP32!
