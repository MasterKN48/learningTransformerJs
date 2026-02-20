# Transformer.Vision 🚀

A professional-grade, browser-based object detection suite powered by **Transformers.js (v3)**, **React**, and **Tailwind CSS v4**. This application executes state-of-the-art computer vision models entirely on the client-side using ONNX Runtime.

## ⚙️ System Features

- **Execution Provider Selection**: Manually choose between hardware acceleration backends.
  - **WebGPU**: High-performance GPU acceleration.
  - **WASM**: Multi-threaded WebAssembly execution.
  - **WebNN**: Native neural network API.
  - **CPU**: Standard execution.
- **Hardware Detection**: Live detection of browser capabilities (WebGPU/WebNN support status shown in UI).
- **File Constraints**: Built-in 10MB safety limit for image uploads to prevent browser memory exhaustion.
- **Dynamic Quantization**: Toggle 8-bit quantization for reduced memory footprint.
- **Persistent Storage**: Models are cached in the browser's **Cache API / IndexedDB**, meaning they only download once.
- **Env-Driven Configuration**: Easily change the underlying model architecture via `.env`.

## 🛠️ Advanced Configuration

The application reads the default model from your environment:

```env
VITE_MODEL_NAME=onnx-community/rfdetr_medium-ONNX
```

### Execution Backends Explained

1. **WebGPU**: The fastest backend for modern browsers (Chrome/Edge 113+). It maps neural network operations directly to your GPU.
2. **WASM (WebAssembly)**: Highly optimized for CPUs. Uses SIMD and multi-threading for solid performance on devices without modern GPU access.
3. **Quantization**: Converts 32-bit floating point weights to 8-bit integers. This reduces model size by ~75% (e.g., from 100MB to 25MB) and significantly speeds up processing on mobile/low-end devices.

## 🚀 Getting Started

### Prerequisites

- [Bun](https://bun.sh/) (Recommended) or Node.js.

### Installation & Run

1. **Install dependencies**:
   ```bash
   bun install
   ```
2. **Configure environment**:
   Create a `.env` file (already initialized in this repo):
   ```bash
   VITE_MODEL_NAME=onnx-community/rfdetr_medium-ONNX
   ```
3. **Launch Dev Suite**:
   ```bash
   bun dev
   ```

## 🧠 Technical Architecture

- **AI Engine**: `@huggingface/transformers`
- **UI Framework**: React 19 + Lucide Icons
- **Design System**: Tailwind CSS v4 (Modern Glassmorphism)
- **Model Storage**: Browser Cache Storage API

## 📝 Privacy

Images uploaded to this tool **never leave your local machine**. All computation is performed within your browser's memory space, making it ideal for processing sensitive visual data.

---

Built for learning Advanced Transformers.js implementation.
