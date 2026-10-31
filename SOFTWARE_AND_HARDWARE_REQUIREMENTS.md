# DMESR Emotion-Aware AI — Software and Hardware Requirements Specification

**Document Version:** 1.0.0  
**Project:** DMESR (Deep Multimodal Emotion State Recurrent AI) & Visual Chat 3D Avatar System  
**Date:** October 2026  
**Status:** Active Production / Development  

---

## 1. Executive Summary

DMESR is an end-to-end multimodal emotion-aware conversational AI platform. It perceives, fuses, and responds to human emotional states across four distinct channels in real time:
1. **Text Emotion (Verbal):** RoBERTa (`SamLowe/roberta-base-go_emotions`)
2. **Vocal Emotion (Acoustic SER):** Wav2Vec 2.0 (`superb/wav2vec2-base-superb-er`)
3. **Facial Emotion (Visual FER):** Vision Transformer (`HardlyHumans/Facial-expression-detection` + OpenCV Haar Cascade)
4. **Multimodal Emotion State Fusion:** Deep State-Space Recurrent Fusion Model (`dmesr_mosei_best.pt`)

The system couples multimodal perception with an LLM conversational reasoning layer (Qwen 27B / LLaMA via Groq API), streaming neural Text-to-Speech (Edge-TTS female voice synthesis engine), and real-time interactive 3D avatars (Sky and Tufrilla VRM female models) featuring procedural audio-driven lip-sync and affective blendshape expressions.

---

## 2. Hardware Requirements

### 2.1 Server / Backend Host Machine

| Component | Minimum Specification (Functional) | Recommended Specification (Optimal Performance) |
| :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i5 (10th Gen+) or AMD Ryzen 5 3600 (6 Cores / 12 Threads) | Intel Core i7 / i9 (12th Gen+) or AMD Ryzen 7 / 9 5800X+ (8+ Cores / 16 Threads) |
| **Graphics (GPU)** | NVIDIA GPU with **6 GB VRAM** (e.g. GTX 1660 Ti, RTX 2060, RTX 3060 Laptop) | NVIDIA GPU with **8 GB - 16 GB VRAM** (e.g. RTX 3070, RTX 4070, RTX 4080, T4, A10G) |
| **CUDA Compute** | Compute Capability 7.5+ | Compute Capability 8.0+ (Ampere, Ada Lovelace) |
| **System Memory (RAM)** | **16 GB DDR4** | **32 GB DDR4 / DDR5** |
| **Storage (Disk)** | 25 GB free space (SATA SSD) | 50 GB free space (NVMe M.2 SSD, >= 2500 MB/s read) |
| **Network Interface** | 10 Mbps broadband connection | 100+ Mbps low-latency broadband connection |

> **CPU-Only Fallback Note:** While all PyTorch models will initialize and execute on CPU (`device="cpu"`), parallel inference latency will increase from ~120–250 ms (CUDA) to ~900–2200 ms (CPU), impacting real-time conversational fluidity.

---

### 2.2 Client / End-User Machine (Browser Frontend)

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Processor (CPU)** | Dual-Core x86_64 or Apple Silicon M1 | Quad-Core 2.5 GHz+ or Apple Silicon M2/M3 |
| **Memory (RAM)** | 8 GB RAM | 16 GB RAM |
| **Graphics (GPU)** | Integrated Graphics with WebGL 2.0 (Intel Iris Xe, AMD Radeon Vega) | Dedicated GPU (NVIDIA GTX 1050 / Apple M-series GPU) |
| **Display** | 1280 × 720 (HD) | 1920 × 1080 (Full HD) or higher |
| **Audio Input** | Integrated or USB Microphone (16 kHz minimum sampling) | Directional USB or Headset Microphone with noise cancellation |
| **Audio Output** | Standard stereo speakers or headphones | Low-latency stereo headphones |
| **Video Input** | Standard 720p 30fps USB / Built-in Webcam | 1080p 30/60fps Webcam with good front lighting |

---

## 3. Software Requirements

### 3.1 Operating System Compatibility

| Operating System | Support Level | Notes |
| :--- | :--- | :--- |
| **Microsoft Windows** | Primary / Full Support | Windows 10 (Build 19044+) or Windows 11 (64-bit). Verified active environment. |
| **Linux** | Primary / Full Support | Ubuntu 20.04 / 22.04 / 24.04 LTS, Debian 12, Arch Linux (64-bit). |
| **macOS** | Supported (CPU/MPS) | macOS Monterey 12.0+ (Apple Silicon M1/M2/M3 native via MPS backend). |

---

### 3.2 Backend Runtime & Dependencies

#### 3.2.1 Core Runtime
* **Python:** 3.10.x, 3.11.x, or 3.12.x (Verified: Python 3.11 / 3.12 via Miniconda / Anaconda)
* **CUDA Toolkit:** CUDA 11.8 or CUDA 12.1 / 12.4 with compatible NVIDIA Display Driver (>= 535.xx)
* **FFmpeg:** Installed and added to system `PATH` (essential for WebM/Opus audio stream decoding)

#### 3.2.2 Python Package Dependencies (`backend/requirements.txt`)

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `fastapi` | `==0.115.0` | Asynchronous REST and WebSocket API web framework |
| `uvicorn[standard]` | `==0.30.0` | High-performance ASGI production server |
| `python-multipart` | `==0.0.9` | Multipart form-data streaming parser (audio/video uploads) |
| `torch` | `>=2.0.0` | Core deep learning and tensor computation engine (CUDA-enabled) |
| `torchvision` | `>=0.15.0` | Image transforms and computer vision utilities |
| `torchaudio` | `>=2.0.0` | Audio signal processing, spectrograms, and waveform transforms |
| `transformers` | `>=4.40.0` | Hugging Face transformer models (ViT, Wav2Vec2, RoBERTa) |
| `openai-whisper` | `>=20231117` | Local Automatic Speech Recognition (ASR) fallback |
| `opencv-python-headless` / `opencv-python` | `>=4.8.0` | Real-time face detection (Haar Cascade) and image preprocessing |
| `numpy` | `>=1.24.0` | Matrix math and numerical array processing |
| `scipy` | `>=1.11.0` | Audio DSP, resampling, and spatial filtering |
| `pydantic` | `>=2.0.0` | Request/response data validation and serialization |
| `edge-tts` | `>=6.1.0` | Real-time Microsoft Edge neural speech synthesis |
| `httpx` | `>=0.27.0` | Asynchronous HTTP client for external service dispatch |
| `groq` | `>=0.9.0` | Groq Cloud SDK (Whisper-large-v3-turbo STT and Qwen/LLaMA LLM) |
| `openai` | `>=1.30.0` | OpenAI API client compatibility layer |
| `google-genai` | `>=1.0.0` | Google Gemini API SDK (optional multi-LLM provider) |

---

### 3.3 Frontend Runtime & Dependencies

#### 3.3.1 Core Runtime
* **Node.js:** v18.x, v20.x LTS, or v22.x LTS
* **Package Manager:** `npm` (v9.x+), `pnpm` (v8.x+), or `yarn` (v1.22+)

#### 3.3.2 Frontend Package Dependencies (`frontend/package.json`)

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `react` | `^19.0.0` | Declarative UI framework |
| `react-dom` | `^19.0.0` | React DOM rendering engine |
| `vite` | `^6.0.0` | Modern, ultra-fast frontend build tool and dev server |
| `@vitejs/plugin-react` | `^4.4.0` | Fast Refresh and JSX transformation for Vite |
| `tailwindcss` | `^4.0.0` | Utility-first CSS design framework |
| `@tailwindcss/vite` | `^4.0.0` | First-class Tailwind CSS v4 compiler integration |
| `three` | `^0.172.0` | WebGL 3D rendering library |
| `@react-three/fiber` | `^9.0.0` | React reconciler for Three.js scenes |
| `@react-three/drei` | `^10.0.0` | Three.js helpers (OrbitControls, loaders, lighting) |
| `@pixiv/three-vrm` | `^3.5.5` | Humanoid VRM avatar loader and expression/blendshape controller |
| `zustand` | `^5.0.0` | Global state management for sessions, modes, and emotion history |
| `axios` | `^1.7.0` | HTTP client for REST endpoints with reverse-proxy support |
| `uuid` | `^11.0.0` | Unique turn ID and session ID generator |

---

### 3.4 Web Browser Specifications

The client application operates entirely within modern web browsers with zero desktop client installation required.

| Browser | Minimum Version | Required Web APIs |
| :--- | :--- | :--- |
| **Google Chrome** | 109+ (Desktop & Mobile) | WebGL 2.0, MediaDevices (`getUserMedia`), Web Audio API, MediaRecorder |
| **Microsoft Edge** | 109+ (Desktop) | WebGL 2.0, MediaDevices, Web Audio API, MediaRecorder |
| **Mozilla Firefox** | 115+ (Desktop) | WebGL 2.0, MediaDevices, Web Audio API, MediaRecorder |
| **Apple Safari** | 16.4+ (macOS & iOS) | WebGL 2.0, MediaDevices, Web Audio API |

---
