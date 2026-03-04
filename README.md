# lfm2-web-runtime
The world's first dedicated WebGPU engine specifically engineered for Liquid AI's LFM2 architecture.

Current web-AI frameworks are built for Transformers. lfm2-web-runtime is different. It is the first inference engine designed from the ground up to leverage the unique Linear State Space Model (SSM) architecture of Liquid AI’s LFM2 models natively in the browser.

By moving away from generic implementations, this runtime achieves unprecedented stability and performance on consumer hardware, making high-quality AI accessible to everyone, everywhere.

💎 Why This Engine is Unique

🥇 First of its Kind: There is currently no other framework dedicated to optimizing the LFM2 architecture for WebGPU execution.

🧠 Native SSM Optimization: We leverage the linear complexity of Liquid Foundation Models. Unlike Transformers, your memory usage won't explode as the conversation gets longer.

🛡️ Anti-Crash Precompilation: We use a proprietary staged-compilation strategy. By breaking down heavy shader tasks, we prevent browser freezes and TDR (Timeout Detection and Recovery) crashes on integrated GPUs and older laptops.

🔋 Adaptive Hardware Scaling: The engine detects your device's capabilities in real-time, switching between "Lite" and "High-Performance" kernels automatically.

🏎️ Performance Profiles

Feature	💻 Low-End Device	🚀 High-End GPU
Model Variant	LFM2-350M (4-bit / INT4)	LFM2-1B+ / 350M (FP16)
Startup Strategy	Sequential Shader Compiling	Parallel Pipeline Creation
VRAM Management	Aggressive Caching (< 1GB)	Maximum Throughput Mode
Stability	Enhanced Timeout Protection	Low-Latency Inference
🛠️ Tech-Stack

Core: TypeScript & WebAssembly (Wasm)

Acceleration: WebGPU (Custom WGSL Kernels for Linear Recurrence)

Architecture: Liquid AI LFM2 (State Space Models)

Distribution: Optimized ONNX / Safetensors with external data streaming

📅 Project Roadmap

[x] Phase 1: Specialized inference logic for LFM2-350M.

[ ] Phase 2: IndexedDB Caching for instant 0-second model loading.

[ ] Phase 3: Full support for the entire LFM2 model family (1B, 3B+).

[ ] Phase 4: Multi-modal support (Vision-to-Text) within the LFM2 ecosystem.

🤝 Contribution & Status
This is an active, constantly updated project. As the first and only dedicated runtime for LFM2, we welcome contributions from the community to help push the boundaries of what's possible with Liquid AI on the web.
