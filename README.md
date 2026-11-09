---
title: cagents
emoji: 🔎
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
suggested_hardware: l4x1
---

# cagents

A two-model workspace for authorized defensive security review.

1. Request access to [Bimo 7B](https://huggingface.co/iBimo-o/Bimo-7B-CyberAgent) and [XORTRON AgentRICO](https://huggingface.co/darkc0de/XORTRON-AgentRICO-GGUF?not-for-all-audiences=true) as needed. Enter a Hugging Face read token with permission to access the models. GitHub's deployment secret is separate.
2. Describe the task and run Bimo. The Space uses [HKUDS OpenHarness](https://github.com/HKUDS/OpenHarness) with an in-process Bimo provider and an empty tool registry.
3. Find and load an approved XORTRON GGUF using [wllama](https://github.com/ngxson/wllama). The default is the publisher’s 3-bit-class Q3_K_M file (about 13.5 GB). Q3_K_S and Q2_K are smaller choices if your device is short on memory. Open the [standalone Space app](https://wuhp-cagents.hf.space/) if WebGPU is unavailable in the Hugging Face iframe.
4. Run an independent browser review with Bimo's draft as optional context. To send that browser review back to the Space, select the notes checkbox before running Bimo again. Both outputs are text for human review.

## Runtime

- Bimo uses NVIDIA CUDA when available and otherwise runs with a shorter context and output budget on CPU. NVIDIA L4 is the suggested hardware for higher throughput. Bimo downloads its original safetensors on the Space at first authorized use and quantizes to NF4 when loading. The process retains one loaded model revision and checks the current visitor's access before reusing it. Only one Bimo generation runs at a time. Set the optional Space variable `BIMO_REVISION` to a commit SHA to pin the source revision.
- XORTRON uses the browser's GPU and needs a WebGPU-compatible browser, sufficient GPU memory, and storage for the selected GGUF. A current Chromium browser with WebAssembly JSPI can read large files in chunks. Adjust the GPU layer count if the full model does not fit in VRAM. The browser sends its read token directly to Hugging Face and verifies access before using cached weights. Model files can remain in site storage after unload. The page never stores the token in site storage.
- The Space does not package any model weights. OpenHarness uses a one-turn text-review loop without filesystem, shell, or network tools. Users must meet the XORTRON publisher's access terms.

## Deployment

The GitHub Actions workflow checks the source and syncs the Docker app to [wuhp/cagents](https://huggingface.co/spaces/wuhp/cagents) on pushes to `main`. The repository's `HF_TOKEN` secret needs write access to the destination Space. Hugging Face then builds the Docker image. Select GPU hardware in the Space settings if you want faster Bimo inference. The app exposes `/healthz` with a CUDA availability flag.
