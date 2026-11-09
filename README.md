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

Chat with Bimo 7B or XORTRON 27B in the same conversation. Choose the reply model next to the message box. Settings contains the Hugging Face token, XORTRON controls, and the option to clear the chat and attachments.

- **Bimo 7B** runs on the Space through [OpenHarness](https://github.com/HKUDS/OpenHarness). It uses CUDA when available and otherwise runs in 4-bit on CPU with a shorter context and output budget. The first authorized request downloads and loads the model. Set `BIMO_REVISION` to a Hub commit SHA to pin the weights.
- **XORTRON 27B** generates in the visitor's browser using [wllama](https://github.com/ngxson/wllama) and WebGPU. OpenHarness sends each generation step to the browser. Find the GGUF files and load the default Q3_K_M (about 13.5 GB) or a smaller variant. Use a recent Chromium browser with sufficient GPU memory and storage. [Standalone Space app](https://wuhp-cagents.hf.space/)
- **Tools and memory.** Both models use the OpenHarness query engine for multi-step turns. They can search and read numbered excerpts from the current chat's uploads. OpenHarness can compact the context during a turn. The browser saves the transcript and attached files when storage is available and supplies recent messages to either model. Each turn sends the message, recent history, and attachments to the Space for orchestration. Chat data is scoped to the request, and the Hugging Face token stays out of browser storage.
- **Uploads** accept up to six UTF-8 files, including text, source code, JSON, CSV, and logs. Each file can be up to 96 KB.

Enter a read token with access to the model repositories in Settings. Visitors must meet the XORTRON publisher's access terms. [Bimo repository](https://huggingface.co/iBimo-o/Bimo-7B-CyberAgent) · [XORTRON terms and files](https://huggingface.co/darkc0de/XORTRON-AgentRICO-GGUF?not-for-all-audiences=true)

## Deployment

GitHub Actions checks the source and uploads the Docker app to the Space on pushes to `main`. The repository's `HF_TOKEN` secret needs write access to the Space. The app exposes `/healthz` and `/api/status`.
