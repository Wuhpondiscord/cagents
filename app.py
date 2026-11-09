"""Docker Space backend. All model weights are downloaded on demand."""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Annotated

import httpx
from anyio.to_thread import run_sync
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
from huggingface_hub.errors import HfHubHTTPError
from openharness.api.usage import UsageSnapshot
from pydantic import ValidationError

from agent import AgentProvider, ChatRequest, XORTRON, run_chat

BIMO = "iBimo-o/Bimo-7B-CyberAgent"
REVISION = os.environ.get("BIMO_REVISION", "main")
WEIGHT = "model-00001-of-00004.safetensors"
MAX_INPUT = 8192
MAX_OUTPUT = 512
CPU_MAX_INPUT = 4096
CPU_MAX_OUTPUT = 256
WEB = Path(__file__).resolve().parent / "web"
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/assets", StaticFiles(directory=WEB), name="assets")
model_cache = None
active_job = None
bimo_phase = None
logger = logging.getLogger("cagents")


@app.middleware("http")
async def set_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/healthz")
def health():
    import torch

    return {"status": "ok", "bimo_cuda": torch.cuda.is_available()}


@app.get("/api/status")
def status():
    return {"bimo": bimo_phase or ("online" if model_cache else "needs_load")}


class ReviewError(ValueError):
    """An error message safe to return to the current visitor."""


def authorized_revision(token):
    try:
        HfApi().whoami(token=token)
        metadata = get_hf_file_metadata(
            hf_hub_url(BIMO, WEIGHT, revision=REVISION), token=token, timeout=15
        )
    except (HfHubHTTPError, httpx.HTTPStatusError) as exc:
        status = exc.response.status_code if exc.response is not None else 0
        if status in {401, 403, 404}:
            raise ReviewError(
                "Hugging Face denied access. Check your read token and Bimo model access."
            ) from None
        raise ReviewError("Hugging Face could not verify Bimo access. Try again shortly.") from None
    if not metadata.commit_hash:
        raise ReviewError("Hugging Face did not return a Bimo model revision.")
    return metadata.commit_hash


def load_bimo(token):
    global model_cache
    import torch

    revision = authorized_revision(token)
    if model_cache is not None and model_cache[0] == revision:
        return model_cache[1:]

    from transformers import AutoTokenizer

    if model_cache is not None:
        model_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    source = {"revision": revision, "token": token, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(BIMO, **source)
    model = load_quantized_model(BIMO, **source)
    model_cache = (revision, tokenizer, model)
    return tokenizer, model


def load_quantized_model(model_id, **source):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    has_cuda = torch.cuda.is_available()
    device = "cuda:0" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda and not torch.cuda.is_bf16_supported() else torch.bfloat16
    if not has_cuda:
        torch.set_num_threads(2)
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        **source,
        use_safetensors=True,
        device_map={"": device},
        dtype=dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    ).eval()


class BimoProvider(AgentProvider):
    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

    def complete(self, request):
        import torch

        messages = self.build_messages(request)
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        input_size = encoded["input_ids"].shape[-1]
        on_cpu = self.model.device.type == "cpu"
        input_limit = CPU_MAX_INPUT if on_cpu else MAX_INPUT
        if input_size > input_limit:
            raise ReviewError(f"The Bimo input exceeds {input_limit:,} tokens. Shorten your message.")
        output_limit = CPU_MAX_OUTPUT if on_cpu else MAX_OUTPUT
        encoded = encoded.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=min(request.max_tokens, output_limit),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        tokens = output[0, input_size:]
        answer = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return answer, UsageSnapshot(input_tokens=input_size, output_tokens=len(tokens))

    async def generate(self, request):
        return await run_sync(self.complete, request)


class BrowserProvider(AgentProvider):
    def __init__(self, socket):
        super().__init__()
        self.socket = socket

    async def generate(self, request):
        await self.socket.send_json({
            "generate": self.build_messages(request),
            "max_tokens": request.max_tokens,
        })
        result = await asyncio.wait_for(self.socket.receive_json(), timeout=300)
        if not isinstance(result, dict):
            raise ReviewError("The browser returned an invalid response.")
        if result.get("error"):
            raise ReviewError("XORTRON could not generate. Try a shorter message or reload the model.")
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 16000:
            raise ReviewError("XORTRON did not return a usable response.")
        return answer.strip(), UsageSnapshot()


async def review_bimo(review, token, updates):
    global bimo_phase
    try:
        await updates.put({"status": "Checking access and loading Bimo if needed…"})
        tokenizer, model = await run_sync(load_bimo, token)
        provider = BimoProvider(tokenizer, model)
        bimo_phase = "busy"
        device = "CPU" if model is not None and model.device.type == "cpu" else "GPU"
        await updates.put({"status": f"OpenHarness is running Bimo on the Space {device}…"})
        answer = await run_chat(provider, BIMO, review, updates.put, output_tokens=MAX_OUTPUT)
        if not answer:
            raise ReviewError(provider.error or "Bimo returned no answer. Try a shorter message.")
        await updates.put({"answer": answer})
    except ReviewError as exc:
        await updates.put({"error": str(exc)})
    except Exception as exc:
        logger.error("Bimo chat failed (%s)", type(exc).__name__)
        await updates.put({"error": "Bimo could not load or generate. Check model access and available Space memory."})
    finally:
        bimo_phase = None
        await updates.put(None)


@app.post("/api/bimo")
async def bimo(review: ChatRequest, authorization: Annotated[str | None, Header()] = None):
    global active_job, bimo_phase
    match = re.fullmatch(r"Bearer (hf_[A-Za-z0-9]{10,250})", authorization or "")
    if not match:
        raise HTTPException(401, "Enter a valid Hugging Face read token.")
    if active_job is not None and not active_job.done():
        raise HTTPException(429, "Bimo is busy. Retry after the current review finishes.")
    updates = asyncio.Queue()
    bimo_phase = "checking" if model_cache else "loading"
    active_job = asyncio.create_task(review_bimo(review, match[1], updates))

    async def events():
        while True:
            try:
                update = await asyncio.wait_for(updates.get(), 8)
            except asyncio.TimeoutError:
                yield ": working\n\n"
                continue
            if update is None:
                return
            yield f"data: {json.dumps(update)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})


@app.websocket("/api/xortron")
async def xortron(socket: WebSocket):
    origin = socket.headers.get("origin")
    host = os.environ.get("SPACE_HOST") or socket.headers.get("host")
    if origin not in {f"https://{host}", f"http://{host}"}:
        await socket.close(code=1008)
        return
    await socket.accept()
    try:
        payload = await asyncio.wait_for(socket.receive_json(), timeout=15)
        chat = ChatRequest.model_validate(payload)
        provider = BrowserProvider(socket)
        answer = await run_chat(provider, XORTRON, chat, socket.send_json, output_tokens=384)
        if not answer:
            raise ReviewError(provider.error or "XORTRON returned no answer. Try a shorter message.")
        await socket.send_json({"answer": answer})
    except WebSocketDisconnect:
        return
    except (ValidationError, json.JSONDecodeError):
        await socket.send_json({"error": "Invalid chat request."})
    except (ReviewError, TimeoutError) as exc:
        await socket.send_json({"error": str(exc) or "The browser model did not respond in time."})
    except Exception as exc:
        logger.error("Browser chat failed (%s)", type(exc).__name__)
        await socket.send_json({"error": "The browser chat was interrupted. Try again."})
    finally:
        try:
            await socket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass
