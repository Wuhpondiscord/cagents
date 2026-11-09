const $ = (id) => document.getElementById(id);
const HF = "https://huggingface.co";
const XOR = "darkc0de/XORTRON-AgentRICO-GGUF";
const PREFERRED_GGUF = /[._-]q3_k_m(?:-\d{5}-of-\d{5})?\.gguf$/i;
const TOKEN = /^hf_[A-Za-z0-9]{10,250}$/;
let llama = null;
let options = [];
let browserBusy = false;


const memoryKey = "cagents-chat-v1";
const labels = { bimo: "Bimo 7B", xor: "XORTRON 27B" };
const states = {
  online: "Online", needs_load: "Needs load", busy: "Working", loading: "Loading…",
  checking: "Checking…", offline: "Offline", unavailable: "WebGPU unavailable",
};
let transcript = [];
let uploads = [];
let sending = false;
let readingFiles = false;

$("embed").hidden = window.self === window.top;
$("clear").onclick = () => { $("token").value = ""; };
$("show-chat").onclick = () => showSettings(false);
$("show-settings").onclick = () => showSettings(true);

function showSettings(show) {
  $("chat").hidden = show;
  $("settings").hidden = !show;
  $("show-chat").setAttribute("aria-pressed", String(!show));
  $("show-settings").setAttribute("aria-pressed", String(show));
  $(show ? "settings-title" : "prompt").focus();
}

function readToken() {
  const token = $("token").value.trim();
  if (!TOKEN.test(token)) throw new Error("Enter a Hugging Face read token in Settings.");
  return token;
}

function setModelState(model, state) {
  $(model + "-dot").dataset.state = state;
  $(model + "-state").textContent = states[state] ?? states.offline;
}

async function refreshBimo() {
  try {
    const response = await fetch("/api/status", { cache: "no-store", signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error();
    setModelState("bimo", (await response.json()).bimo);
  } catch {
    setModelState("bimo", "offline");
  }
}

function setControls() {
  $("send").disabled = sending || browserBusy || readingFiles;
  $("model").disabled = sending;
  $("attach").disabled = sending || readingFiles;
  $("new-chat").disabled = sending || readingFiles;
  $("files").disabled = sending || browserBusy || !!llama;
  $("load").disabled = sending || browserBusy || !!llama || !options.length;
  $("weights").disabled = sending || browserBusy || !!llama || !options.length;
  $("layers").disabled = sending || browserBusy || !!llama;
  $("unload").disabled = sending || browserBusy || !llama;
  for (const button of $("uploads").querySelectorAll("button")) button.disabled = sending;
}

async function browserAction(work) {
  if (sending || browserBusy) return;
  browserBusy = true;
  setControls();
  try {
    await work();
  } catch (error) {
    $("local-status").textContent = error.message || "The browser operation failed.";
    if (!llama) {
      setModelState("xor", navigator.gpu && typeof WebAssembly.Suspending === "function"
        ? "needs_load" : "unavailable");
    }
  } finally {
    browserBusy = false;
    setControls();
  }
}
async function hfRequest(url, token, init = {}) {
  const response = await fetch(url, {
    ...init,
    headers: { Authorization: "Bearer " + token },
    cache: "no-store",
    credentials: "omit",
  });
  if ([401, 403, 404].includes(response.status)) {
    throw new Error("Hugging Face denied access. Check your token and XORTRON access approval.");
  }
  if (!response.ok) throw new Error("Hugging Face request failed (" + response.status + ").");
  return response;
}

$("files").onclick = () => browserAction(async () => {
  if (!$("eligible").checked) throw new Error("Confirm eligibility under the XORTRON model terms.");
  const token = readToken();
  options = [];
  $("weights").replaceChildren(new Option("Loading files…"));
  const response = await hfRequest(HF + "/api/models/" + XOR + "?blobs=true", token);
  const info = await response.json();
  if (!/^[a-f0-9]{40}$/i.test(info.sha)) throw new Error("Hugging Face did not return a model revision.");
  options = (info.siblings ?? [])
    .filter(({ rfilename }) => typeof rfilename === "string" &&
      rfilename.toLowerCase().endsWith(".gguf") && !/mmproj/i.test(rfilename) &&
      (!/-\d{5}-of-\d{5}\.gguf$/i.test(rfilename) || /-00001-of-\d{5}\.gguf$/i.test(rfilename)))
    .map(({ rfilename, size }) => ({ path: rfilename, size, revision: info.sha }));
  options.sort((a, b) => (a.size ?? Infinity) - (b.size ?? Infinity) || a.path.localeCompare(b.path));
  $("weights").replaceChildren(...options.map(({ path, size }, i) =>
    new Option(size ? path + " · " + (size / 1e9).toFixed(1) + " GB" : path, String(i))));

  if (!options.length) {
    $("weights").append(new Option("No GGUF files found"));
    throw new Error("No text-model GGUF files were listed by the repository.");
  }
  const preferredIndex = options.findIndex(({ path }) => PREFERRED_GGUF.test(path));
  if (preferredIndex >= 0) $("weights").selectedIndex = preferredIndex;
  $("local-status").textContent = preferredIndex >= 0
    ? "Q3_K_M is selected for WebGPU. Choose a smaller file if your device has limited memory."
    : "Select a quantization and load it with WebGPU.";
});

$("load").onclick = () => browserAction(async () => {
  setModelState("xor", "loading");
  if (!$("eligible").checked) throw new Error("Confirm eligibility under the XORTRON model terms.");
  if (!navigator.gpu || !await navigator.gpu.requestAdapter()) {
    throw new Error("WebGPU is unavailable. Open the Space directly in a WebGPU-enabled browser.");
  }
  if (typeof WebAssembly.Suspending !== "function") {
    throw new Error("This model needs a current Chromium browser with WebAssembly JSPI support.");
  }
  const layers = Number($("layers").value);
  if (!Number.isInteger(layers) || layers < 0 || layers > 999) {
    throw new Error("Choose 0 for all layers or a positive whole number up to 999.");
  }
  const token = readToken();
  const { path, revision } = options[Number($("weights").value)];
  const url = HF + "/" + XOR + "/resolve/" + revision + "/" + path.split("/").map(encodeURIComponent).join("/");
  // Require a fresh gate check before a browser-cached model can be used.
  $("local-status").textContent = "Checking access to the selected GGUF…";
  await Promise.all([
    hfRequest(HF + "/api/whoami-v2", token),
    hfRequest(url, token, { method: "HEAD" }),
  ]);

  const { Wllama } = await import("/assets/vendor/index.js");
  let offloaded = 0;
  const recordOffload = (...parts) => {
    const match = parts.join(" ").match(/offloaded\s+(\d+)\/\d+\s+layers\s+to\s+GPU/i);
    if (match) offloaded = Number(match[1]);
  };
  const candidate = new Wllama({ default: "/assets/vendor/wasm/wllama.wasm" }, {
    logger: { debug: recordOffload, log: recordOffload, warn: recordOffload, error: recordOffload },
    parallelDownloads: 2,
  });
  if (!candidate.isSupportWebGPU()) throw new Error("This browser cannot run this WebGPU build of wllama.");
  $("local-status").textContent = "Downloading or opening cached GGUF weights…";
  try {
    await candidate.loadModelFromUrl(url, {
      headers: { Authorization: "Bearer " + token },
      n_ctx: 4096,
      n_parallel: 1,
      n_gpu_layers: layers || 999,
      progressCallback: ({ loaded, total }) => {
        const gb = (loaded / 1e9).toFixed(2);
        $("local-status").textContent = total > 0 && loaded >= total
          ? "Preparing the GGUF on your GPU…"
          : "Downloaded " + gb + " GB…";
      },
    });
    if (!offloaded) throw new Error("The runtime did not confirm GPU offload.");
    llama = candidate;
    setModelState("xor", "online");
    $("local-status").textContent = "Ready. WebGPU offloaded " + offloaded + " layers.";
  } catch (error) {
    await candidate.exit().catch(() => {});
    if (error.message === "The runtime did not confirm GPU offload.") throw error;
    throw new Error("The GGUF could not load. Check model access and browser storage, or try a smaller quantization.");
  }
});

$("unload").onclick = () => browserAction(async () => {
  const current = llama;
  llama = null;
  setModelState("xor", "needs_load");
  await current.exit();
  $("local-status").textContent = "Model unloaded. Downloaded weights may remain in this browser's site storage.";
});


function saveMemory() {
  try {
    localStorage.setItem(memoryKey, JSON.stringify({ transcript, uploads }));
  } catch {
    $("files-help").textContent = "Browser storage is unavailable. This chat and its files stay available until you reload. Messages and files are sent to the Space for OpenHarness.";
  }
}

try {
  const saved = JSON.parse(localStorage.getItem(memoryKey));
  if (Array.isArray(saved?.transcript)) transcript = saved.transcript.slice(-100);
  if (Array.isArray(saved?.uploads)) uploads = saved.uploads.slice(0, 6);
} catch {
  // Chat still works when browser storage is unavailable.
}

function addMessage({ role, text, model }) {
  const article = document.createElement("article");
  article.className = "message " + role;
  const title = document.createElement("small");
  title.textContent = role === "user" ? "You" : labels[model] ?? "Assistant";
  const body = document.createElement("p");
  body.textContent = text;
  article.append(title, body);
  $("transcript").append(article);
  $("welcome").hidden = true;
  return article;
}

function renderUploads() {
  $("uploads").replaceChildren();
  uploads.forEach(({ name }, index) => {
    const entry = document.createElement("li");
    entry.append(name);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary";
    remove.textContent = "×";
    remove.setAttribute("aria-label", "Remove " + name);
    remove.onclick = () => {
      uploads.splice(index, 1);
      saveMemory();
      renderUploads();
    };
    entry.append(remove);
    $("uploads").append(entry);
  });
  setControls();
}

for (const message of transcript) addMessage(message);
renderUploads();

$("new-chat").onclick = () => {
  transcript = [];
  uploads = [];
  saveMemory();
  $("transcript").replaceChildren();
  $("welcome").hidden = false;
  $("prompt").value = "";
  $("activity").textContent = "";
  renderUploads();
  showSettings(false);
};

$("attach").onchange = async () => {
  readingFiles = true;
  setControls();
  const picked = [...$("attach").files];
  $("attach").value = "";
  const errors = [];
  for (const file of picked) {
    if (uploads.length === 6) {
      errors.push("This chat already has six files.");
      break;
    }
    if (!file.size || file.size > 96000 || file.name.length > 120) {
      errors.push(file.name + " must be nonempty, at most 96 KB, and have a name up to 120 characters.");
      continue;
    }
    if (uploads.some(({ name }) => name === file.name)) {
      errors.push(file.name + " is already attached.");
      continue;
    }
    try {
      const text = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer());
      if (!text.trim() || /[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(text)) throw new Error();
      uploads.push({ name: file.name, text });
    } catch {
      errors.push(file.name + " must contain UTF-8 text.");
    }
  }
  readingFiles = false;
  saveMemory();
  renderUploads();
  $("activity").textContent = errors.join(" ");
};

function showAgentUpdate(update) {
  const tools = { search_uploads: "Searching your uploads…", read_upload: "Reading your upload…" };
  if (update.tool) $("activity").textContent = tools[update.tool] ?? "OpenHarness is using a tool…";
  if (update.status) $("activity").textContent = update.status;
}

async function chatWithBimo(payload, token) {
  const response = await fetch("/api/bimo", {
    method: "POST",
    headers: { Authorization: "Bearer " + token, "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(typeof error.detail === "string" ? error.detail : "Bimo request failed (" + response.status + ").");
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += value;
      let end;
      while ((end = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        if (!frame.startsWith("data: ")) continue;
        const update = JSON.parse(frame.slice(6));
        if (update.error) throw new Error(update.error);
        if (update.answer) return update.answer;
        showAgentUpdate(update);
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
  throw new Error("The Bimo connection ended before an answer arrived.");
}

function chatWithXortron(payload) {
  return new Promise((resolve, reject) => {
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(scheme + "//" + location.host + "/api/xortron");
    const abort = new AbortController();
    socket.onopen = () => socket.send(JSON.stringify(payload));
    socket.onerror = () => reject(new Error("Could not connect to OpenHarness."));
    socket.onclose = () => {
      abort.abort();
      reject(new Error("The XORTRON connection ended before an answer arrived."));
    };
    socket.onmessage = async (event) => {
      try {
        const update = JSON.parse(event.data);
        if (update.error) throw new Error(update.error);
        if (update.answer) {
          resolve(update.answer);
          socket.close();
          return;
        }
        showAgentUpdate(update);
        if (update.generate) {
          $("activity").textContent = "XORTRON is generating on this device…";
          const result = await llama.createChatCompletion({
            messages: update.generate,
            chat_template_kwargs: { enable_thinking: false },
            max_tokens: update.max_tokens,
            temperature: 0.2,
            abortSignal: abort.signal,
          });
          const message = result.choices[0]?.message;
          let answer = message?.content ?? "";
          const call = message?.tool_calls?.[0]?.function;
          if (!answer && call) {
            answer = "<agent_tool>" + JSON.stringify({
              name: call.name, arguments: JSON.parse(call.arguments),
            }) + "</agent_tool>";
          }
          if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ answer }));
          }
        }
      } catch (error) {
        reject(error);
        socket.close();
      }
    };
  });
}

$("composer").onsubmit = async (event) => {
  event.preventDefault();
  if (sending || browserBusy || readingFiles) return;
  const prompt = $("prompt").value.trim();
  if (!prompt) return;
  const model = $("model").value;
  let token;
  try {
    if (model === "bimo") token = readToken();
    else if (!llama || !$("eligible").checked) {
      throw new Error("Confirm eligibility and load XORTRON in Settings first.");
    }
  } catch (error) {
    $("activity").textContent = error.message;
    return;
  }

  const history = transcript.slice(-16).map(({ role, text }) => ({ role, text: text.slice(0, 5000) }));
  const payload = { prompt, history, uploads };
  const user = { role: "user", text: prompt };
  const bubble = addMessage(user);
  $("prompt").value = "";
  sending = true;
  setControls();
  setModelState(model, model === "bimo" ? "checking" : "busy");
  $("activity").textContent = "Opening an OpenHarness turn…";
  $("composer").scrollIntoView({ block: "end" });
  try {
    const text = model === "bimo" ? await chatWithBimo(payload, token) : await chatWithXortron(payload);
    const answer = { role: "assistant", text, model };
    transcript = [...transcript, user, answer].slice(-100);
    saveMemory();
    addMessage(answer);
    $("activity").textContent = "";
    $("composer").scrollIntoView({ block: "end" });
  } catch (error) {
    bubble.remove();
    $("welcome").hidden = transcript.length > 0;
    if (!$("prompt").value.trim()) $("prompt").value = prompt;
    $("activity").textContent = error.message || "Chat request failed.";
  } finally {
    sending = false;
    setControls();
    if (model === "bimo") refreshBimo();
    else setModelState("xor", llama ? "online" : "needs_load");
  }
};

$("prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});

setModelState("xor", navigator.gpu && typeof WebAssembly.Suspending === "function"
  ? "needs_load" : "unavailable");
refreshBimo();
setInterval(refreshBimo, 15000);
