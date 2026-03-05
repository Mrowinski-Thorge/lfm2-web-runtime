import './style.css';
import { LFM2Engine } from './engine.ts';
import { LFM2Tokenizer } from './tokenizer.ts';
import { detectDevice, formatDeviceSummary } from './device.ts';
import { isModelCached, clearModelCache } from './cache.ts';

// ── DOM references ──────────────────────────────────────────────────

const $gpuStatus = document.getElementById('gpu-status')!;
const $deviceInfo = document.getElementById('device-info')!;
const $cacheStatus = document.getElementById('cache-status')!;
const $recommendation = document.getElementById('recommendation')!;
const $btnLoad = document.getElementById('btn-load') as HTMLButtonElement;
const $btnClearCache = document.getElementById('btn-clear-cache') as HTMLButtonElement;
const $progressContainer = document.getElementById('progress-container')!;
const $progressFill = document.getElementById('progress-fill')!;
const $progressText = document.getElementById('progress-text')!;
const $chatContainer = document.getElementById('chat-container')!;
const $messages = document.getElementById('messages')!;
const $chatForm = document.getElementById('chat-form') as HTMLFormElement;
const $userInput = document.getElementById('user-input') as HTMLTextAreaElement;
const $btnSend = document.getElementById('btn-send') as HTMLButtonElement;

// ── State ───────────────────────────────────────────────────────────

const engine = new LFM2Engine();
const tokenizer = new LFM2Tokenizer();
let generating = false;

const MAX_NEW_TOKENS = 256;
const TEMPERATURE = 0.7;
const TOP_K = 40;

// ── Device detection + cache check (runs immediately) ───────────────

(async () => {
  // Detect device capabilities
  const device = await detectDevice();

  // GPU status
  if (device.gpu.supported) {
    $gpuStatus.textContent = `GPU: ${device.gpu.adapterName}`;
    $gpuStatus.classList.add('status-ok');
  } else {
    $gpuStatus.textContent = `GPU: ${device.gpu.adapterName}`;
    $gpuStatus.classList.add('status-error');
    $btnLoad.disabled = true;
    $btnLoad.textContent = 'WebGPU not supported';
  }

  // Device summary
  $deviceInfo.textContent = formatDeviceSummary(device);

  // Recommendation
  const tierColors: Record<string, string> = {
    unsupported: 'status-error',
    low: 'status-warn',
    medium: 'status-ok',
    high: 'status-ok',
  };
  $recommendation.textContent = device.recommendation;
  $recommendation.classList.add(tierColors[device.tier] || 'status-warn');

  // Check if model is cached
  const cached = await isModelCached('lfm2-350m-q4-v1');
  if (cached) {
    $cacheStatus.textContent = 'Model: cached (offline ready)';
    $cacheStatus.classList.add('status-ok');
    $btnClearCache.classList.remove('hidden');
  } else {
    $cacheStatus.textContent = 'Model: not cached (will download ~481 MB)';
    $cacheStatus.classList.add('status-warn');
  }
})();

// ── Clear cache button ──────────────────────────────────────────────

$btnClearCache.addEventListener('click', async () => {
  $btnClearCache.disabled = true;
  await clearModelCache();
  $cacheStatus.textContent = 'Model: cache cleared';
  $cacheStatus.className = 'status-warn';
  $btnClearCache.classList.add('hidden');
  $btnClearCache.disabled = false;
});

// ── Load Engine ─────────────────────────────────────────────────────

$btnLoad.addEventListener('click', async () => {
  $btnLoad.disabled = true;
  $btnLoad.textContent = 'Loading...';
  $progressContainer.classList.remove('hidden');

  const setProgress = (pct: number, msg: string) => {
    $progressFill.style.width = `${pct}%`;
    $progressText.textContent = msg;
  };

  try {
    // Load tokenizer (0-10% of progress bar)
    setProgress(0, 'Loading tokenizer...');
    await tokenizer.init((pct, msg) => {
      setProgress(Math.round(pct * 0.1), msg);
    });

    // Load engine with IndexedDB caching (10-100%)
    setProgress(10, 'Loading ONNX model...');
    await engine.init((pct, msg) => {
      setProgress(10 + Math.round(pct * 0.9), msg);
    });

    setProgress(100, 'Ready!');

    // Update cache status
    $cacheStatus.textContent = 'Model: cached (offline ready)';
    $cacheStatus.className = 'status-ok';
    $btnClearCache.classList.remove('hidden');

    // Show chat UI
    $progressContainer.classList.add('hidden');
    $btnLoad.classList.add('hidden');
    $btnClearCache.classList.add('hidden');
    document.getElementById('controls')!.classList.add('hidden');
    $chatContainer.classList.remove('hidden');
    $userInput.disabled = false;
    $btnSend.disabled = false;
    $userInput.focus();

    addMessage('system', 'Engine loaded. You can start chatting.');
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    setProgress(0, `Error: ${errorMsg}`);
    $progressFill.style.background = '#f85149';
    $btnLoad.textContent = 'Retry';
    $btnLoad.disabled = false;
    console.error('[Main] Load failed:', err);
  }
});

// ── Chat ────────────────────────────────────────────────────────────

$chatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = $userInput.value.trim();
  if (!text || generating) return;
  handleUserMessage(text);
});

// Allow Enter to send (Shift+Enter for newline)
$userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    $chatForm.dispatchEvent(new Event('submit'));
  }
});

async function handleUserMessage(text: string) {
  $userInput.value = '';
  addMessage('user', text);

  generating = true;
  $userInput.disabled = true;
  $btnSend.disabled = true;

  const botEl = addMessage('bot', '');
  const cursorSpan = document.createElement('span');
  cursorSpan.className = 'cursor';
  botEl.appendChild(cursorSpan);

  try {
    await generate(text, (token) => {
      const textNode = document.createTextNode(token);
      botEl.insertBefore(textNode, cursorSpan);
      $messages.scrollTop = $messages.scrollHeight;
    });
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    botEl.insertBefore(document.createTextNode(`[Error: ${errorMsg}]`), cursorSpan);
  }

  cursorSpan.remove();

  generating = false;
  $userInput.disabled = false;
  $btnSend.disabled = false;
  $userInput.focus();
}

// ── Text generation (top-k sampling with temperature) ───────────────

async function generate(
  prompt: string,
  onToken: (text: string) => void,
): Promise<void> {
  const inputIds = tokenizer.encode(prompt);
  const allIds = [...inputIds];

  for (let step = 0; step < MAX_NEW_TOKENS; step++) {
    const logits = await engine.forward(allIds);

    // Extract logits for the last token position
    // logits shape: [1, seqLen, vocabSize] flattened
    const vocabSize = logits.length / allIds.length;
    const lastTokenLogits = logits.slice((allIds.length - 1) * vocabSize);

    const nextId = sampleTopK(lastTokenLogits, TEMPERATURE, TOP_K);

    if (tokenizer.eosTokenId !== 0 && nextId === tokenizer.eosTokenId) {
      break;
    }

    allIds.push(BigInt(nextId));

    const tokenText = tokenizer.decodeToken(nextId);
    onToken(tokenText);

    // Breathe every few tokens to keep the UI responsive
    if (step % 4 === 0) {
      await new Promise((r) => setTimeout(r, 0));
    }
  }
}

// ── Sampling ────────────────────────────────────────────────────────

function sampleTopK(
  logits: Float32Array,
  temperature: number,
  k: number,
): number {
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    scaled[i] = logits[i] / temperature;
  }

  const indices = Array.from({ length: scaled.length }, (_, i) => i);
  indices.sort((a, b) => scaled[b] - scaled[a]);
  const topK = indices.slice(0, k);

  let maxVal = -Infinity;
  for (const idx of topK) {
    if (scaled[idx] > maxVal) maxVal = scaled[idx];
  }

  const exps: number[] = [];
  let sumExp = 0;
  for (const idx of topK) {
    const e = Math.exp(scaled[idx] - maxVal);
    exps.push(e);
    sumExp += e;
  }

  let r = Math.random() * sumExp;
  for (let i = 0; i < topK.length; i++) {
    r -= exps[i];
    if (r <= 0) return topK[i];
  }

  return topK[0];
}

// ── UI helpers ──────────────────────────────────────────────────────

function addMessage(role: 'user' | 'bot' | 'system', text: string): HTMLDivElement {
  const el = document.createElement('div');
  el.className = `message ${role}`;
  if (text) el.textContent = text;
  $messages.appendChild(el);
  $messages.scrollTop = $messages.scrollHeight;
  return el;
}
