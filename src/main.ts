import './style.css';
import { LFM2Engine } from './engine.ts';
import { LFM2Tokenizer } from './tokenizer.ts';

// ── DOM references ──────────────────────────────────────────────────

const $gpuStatus = document.getElementById('gpu-status')!;
const $btnLoad = document.getElementById('btn-load') as HTMLButtonElement;
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

// ── GPU check (runs immediately) ────────────────────────────────────

(async () => {
  const gpu = await LFM2Engine.checkGPU();
  if (gpu.supported) {
    $gpuStatus.textContent = `GPU: ${gpu.adapterName}`;
    $gpuStatus.style.color = '#3fb950';
  } else {
    $gpuStatus.textContent = `GPU: ${gpu.adapterName}`;
    $gpuStatus.style.color = '#f85149';
    $btnLoad.disabled = true;
    $btnLoad.textContent = 'WebGPU not supported';
  }
})();

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
    // Load tokenizer and engine in sequence (tokenizer first, it's small)
    setProgress(0, 'Loading tokenizer...');
    await tokenizer.init((pct, msg) => {
      // Tokenizer gets 0-15% of the bar
      setProgress(Math.round(pct * 0.15), msg);
    });

    setProgress(15, 'Loading ONNX model...');
    await engine.init((pct, msg) => {
      // Engine gets 15-100% of the bar
      setProgress(15 + Math.round(pct * 0.85), msg);
    });

    setProgress(100, 'Ready!');

    // Show chat UI
    $progressContainer.classList.add('hidden');
    $btnLoad.classList.add('hidden');
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
      // Insert text before the cursor element
      const textNode = document.createTextNode(token);
      botEl.insertBefore(textNode, cursorSpan);
      $messages.scrollTop = $messages.scrollHeight;
    });
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    botEl.insertBefore(document.createTextNode(`[Error: ${errorMsg}]`), cursorSpan);
  }

  // Remove blinking cursor
  cursorSpan.remove();

  generating = false;
  $userInput.disabled = false;
  $btnSend.disabled = false;
  $userInput.focus();
}

// ── Text generation (greedy / top-k sampling with temperature) ──────

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

    // Sample next token
    const nextId = sampleTopK(lastTokenLogits, TEMPERATURE, TOP_K);

    // Check for EOS
    if (tokenizer.eosTokenId !== 0 && nextId === tokenizer.eosTokenId) {
      break;
    }

    allIds.push(BigInt(nextId));

    // Decode and stream
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
  // Apply temperature
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    scaled[i] = logits[i] / temperature;
  }

  // Find top-k indices
  const indices = Array.from({ length: scaled.length }, (_, i) => i);
  indices.sort((a, b) => scaled[b] - scaled[a]);
  const topK = indices.slice(0, k);

  // Softmax over top-k
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

  // Weighted random selection
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
