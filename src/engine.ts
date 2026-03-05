import * as ort from 'onnxruntime-web/webgpu';
import {
  getCachedModel,
  cacheModel,
  fetchWithProgress,
  type CachedModel,
} from './cache.ts';

export type ProgressCallback = (pct: number, msg: string) => void;

const MODEL_CACHE_KEY = 'lfm2-350m-q4-v1';

const HF_BASE =
  'https://huggingface.co/Thorge-AI/lfm2-web-runtime-weights/resolve/main/Models/350M/';

const GRAPH_FILE = 'model_q4.onnx';
const WEIGHTS_FILE = 'model_q4.onnx_data';

/** Async breath - yields to the event loop to prevent GPU driver timeouts */
function breathe(ms = 100): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function formatMB(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1);
}

/**
 * LFM2Engine - WebGPU inference engine for Liquid AI LFM2-350M (INT4)
 *
 * Features:
 *  - IndexedDB caching for offline usage
 *  - Single-threaded WASM (no COOP/COEP headers needed = works on GitHub Pages)
 *  - Anti-crash: basic graph optimization, breathe() delays
 */
export class LFM2Engine {
  private session: ort.InferenceSession | null = null;

  get isReady(): boolean {
    return this.session !== null;
  }

  get inputNames(): readonly string[] {
    return this.session?.inputNames ?? [];
  }

  get outputNames(): readonly string[] {
    return this.session?.outputNames ?? [];
  }

  // ── GPU capability check ──────────────────────────────────────────

  static async checkGPU(): Promise<{ supported: boolean; adapterName: string }> {
    if (!navigator.gpu) {
      return { supported: false, adapterName: 'WebGPU not available' };
    }
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
      if (!adapter) {
        return { supported: false, adapterName: 'No adapter found' };
      }
      const info = adapter.info as unknown as Record<string, string>;
      const name =
        info?.['device'] || info?.['description'] || info?.['vendor'] || 'Unknown GPU';
      return { supported: true, adapterName: name };
    } catch {
      return { supported: false, adapterName: 'Adapter request failed' };
    }
  }

  // ── Initialisation ────────────────────────────────────────────────

  async init(onProgress?: ProgressCallback): Promise<void> {
    const report = (pct: number, msg: string) => {
      onProgress?.(pct, msg);
    };

    report(0, 'Preparing WebGPU backend...');
    await breathe(150);

    // Single-threaded WASM - no SharedArrayBuffer needed (GitHub Pages compatible)
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = false;

    // Try loading from IndexedDB cache first
    report(5, 'Checking local cache...');
    const cached = await getCachedModel(MODEL_CACHE_KEY);

    let graphData: ArrayBuffer;
    let weightsData: ArrayBuffer;

    if (cached) {
      report(10, 'Loading model from cache...');
      graphData = cached.graph;
      weightsData = cached.weights;
      report(60, 'Model loaded from cache.');
    } else {
      // Download model graph (small, ~180KB)
      report(10, `Downloading model graph (${GRAPH_FILE})...`);
      graphData = await fetchWithProgress(
        HF_BASE + GRAPH_FILE,
        (loaded, total) => {
          if (total > 0) {
            const pct = 10 + Math.round((loaded / total) * 5);
            report(pct, `Downloading graph... ${formatMB(loaded)} / ${formatMB(total)} MB`);
          }
        },
      );
      await breathe();

      // Download model weights (large, ~481MB)
      report(15, `Downloading weights (~481 MB)...`);
      weightsData = await fetchWithProgress(
        HF_BASE + WEIGHTS_FILE,
        (loaded, total) => {
          if (total > 0) {
            const pct = 15 + Math.round((loaded / total) * 45);
            report(pct, `Downloading weights... ${formatMB(loaded)} / ${formatMB(total)} MB`);
          } else {
            report(30, `Downloading weights... ${formatMB(loaded)} MB`);
          }
        },
      );
      await breathe();

      // Cache in IndexedDB for offline usage
      report(60, 'Caching model for offline use...');
      try {
        await cacheModel({
          key: MODEL_CACHE_KEY,
          graph: graphData,
          weights: weightsData,
          timestamp: Date.now(),
          version: '350m-q4-v1',
        } satisfies CachedModel);
        report(65, 'Model cached successfully.');
      } catch (e) {
        console.warn('[Engine] Failed to cache model:', e);
        report(65, 'Cache failed (will re-download next time).');
      }
    }

    // Create ONNX session from buffered data
    report(70, 'Creating inference session...');
    await breathe();

    this.session = await ort.InferenceSession.create(
      new Uint8Array(graphData),
      {
        executionProviders: [{ name: 'webgpu' as const }],
        graphOptimizationLevel: 'basic',
        preferredOutputLocation: 'gpu-buffer',
        externalData: [
          {
            path: WEIGHTS_FILE,
            data: new Uint8Array(weightsData),
          },
        ],
      },
    );

    report(85, 'Session created, warming up...');
    await breathe();

    await this.warmup();
    report(100, 'Engine ready!');
  }

  // ── Warm-up ───────────────────────────────────────────────────────

  private async warmup(): Promise<void> {
    if (!this.session) return;

    const feeds: Record<string, ort.Tensor> = {};

    for (let i = 0; i < this.session.inputNames.length; i++) {
      const name = this.session.inputNames[i];
      const meta = this.session.inputMetadata?.[i] as
        | { dims?: number[]; type?: string }
        | undefined;

      let dims = meta?.dims ? meta.dims.map((d) => (d < 0 ? 1 : d)) : [1, 1];
      if (dims.length === 0) dims = [1, 1];

      const type = meta?.type ?? 'int64';
      const size = dims.reduce((a, b) => a * b, 1);

      if (type === 'int64') {
        feeds[name] = new ort.Tensor('int64', new BigInt64Array(size), dims);
      } else {
        feeds[name] = new ort.Tensor('float32', new Float32Array(size), dims);
      }
    }

    try {
      const results = await this.session.run(feeds);
      for (const key of Object.keys(results)) {
        results[key].dispose();
      }
    } catch (e) {
      console.warn('[LFM2] Warm-up pass skipped:', e);
    }
  }

  // ── Inference (single forward pass) ───────────────────────────────

  async forward(inputIds: bigint[]): Promise<Float32Array> {
    if (!this.session) throw new Error('Engine not initialised');

    const seqLen = inputIds.length;

    const feeds: Record<string, ort.Tensor> = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(inputIds), [1, seqLen]),
      attention_mask: new ort.Tensor(
        'int64',
        BigInt64Array.from(inputIds.map(() => BigInt(1))),
        [1, seqLen],
      ),
    };

    const results = await this.session.run(feeds);

    const outputName = this.session.outputNames[0];
    const output = results[outputName];
    const logits = (await output.getData()) as Float32Array;

    for (const key of Object.keys(results)) {
      results[key].dispose();
    }

    return logits;
  }

  // ── Cleanup ───────────────────────────────────────────────────────

  async release(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
  }
}
