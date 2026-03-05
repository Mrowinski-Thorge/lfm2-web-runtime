import * as ort from 'onnxruntime-web/webgpu';

export type ProgressCallback = (pct: number, msg: string) => void;

/** Async breath - yields to the event loop to prevent GPU driver timeouts */
function breathe(ms = 100): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * LFM2Engine - WebGPU inference engine for Liquid AI LFM2-350M (INT4)
 *
 * Anti-crash strategy:
 *  - powerPreference: 'low-power' on adapter request
 *  - graphOptimizationLevel: 'basic' to prevent shader-compile timeouts
 *  - breathe() calls during heavy async work
 */
export class LFM2Engine {
  private session: ort.InferenceSession | null = null;

  private readonly baseUrl =
    'https://huggingface.co/Thorge-AI/lfm2-web-runtime-weights/resolve/main/';

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
        powerPreference: 'low-power',
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
    await breathe(150); // initial safety delay

    // Configure env-level WASM settings
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    report(10, 'Downloading model (INT4, ~481 MB)...');
    await breathe();

    this.session = await ort.InferenceSession.create(
      this.baseUrl + 'model_q4.onnx',
      {
        executionProviders: [{ name: 'webgpu' as const }],
        graphOptimizationLevel: 'basic',
        preferredOutputLocation: 'gpu-buffer',
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
      // Ensure at least [1,1]
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
