import { AutoTokenizer } from '@xenova/transformers';

// Load tokenizer files from the Models/350M subfolder via full URL
const HF_TOKENIZER_BASE =
  'https://huggingface.co/Thorge-AI/lfm2-web-runtime-weights/resolve/main/Models/350M';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type HFTokenizer = any;

export type TokenizerProgressCallback = (pct: number, msg: string) => void;

export class LFM2Tokenizer {
  private tok: HFTokenizer = null;
  public eosTokenId = 0;

  get isReady(): boolean {
    return this.tok !== null;
  }

  async init(onProgress?: TokenizerProgressCallback): Promise<void> {
    onProgress?.(0, 'Loading tokenizer...');

    // Pass full URL so @xenova/transformers fetches from the correct subfolder
    this.tok = await AutoTokenizer.from_pretrained(HF_TOKENIZER_BASE, {
      progress_callback: (data: Record<string, unknown>) => {
        if (data['status'] === 'progress' && typeof data['progress'] === 'number') {
          onProgress?.(Math.round(data['progress'] as number), 'Downloading tokenizer...');
        }
      },
    });

    // Resolve the EOS token id (<|endoftext|>)
    const candidate: unknown = this.tok.eos_token_id;
    this.eosTokenId = typeof candidate === 'number' ? candidate : 0;

    onProgress?.(100, 'Tokenizer ready!');
    console.log(`[Tokenizer] Loaded. EOS token id = ${this.eosTokenId}`);
  }

  /** Encode text into token ids (as bigint[] for ONNX int64 tensors) */
  encode(text: string): bigint[] {
    if (!this.tok) throw new Error('Tokenizer not loaded');
    const out: { input_ids: number[] | number[][] } = this.tok(text, {
      return_tensor: false,
    }) as { input_ids: number[] | number[][] };

    const ids = Array.isArray(out.input_ids[0])
      ? (out.input_ids as number[][])[0]
      : (out.input_ids as number[]);

    return ids.map((id) => BigInt(id));
  }

  /** Decode token ids back to text */
  decode(ids: number[]): string {
    if (!this.tok) throw new Error('Tokenizer not loaded');
    return this.tok.decode(ids, { skip_special_tokens: true }) as string;
  }

  /** Decode a single token (for streaming output) */
  decodeToken(id: number): string {
    if (!this.tok) throw new Error('Tokenizer not loaded');
    return this.tok.decode([id], { skip_special_tokens: false }) as string;
  }
}
