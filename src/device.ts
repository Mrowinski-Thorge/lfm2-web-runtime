/**
 * Device detection and model recommendation.
 * Checks GPU capabilities, memory, CPU cores, and mobile status
 * to recommend the best model variant for the user's hardware.
 */

export interface DeviceInfo {
  gpu: {
    supported: boolean;
    adapterName: string;
    vendor: string;
    maxBufferSize: number;
  };
  memoryGB: number | null;
  cores: number;
  isMobile: boolean;
  tier: 'unsupported' | 'low' | 'medium' | 'high';
  recommendation: string;
}

export async function detectDevice(): Promise<DeviceInfo> {
  const memoryGB =
    'deviceMemory' in navigator
      ? (navigator as unknown as Record<string, number>).deviceMemory
      : null;

  const info: DeviceInfo = {
    gpu: {
      supported: false,
      adapterName: 'Not available',
      vendor: 'Unknown',
      maxBufferSize: 0,
    },
    memoryGB: memoryGB ?? null,
    cores: navigator.hardwareConcurrency || 1,
    isMobile: /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent),
    tier: 'unsupported',
    recommendation: '',
  };

  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
      if (adapter) {
        info.gpu.supported = true;
        const ai = adapter.info as unknown as Record<string, string>;
        info.gpu.adapterName =
          ai?.['device'] || ai?.['description'] || ai?.['vendor'] || 'Unknown GPU';
        info.gpu.vendor = ai?.['vendor'] || 'Unknown';
        info.gpu.maxBufferSize = adapter.limits?.maxBufferSize ?? 0;
      }
    } catch {
      // GPU detection failed
    }
  }

  // Determine tier
  if (!info.gpu.supported) {
    info.tier = 'unsupported';
    info.recommendation =
      'WebGPU is not supported on this browser/device. Please use Chrome 113+ or Edge 113+.';
  } else if (info.isMobile && (!info.memoryGB || info.memoryGB < 6)) {
    info.tier = 'low';
    info.recommendation =
      'Mobile device with limited resources. LFM2-350M INT4 may work but expect slow performance.';
  } else if (
    (info.memoryGB && info.memoryGB >= 8) ||
    info.gpu.maxBufferSize > 2_000_000_000 ||
    (!info.isMobile && info.cores >= 8)
  ) {
    info.tier = 'high';
    info.recommendation =
      'LFM2-350M INT4 recommended. Your device has strong hardware for in-browser inference.';
  } else {
    info.tier = 'medium';
    info.recommendation =
      'LFM2-350M INT4 should work. Performance depends on your GPU.';
  }

  return info;
}

/** Format device info into a human-readable string for the status bar. */
export function formatDeviceSummary(device: DeviceInfo): string {
  const parts: string[] = [];

  if (device.gpu.supported) {
    parts.push(`GPU: ${device.gpu.adapterName}`);
  } else {
    parts.push('GPU: Not available');
  }

  if (device.memoryGB) {
    parts.push(`RAM: ${device.memoryGB}GB`);
  }

  parts.push(`Cores: ${device.cores}`);

  if (device.isMobile) {
    parts.push('Mobile');
  }

  return parts.join(' | ');
}
