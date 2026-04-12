import type { AlgorithmResult, Methods, SurfaceData } from './types';

const BASE = '/api';

export async function fetchMethods(): Promise<Methods> {
  const res = await fetch(`${BASE}/methods`);
  return res.json();
}

export async function fetchSurface(
  x1_min: number, x1_max: number,
  x2_min: number, x2_max: number,
): Promise<SurfaceData> {
  const res = await fetch(`${BASE}/surface`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x1_min, x1_max, x2_min, x2_max }),
  });
  return res.json();
}

export async function runAlgorithm(params: Record<string, unknown>): Promise<AlgorithmResult & { error?: string }> {
  const res = await fetch(`${BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export function downloadCsv() {
  window.location.href = `${BASE}/download`;
}
