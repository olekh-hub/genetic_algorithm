/// <reference types="vite/client" />

interface Plotly {
  react(el: HTMLElement, data: object[], layout: object, config?: Record<string, unknown>): Promise<void>;
  relayout(el: HTMLElement, update: Record<string, unknown>): void;
  restyle(el: HTMLElement, update: object, traceIndices?: number[]): void;
  purge(el: HTMLElement): void;
}

declare const Plotly: Plotly;
