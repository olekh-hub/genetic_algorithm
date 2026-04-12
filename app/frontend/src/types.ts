export type Variant = 'binary' | 'real';

export interface Methods {
  binary_crossover: string[];
  binary_mutation: string[];
  real_crossover: string[];
  real_mutation: string[];
}

export interface AlgorithmResult {
  best_fitness: number;
  best_variables: number[];
  best_history: number[];
  best_positions_history: number[][];
  population_history: { pos: number[][]; fit: number[] }[];
  mean_history: number[];
  std_history: number[];
  elapsed: number;
  total_epochs: number;
  stopped_early: boolean;
  saved_as: string;
}

export interface SurfaceData {
  x: number[][];
  y: number[][];
  z: number[][];
}
