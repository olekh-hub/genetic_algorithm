import { useRef } from 'react';
import { cn } from '@/lib/utils';
import type { Variant, Methods } from '../types';
import type { Range } from '../App';

interface Props {
  variant: Variant;
  methods: Methods | null;
  isRunning: boolean;
  hasResults: boolean;
  bounds: { a: number; b: number };
  x1Range: Range;
  x2Range: Range;
  onVariantChange: (v: Variant) => void;
  onRun: (params: Record<string, unknown>) => void;
  onDownload: () => void;
  onBoundsChange: (a: number, b: number) => void;
  onViewRangeChange: (axis: 'x1' | 'x2', min: number, max: number) => void;
  progress: number;
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-[13px] text-[#888]">{label}</span>
      {children}
    </div>
  );
}

function Num(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input type="number" {...props}
    className="w-[116px] h-9 px-2.5 bg-[#1e1e22] border border-[#2a2a2e] rounded text-[13px] font-mono text-[#ccc] outline-none focus:border-[#555] transition-colors" />;
}

function Small(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input type="number" {...props}
    className="w-[80px] h-8 px-2 bg-[#1e1e22] border border-[#2a2a2e] rounded text-[13px] font-mono text-[#ccc] outline-none focus:border-[#555] transition-colors" />;
}

function Sel({ name, options }: { name: string; options: string[] }) {
  return (
    <select name={name}
      className="w-[116px] h-9 px-2.5 bg-[#1e1e22] border border-[#2a2a2e] rounded text-[13px] text-[#ccc] outline-none focus:border-[#555] transition-colors appearance-none cursor-pointer"
      style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='8' fill='%23666' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 8px center', paddingRight: '22px' }}>
      {options.map(m => <option key={m} value={m}>{m.replace(/_/g, ' ')}</option>)}
    </select>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="py-3">
      <div className="text-[12px] font-medium text-[#666] mb-2.5">{title}</div>
      {children}
    </div>
  );
}

function RangeRow({ label, range, onChange }: { label: string; range: Range; onChange: (min: number, max: number) => void }) {
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-[13px] text-[#888]">{label}</span>
      <div className="flex items-center gap-1.5">
        <Small value={range.min} step={0.1} onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) onChange(v, range.max); }} />
        <span className="text-[11px] text-[#555]">..</span>
        <Small value={range.max} step={0.1} onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) onChange(range.min, v); }} />
      </div>
    </div>
  );
}

export default function Sidebar({
  variant, methods, isRunning, hasResults,
  bounds, x1Range, x2Range,
  onVariantChange, onRun, onDownload, onBoundsChange, onViewRangeChange, progress,
}: Props) {
  const crossoverMethods = variant === 'binary' ? methods?.binary_crossover : methods?.real_crossover;
  const mutationMethods = variant === 'binary' ? methods?.binary_mutation : methods?.real_mutation;
  const formRef = useRef<HTMLFormElement>(null);

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const fd = new FormData(e.currentTarget);
    const p: Record<string, unknown> = {};
    fd.forEach((val, key) => { p[key] = val; });
    p.variant = variant;
    p.maximize = (e.currentTarget.querySelector('#maximize') as HTMLInputElement).checked;
    p.early_stop = (e.currentTarget.querySelector('#early_stop') as HTMLInputElement).checked;
    const x1 = parseFloat(p.start_x1 as string), x2 = parseFloat(p.start_x2 as string);
    delete p.start_x1; delete p.start_x2;
    if (!isNaN(x1) && !isNaN(x2)) p.start_point = [x1, x2];
    onRun(p);
  }

  return (
    <form ref={formRef} className="bg-[#151518] border-r border-[#222] flex flex-col h-screen min-h-0" onSubmit={handleSubmit}>
      <div className="flex-1 min-h-0 overflow-y-auto px-4 pt-4 pb-2 divide-y divide-[#222]">

        {/* Header */}
        <div className="pb-3">
          <div className="text-[16px] font-semibold text-[#e0e0e0]">GA Workbench</div>
          <div className="text-[12px] text-[#555] mt-0.5">Rosenbrock function optimization</div>

          {/* Variant */}
          <div className="flex gap-0 mt-3 border border-[#2a2a2e] rounded overflow-hidden">
            {(['binary', 'real'] as Variant[]).map(v => (
              <button key={v} type="button" onClick={() => onVariantChange(v)}
                className={cn(
                  'flex-1 py-2 text-[13px] font-medium transition-colors capitalize cursor-pointer',
                  variant === v ? 'bg-[#252528] text-[#e0e0e0]' : 'text-[#666] hover:text-[#999] hover:bg-[#1c1c1f]'
                )}>
                {v}
              </button>
            ))}
          </div>
        </div>

        {/* View range */}
        <Group title="View range">
          <RangeRow label="x1" range={x1Range} onChange={(min, max) => onViewRangeChange('x1', min, max)} />
          <RangeRow label="x2" range={x2Range} onChange={(min, max) => onViewRangeChange('x2', min, max)} />
        </Group>

        {/* Search bounds */}
        <Group title="Search bounds">
          <Row label="a (lower)">
            <Num name="a" defaultValue={bounds.a} step={0.1}
              onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) onBoundsChange(v, bounds.b); }} />
          </Row>
          <Row label="b (upper)">
            <Num name="b" defaultValue={bounds.b} step={0.1}
              onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) onBoundsChange(bounds.a, v); }} />
          </Row>
          <Row label="Dimensions"><Num name="n_dims" defaultValue={2} min={2} /></Row>
          {variant === 'binary' && <Row label="Bits"><Num name="n_bits" defaultValue={50} min={4} /></Row>}
          <div className="pt-2.5 mt-2 border-t border-[#222]">
            <div className="text-[12px] text-[#555] mb-2">Start point (optional)</div>
            <div className="flex gap-2">
              <Small name="start_x1" placeholder="x1" step={0.1} />
              <Small name="start_x2" placeholder="x2" step={0.1} />
            </div>
          </div>
        </Group>

        {/* Population */}
        <Group title="Population">
          <Row label="Size"><Num name="pop_size" defaultValue={100} min={4} /></Row>
          <Row label="Epochs"><Num name="epochs" defaultValue={200} min={1} /></Row>
          <Row label="Selected"><Num name="n_best" defaultValue={20} min={2} /></Row>
          <Row label="Elite"><Num name="elite_size" defaultValue={2} min={0} /></Row>
          <Row label="Tournament k"><Num name="k" defaultValue={3} min={2} /></Row>
        </Group>

        {/* Operators */}
        <Group title="Operators">
          <Row label="Selection"><Sel name="selection" options={['best', 'roulette', 'tournament']} /></Row>
          <Row label="Crossover"><Sel name="crossover" key={`c-${variant}`} options={crossoverMethods ?? []} /></Row>
          <Row label="Mutation"><Sel name="mutation" key={`m-${variant}`} options={mutationMethods ?? []} /></Row>
          <Row label="P(cross)"><Num name="crossover_prob" defaultValue={0.9} min={0} max={1} step={0.01} /></Row>
          <Row label="P(mutate)"><Num name="mutation_prob" defaultValue={0.1} min={0} max={1} step={0.01} /></Row>
          {variant === 'binary' && <Row label="P(inversion)"><Num name="inversion_prob" defaultValue={0.1} min={0} max={1} step={0.01} /></Row>}
          {variant === 'real' && <Row label="Sigma"><Num name="sigma" defaultValue={0.1} min={0} step={0.01} /></Row>}
          <label className="flex items-center gap-2 py-1 cursor-pointer">
            <input type="checkbox" id="maximize" className="w-4 h-4 accent-[#888] cursor-pointer" />
            <span className="text-[13px] text-[#888]">Maximize</span>
          </label>
        </Group>

        {/* Early stopping */}
        <Group title="Early stopping">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" id="early_stop" className="w-4 h-4 accent-[#888] cursor-pointer" />
            <span className="text-[13px] text-[#888]">Enable</span>
          </label>
          <Row label="Patience"><Num name="patience" defaultValue={20} min={1} /></Row>
          <Row label="Min delta"><Num name="min_delta" defaultValue={0.000001} step="any" /></Row>
        </Group>
      </div>

      {/* Bottom actions */}
      <div className="shrink-0 px-4 py-3 border-t border-[#222] space-y-2">
        {isRunning && (
          <div className="w-full h-[3px] bg-[#222] rounded-full overflow-hidden">
            <div className="h-full rounded-full transition-[width] duration-300" style={{ width: `${progress}%`, background: 'var(--_accent)' }} />
          </div>
        )}
        <button type="submit" disabled={isRunning}
          className="w-full h-10 rounded bg-[#e0e0e0] text-[#151518] text-[13px] font-semibold hover:bg-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer">
          {isRunning ? 'Running...' : 'Run'}
        </button>
        <button type="button" onClick={onDownload} disabled={!hasResults}
          className="w-full h-9 rounded border border-[#2a2a2e] text-[#666] text-[13px] hover:text-[#999] hover:border-[#444] transition-colors disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer">
          Download CSV
        </button>
      </div>
    </form>
  );
}
