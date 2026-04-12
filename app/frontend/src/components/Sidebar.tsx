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
    <div className="flex items-center justify-between py-1.5">
      <span className="text-[14px] text-[#999]">{label}</span>
      {children}
    </div>
  );
}

function Num(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input type="number" {...props}
    className="w-[120px] h-10 px-3 bg-[#1c1c20] border border-[#2a2a2e] rounded-lg text-[14px] font-mono text-[#ddd] outline-none focus:border-[var(--_accent)] focus:ring-1 focus:ring-[var(--_accent-soft)] transition-all" />;
}

function Small(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input type="number" {...props}
    className="w-[84px] h-9 px-2.5 bg-[#1c1c20] border border-[#2a2a2e] rounded-lg text-[13px] font-mono text-[#ddd] outline-none focus:border-[var(--_accent)] focus:ring-1 focus:ring-[var(--_accent-soft)] transition-all" />;
}

function Sel({ name, options }: { name: string; options: string[] }) {
  return (
    <select name={name}
      className="w-[120px] h-10 px-3 bg-[#1c1c20] border border-[#2a2a2e] rounded-lg text-[14px] text-[#ddd] outline-none focus:border-[var(--_accent)] focus:ring-1 focus:ring-[var(--_accent-soft)] transition-all appearance-none cursor-pointer"
      style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' fill='%23666' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 10px center', paddingRight: '26px' }}>
      {options.map(m => <option key={m} value={m}>{m.replace(/_/g, ' ')}</option>)}
    </select>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="py-3.5">
      <div className="text-[12px] font-semibold text-[var(--_accent)] mb-3 tracking-wide">{title}</div>
      {children}
    </div>
  );
}

function RangeRow({ label, range, onChange }: { label: string; range: Range; onChange: (min: number, max: number) => void }) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-[14px] text-[#999]">{label}</span>
      <div className="flex items-center gap-2">
        <Small value={range.min} step={0.1} onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v)) onChange(v, range.max); }} />
        <span className="text-[12px] text-[#444]">to</span>
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
    <form ref={formRef} className="bg-[#111113] border-r border-[#222] flex flex-col h-screen min-h-0" onSubmit={handleSubmit}>
      <div className="flex-1 min-h-0 overflow-y-auto px-5 pt-5 pb-3 divide-y divide-[#222]">

        {/* Header */}
        <div className="pb-4">
          <div className="text-[18px] font-semibold text-[#eee]">GA Workbench</div>
          <div className="text-[13px] text-[#555] mt-1">Rosenbrock function optimization</div>

          {/* Variant toggle */}
          <div className="flex gap-1 mt-4 p-1 bg-[#1c1c20] rounded-xl">
            {(['binary', 'real'] as Variant[]).map(v => (
              <button key={v} type="button" onClick={() => onVariantChange(v)}
                className={cn(
                  'flex-1 py-2.5 text-[14px] font-medium rounded-lg transition-all capitalize cursor-pointer',
                  variant === v
                    ? 'text-[var(--_accent-fg)] shadow-md'
                    : 'text-[#666] hover:text-[#aaa] hover:bg-[#222]'
                )}
                style={variant === v ? { background: 'var(--_accent)' } : undefined}>
                {v}
              </button>
            ))}
          </div>
        </div>

        {/* View range */}
        <Group title="View range">
          <RangeRow label="x₁" range={x1Range} onChange={(min, max) => onViewRangeChange('x1', min, max)} />
          <RangeRow label="x₂" range={x2Range} onChange={(min, max) => onViewRangeChange('x2', min, max)} />
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
          <div className="pt-3 mt-2.5 border-t border-[#222]">
            <div className="text-[12px] text-[#555] mb-2">Start point (optional)</div>
            <div className="flex gap-2">
              <Small name="start_x1" placeholder="x₁" step={0.1} />
              <Small name="start_x2" placeholder="x₂" step={0.1} />
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
          <label className="flex items-center gap-2.5 py-2 cursor-pointer">
            <input type="checkbox" id="maximize" className="w-4 h-4 accent-[var(--_accent)] cursor-pointer" />
            <span className="text-[14px] text-[#999]">Maximize</span>
          </label>
        </Group>

        {/* Early stopping */}
        <Group title="Early stopping">
          <label className="flex items-center gap-2.5 cursor-pointer">
            <input type="checkbox" id="early_stop" className="w-4 h-4 accent-[var(--_accent)] cursor-pointer" />
            <span className="text-[14px] text-[#999]">Enable</span>
          </label>
          <Row label="Patience"><Num name="patience" defaultValue={20} min={1} /></Row>
          <Row label="Min delta"><Num name="min_delta" defaultValue={0.000001} step="any" /></Row>
        </Group>
      </div>

      {/* Bottom actions */}
      <div className="shrink-0 px-5 py-4 border-t border-[#222] space-y-2.5">
        {isRunning && (
          <div className="w-full h-1 bg-[#222] rounded-full overflow-hidden">
            <div className="h-full rounded-full transition-[width] duration-300" style={{ width: `${progress}%`, background: 'var(--_accent)' }} />
          </div>
        )}
        <button type="submit" disabled={isRunning}
          className="w-full h-11 rounded-xl text-[14px] font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          style={{ background: 'var(--_accent)', color: 'var(--_accent-fg)', boxShadow: isRunning ? 'none' : '0 2px 16px var(--_accent-glow)' }}>
          {isRunning ? 'Running...' : 'Run'}
        </button>
        <button type="button" onClick={onDownload} disabled={!hasResults}
          className="w-full h-10 rounded-xl border border-[#2a2a2e] text-[#777] text-[13px] hover:text-[#aaa] hover:border-[#444] transition-colors disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer">
          Download CSV
        </button>
      </div>
    </form>
  );
}
