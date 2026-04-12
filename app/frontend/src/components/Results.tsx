import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts';
import type { AlgorithmResult } from '../types';

interface Props {
  result: AlgorithmResult;
  maximize: boolean;
}

const axis = {
  tick: { fill: '#666', fontSize: 11, fontFamily: 'IBM Plex Mono' },
  axisLine: { stroke: '#2a2a2e' },
  tickLine: { stroke: '#2a2a2e' },
};

const tip = {
  contentStyle: { background: '#1c1c20', border: '1px solid #2a2a2e', borderRadius: 10, fontFamily: 'IBM Plex Mono', fontSize: 12 },
  labelStyle: { color: '#888' },
};

export default function Results({ result, maximize }: Props) {
  const chartData = result.best_history.map((best, i) => ({
    epoch: i, best, mean: result.mean_history[i], std: result.std_history[i],
  }));

  return (
    <div className="space-y-5">
      <div className="text-[16px] font-semibold text-[#aaa]">Results</div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-x-10 gap-y-2.5 text-[14px] bg-[#1a1a1d] rounded-xl border border-[#252528] px-6 py-5">
        <div className="flex justify-between">
          <span className="text-[#777]">Best fitness</span>
          <span className="font-mono font-medium" style={{ color: 'var(--_accent)' }}>{result.best_fitness.toFixed(8)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[#777]">Time</span>
          <span className="font-mono text-[#ddd]">{result.elapsed.toFixed(3)}s</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[#777]">Mode</span>
          <span className="text-[#ddd]">{maximize ? 'maximize' : 'minimize'}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[#777]">Epochs</span>
          <span className="font-mono text-[#ddd]">
            {result.total_epochs}
            {result.stopped_early && <span className="text-[#fb923c] ml-1.5 text-[12px]">(early stop)</span>}
          </span>
        </div>
        <div className="flex justify-between col-span-2 pt-2.5 mt-1 border-t border-[#252528]">
          <span className="text-[#777]">Best variables</span>
          <span className="font-mono text-[#ddd]">{result.best_variables.map(v => v.toFixed(6)).join(', ')}</span>
        </div>
        <div className="flex justify-between col-span-2">
          <span className="text-[#777]">Saved</span>
          <span className="font-mono text-[#555] text-[12px]">{result.saved_as}</span>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-[#1a1a1d] border border-[#252528] rounded-xl p-5">
          <div className="text-[14px] text-[#777] mb-4">Fitness</div>
          <div className="h-[230px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid stroke="#222" />
                <XAxis dataKey="epoch" {...axis} />
                <YAxis {...axis} />
                <Tooltip {...tip} />
                <Line type="monotone" dataKey="best" stroke="var(--_accent)" strokeWidth={2} dot={false} name="Best" />
                <Line type="monotone" dataKey="mean" stroke="#555" strokeWidth={1} strokeDasharray="4 3" dot={false} name="Mean" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="bg-[#1a1a1d] border border-[#252528] rounded-xl p-5">
          <div className="text-[14px] text-[#777] mb-4">Std deviation</div>
          <div className="h-[230px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid stroke="#222" />
                <XAxis dataKey="epoch" {...axis} />
                <YAxis {...axis} />
                <Tooltip {...tip} />
                <Line type="monotone" dataKey="std" stroke="#888" strokeWidth={2} dot={false} name="Std" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
