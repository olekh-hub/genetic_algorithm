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
  tick: { fill: '#555', fontSize: 10, fontFamily: 'IBM Plex Mono' },
  axisLine: { stroke: '#2a2a2e' },
  tickLine: { stroke: '#2a2a2e' },
};

const tip = {
  contentStyle: { background: '#1e1e22', border: '1px solid #2a2a2e', borderRadius: 4, fontFamily: 'IBM Plex Mono', fontSize: 11 },
  labelStyle: { color: '#666' },
};

export default function Results({ result, maximize }: Props) {
  const chartData = result.best_history.map((best, i) => ({
    epoch: i, best, mean: result.mean_history[i], std: result.std_history[i],
  }));

  return (
    <div className="space-y-4">
      <div className="text-[15px] font-medium text-[#999]">Results</div>

      {/* Stats as a compact list */}
      <div className="grid grid-cols-2 gap-x-8 gap-y-1.5 text-[13px] bg-[#18181b] rounded border border-[#222] px-5 py-4">
        <div className="flex justify-between">
          <span className="text-[#666]">Best fitness</span>
          <span className="font-mono text-[#d4d4d4]">{result.best_fitness.toFixed(8)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[#666]">Time</span>
          <span className="font-mono text-[#d4d4d4]">{result.elapsed.toFixed(3)}s</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[#666]">Mode</span>
          <span className="text-[#d4d4d4]">{maximize ? 'maximize' : 'minimize'}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-[#666]">Epochs</span>
          <span className="font-mono text-[#d4d4d4]">
            {result.total_epochs}
            {result.stopped_early && <span className="text-[#888] ml-1">(early stop)</span>}
          </span>
        </div>
        <div className="flex justify-between col-span-2 pt-1 border-t border-[#222]">
          <span className="text-[#666]">Best variables</span>
          <span className="font-mono text-[#d4d4d4]">{result.best_variables.map(v => v.toFixed(6)).join(', ')}</span>
        </div>
        <div className="flex justify-between col-span-2">
          <span className="text-[#666]">Saved</span>
          <span className="font-mono text-[#555] text-[11px]">{result.saved_as}</span>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-[#18181b] border border-[#222] rounded p-4">
          <div className="text-[13px] text-[#666] mb-3">Fitness</div>
          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid stroke="#222" />
                <XAxis dataKey="epoch" {...axis} />
                <YAxis {...axis} />
                <Tooltip {...tip} />
                <Line type="monotone" dataKey="best" stroke="#d4d4d4" strokeWidth={1.5} dot={false} name="Best" />
                <Line type="monotone" dataKey="mean" stroke="#555" strokeWidth={1} strokeDasharray="3 3" dot={false} name="Mean" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="bg-[#18181b] border border-[#222] rounded p-4">
          <div className="text-[13px] text-[#666] mb-3">Std deviation</div>
          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid stroke="#222" />
                <XAxis dataKey="epoch" {...axis} />
                <YAxis {...axis} />
                <Tooltip {...tip} />
                <Line type="monotone" dataKey="std" stroke="#888" strokeWidth={1.5} dot={false} name="Std" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
