'use client';

import { RotateCcw } from 'lucide-react';
import { LLMSettings, DEFAULT_LLM_SETTINGS } from '@/types';

const MODELS = [
  { key: 'gemma-4-e2b',      label: 'Gemma 4 e2B',           badge: 'Local' },
  { key: 'gemma-4-e4b',      label: 'Gemma 4 e4B',           badge: 'Local' },
  { key: 'gemma-4-31b',      label: 'Gemma 4 31B',           badge: 'API'   },
  { key: 'gemini-flash-lite', label: 'Gemini 3.1 Flash Lite', badge: 'API'  },
];

const SEARCH_MODES: { key: LLMSettings['search_mode']; label: string }[] = [
  { key: 'semantic', label: 'Semantic' },
  { key: 'keyword',  label: 'Keyword'  },
  { key: 'hybrid',   label: 'Hybrid'   },
];

interface SliderRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}

function SliderRow({ label, value, min, max, step, display, onChange }: SliderRowProps) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center">
        <label className="text-xs font-medium text-gray-700 dark:text-gray-300">{label}</label>
        <span className="text-xs font-mono text-diamond-blue">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer accent-diamond-blue"
      />
      <div className="flex justify-between text-[10px] text-gray-400 dark:text-gray-500">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

interface LLMSettingsPanelProps {
  settings: LLMSettings;
  onChange: (s: LLMSettings) => void;
}

export default function LLMSettingsPanel({ settings, onChange }: LLMSettingsPanelProps) {
  const set = <K extends keyof LLMSettings>(key: K, value: LLMSettings[K]) =>
    onChange({ ...settings, [key]: value });

  const activeModel = MODELS.find(m => m.key === settings.model);

  return (
    <div className="w-64 shrink-0 h-full overflow-y-auto border-l border-gray-300 dark:border-gray-700 glass glass-light dark:glass-dark p-4 flex flex-col gap-5">

      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="font-bold text-sm text-gray-900 dark:text-gray-100">
          LLM Settings
        </span>
        <button
          onClick={() => onChange(DEFAULT_LLM_SETTINGS)}
          className="text-gray-400 hover:text-ore-gold transition-colors"
          title="Reset to defaults"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Model */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-gray-700 dark:text-gray-300">Model</label>
        <select
          value={settings.model}
          onChange={e => set('model', e.target.value)}
          className="w-full px-2 py-1.5 rounded-lg border border-gray-300 dark:border-gray-600 bg-white/50 dark:bg-black/50 text-gray-900 dark:text-gray-100 text-xs focus:outline-none focus:ring-1 focus:ring-diamond-blue"
        >
          {MODELS.map(m => (
            <option key={m.key} value={m.key}>
              {m.label} ({m.badge})
            </option>
          ))}
        </select>
        {activeModel && (
          <p className="text-[10px] text-gray-500 dark:text-gray-400">
            {activeModel.badge === 'Local'
              ? '🖥 Served by Ollama (local)'
              : '☁ Served by OpenRouter (API)'}
          </p>
        )}
      </div>

      {/* Search Mode */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-gray-700 dark:text-gray-300">Search Mode</label>
        <div className="flex rounded-lg overflow-hidden border border-gray-300 dark:border-gray-600">
          {SEARCH_MODES.map(m => (
            <button
              key={m.key}
              type="button"
              onClick={() => set('search_mode', m.key)}
              className={`flex-1 py-1 text-xs font-medium transition-colors ${
                settings.search_mode === m.key
                  ? 'bg-diamond-blue text-white'
                  : 'bg-white/30 dark:bg-black/30 text-gray-700 dark:text-gray-300 hover:bg-white/60 dark:hover:bg-black/60'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
        {settings.search_mode === 'semantic' && (
          <p className="text-[10px] text-gray-500 dark:text-gray-400">
            Recommended — best MRR on eval set
          </p>
        )}
      </div>

      <div className="border-t border-gray-300 dark:border-gray-700" />

      {/* Sliders */}
      <div className="space-y-5">
        <SliderRow
          label="Temperature"
          value={settings.temperature}
          min={0} max={2} step={0.05}
          display={settings.temperature.toFixed(2)}
          onChange={v => set('temperature', v)}
        />
        <SliderRow
          label="Top P"
          value={settings.top_p}
          min={0} max={1} step={0.05}
          display={settings.top_p.toFixed(2)}
          onChange={v => set('top_p', v)}
        />
        <SliderRow
          label="Max Tokens"
          value={settings.max_tokens}
          min={64} max={2048} step={64}
          display={settings.max_tokens.toString()}
          onChange={v => set('max_tokens', v)}
        />
      </div>

      <div className="border-t border-gray-300 dark:border-gray-700" />

      {/* Thinking mode */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-medium text-gray-700 dark:text-gray-300">Thinking Mode</p>
          <p className="text-[10px] text-gray-500 dark:text-gray-400 mt-0.5">Show reasoning process (Gemma 4 / Gemini)</p>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={settings.thinking}
          onClick={() => set('thinking', !settings.thinking)}
          className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${
            settings.thinking ? 'bg-diamond-blue' : 'bg-gray-300 dark:bg-gray-600'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${
              settings.thinking ? 'translate-x-4' : 'translate-x-0'
            }`}
          />
        </button>
      </div>
    </div>
  );
}
