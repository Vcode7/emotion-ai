import { useState } from 'react';
import useAppStore from '../../store/useAppStore';

/* ─── Provider Icons ─── */
const PROVIDERS = [
  {
    id: 'openai',
    label: 'OpenAI',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
        <path d="M22.282 9.821a5.985 5.985 0 00-.516-4.91 6.046 6.046 0 00-6.51-2.9A6.065 6.065 0 0011.757.002a6.06 6.06 0 00-5.801 4.218 6.03 6.03 0 00-4.029 2.929 6.05 6.05 0 00.749 7.11 5.98 5.98 0 00.51 4.911 6.05 6.05 0 006.515 2.9A5.99 5.99 0 0013.2 23.998a6.06 6.06 0 005.8-4.218 6.03 6.03 0 004.03-2.93 6.04 6.04 0 00-.748-7.029zM13.2 22.43a4.48 4.48 0 01-2.876-1.04l.141-.081 4.779-2.758a.795.795 0 00.392-.681v-6.737l2.02 1.168a.071.071 0 01.038.052v5.583a4.504 4.504 0 01-4.494 4.494zM3.654 18.439a4.49 4.49 0 01-.535-3.014l.142.085 4.783 2.759a.77.77 0 00.78 0l5.843-3.369v2.332a.08.08 0 01-.033.062L9.74 19.95a4.5 4.5 0 01-6.086-1.511zM2.34 7.896a4.485 4.485 0 012.366-1.973V11.6a.766.766 0 00.388.676l5.815 3.355-2.02 1.168a.076.076 0 01-.071 0l-4.83-2.786A4.504 4.504 0 012.34 7.872v.024zm16.597 3.855l-5.833-3.387L15.119 7.2a.076.076 0 01.071 0l4.83 2.791a4.494 4.494 0 01-.676 8.105v-5.678a.79.79 0 00-.407-.667zm2.01-3.023l-.141-.085-4.774-2.782a.776.776 0 00-.785 0L9.409 9.23V6.897a.066.066 0 01.028-.061l4.83-2.787a4.5 4.5 0 016.68 4.66v.018zm-12.64 4.135l-2.02-1.164a.08.08 0 01-.038-.057V6.075a4.5 4.5 0 017.375-3.453l-.142.08L8.704 5.46a.795.795 0 00-.393.681l-.004 6.722zm1.097-2.365l2.602-1.5 2.607 1.5v2.999l-2.597 1.5-2.607-1.5l-.005-2.999z" />
      </svg>
    ),
  },
  {
    id: 'groq',
    label: 'Groq',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
        <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M8 12h8M12 8v8" strokeWidth="2" stroke="currentColor" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    id: 'gemini',
    label: 'Gemini',
    icon: (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
];

export default function ModePanel() {
  const {
    llmProvider,
    llmApiKey,
    inputMode,
    videoEnabled,
    outputModes,
    settingsPanelOpen,
    setLlmProvider,
    setLlmApiKey,
    setInputMode,
    toggleVideo,
    setOutputMode,
    toggleSettingsPanel,
    resetSession,
  } = useAppStore();

  const [showApiKey, setShowApiKey] = useState(false);

  if (!settingsPanelOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 animate-fade-in"
        onClick={toggleSettingsPanel}
      />

      {/* Panel */}
      <div
        className="fixed top-0 right-0 h-full w-full max-w-sm z-50 glass-panel border-l border-white/[0.08] overflow-y-auto"
        style={{
          animation: 'slideInRight 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards',
          background: 'rgba(10, 14, 26, 0.95)',
          backdropFilter: 'blur(40px)',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/[0.06]">
          <div>
            <h2 className="text-base font-semibold text-white/90">Settings</h2>
            <p className="text-xs text-white/30 mt-0.5">Configure your experience</p>
          </div>
          <button
            onClick={toggleSettingsPanel}
            className="w-8 h-8 rounded-lg bg-white/[0.05] hover:bg-white/[0.1] flex items-center justify-center transition-smooth"
          >
            <svg className="w-4 h-4 text-white/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="px-6 py-5 space-y-7">
          {/* ── LLM Configuration ── */}
          <section>
            <SectionLabel icon="⚡" label="LLM Configuration" />

            {/* Provider selection */}
            <div className="space-y-2 mb-4">
              <label className="text-xs text-white/40 font-medium">Provider</label>
              <div className="grid grid-cols-3 gap-2">
                {PROVIDERS.map((provider) => (
                  <button
                    key={provider.id}
                    onClick={() => setLlmProvider(provider.id)}
                    className={`
                      flex flex-col items-center gap-1.5 px-3 py-2.5 rounded-xl
                      border transition-smooth btn-hover text-center
                      ${llmProvider === provider.id
                        ? 'bg-accent-blue/10 border-accent-blue/40 text-accent-blue'
                        : 'bg-white/[0.03] border-white/[0.06] text-white/50 hover:text-white/70 hover:border-white/[0.12]'
                      }
                    `}
                  >
                    {provider.icon}
                    <span className="text-[11px] font-medium">{provider.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* API Key */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-white/40 font-medium">API Key</label>
                <div className="flex items-center gap-1.5">
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      llmApiKey ? 'bg-emerald-400' : 'bg-sky-400'
                    }`}
                  />
                  <span className="text-[10px] text-white/30">
                    {llmApiKey ? 'Custom Key Set' : 'Using Server .env Key'}
                  </span>
                </div>
              </div>
              <div className="relative">
                <input
                  type={showApiKey ? 'text' : 'password'}
                  value={llmApiKey}
                  onChange={(e) => setLlmApiKey(e.target.value)}
                  placeholder="sk-..."
                  className="w-full px-3 py-2.5 rounded-xl bg-white/[0.04] border border-white/[0.08] text-sm text-white/80 placeholder:text-white/20 focus:border-accent-blue/40 transition-smooth pr-10"
                />
                <button
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60 transition-smooth"
                >
                  {showApiKey ? (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
          </section>

          <Divider />

          {/* ── Input Mode ── */}
          <section>
            <SectionLabel icon="🎯" label="Input Mode" />
            <div className="flex rounded-xl bg-white/[0.03] border border-white/[0.06] p-1">
              <button
                onClick={() => setInputMode('text')}
                className={`
                  flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium
                  transition-smooth
                  ${inputMode === 'text'
                    ? 'bg-accent-blue/15 text-accent-blue border border-accent-blue/20'
                    : 'text-white/40 hover:text-white/60 border border-transparent'
                  }
                `}
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
                </svg>
                Text
              </button>
              <button
                onClick={() => setInputMode('audio')}
                className={`
                  flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium
                  transition-smooth
                  ${inputMode === 'audio'
                    ? 'bg-accent-purple/15 text-accent-purple border border-accent-purple/20'
                    : 'text-white/40 hover:text-white/60 border border-transparent'
                  }
                `}
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
                </svg>
                Audio
              </button>
            </div>
          </section>

          <Divider />

          {/* ── Video ── */}
          <section>
            <SectionLabel icon="📹" label="Video Input" />
            <ToggleSwitch
              checked={videoEnabled}
              onChange={toggleVideo}
              label="Enable Camera"
              description="Analyze facial expressions for emotion detection"
            />
          </section>

          <Divider />

          {/* ── Output Modes ── */}
          <section>
            <SectionLabel icon="🔊" label="Output Modes" />
            <div className="space-y-3">
              <Checkbox
                checked={outputModes.text}
                onChange={(v) => setOutputMode('text', v)}
                label="Text Response"
                description="Show written text responses"
              />
              <Checkbox
                checked={outputModes.voice}
                onChange={(v) => setOutputMode('voice', v)}
                label="Voice Response"
                description="Play spoken audio responses"
              />
              <Checkbox
                checked={outputModes.avatar}
                onChange={(v) => setOutputMode('avatar', v)}
                label="3D Avatar"
                description="Show emotion-reactive robot avatar"
              />
            </div>
          </section>

          <Divider />

          {/* ── Session ── */}
          <section>
            <button
              onClick={resetSession}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-white/[0.03] border border-white/[0.06] text-white/40 hover:text-white/70 hover:bg-white/[0.06] hover:border-white/[0.1] transition-smooth btn-hover text-sm font-medium"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182" />
              </svg>
              Reset Session
            </button>
          </section>
        </div>
      </div>
    </>
  );
}

/* ─── Sub-Components ─── */

function SectionLabel({ icon, label }) {
  return (
    <div className="flex items-center gap-2 mb-3">
      <span className="text-sm">{icon}</span>
      <span className="text-xs font-semibold text-white/60 uppercase tracking-wider">{label}</span>
    </div>
  );
}

function Divider() {
  return <div className="border-t border-white/[0.04]" />;
}

function ToggleSwitch({ checked, onChange, label, description }) {
  return (
    <div
      className="flex items-center justify-between p-3 rounded-xl bg-white/[0.02] border border-white/[0.05] cursor-pointer hover:bg-white/[0.04] transition-smooth"
      onClick={() => onChange(!checked)}
    >
      <div>
        <div className="text-sm font-medium text-white/70">{label}</div>
        {description && (
          <div className="text-[11px] text-white/30 mt-0.5">{description}</div>
        )}
      </div>
      <div
        className={`
          relative w-11 h-6 rounded-full transition-smooth flex-shrink-0
          ${checked
            ? 'bg-accent-blue/30 border border-accent-blue/50'
            : 'bg-white/[0.06] border border-white/[0.1]'
          }
        `}
      >
        <div
          className={`
            absolute top-0.5 w-5 h-5 rounded-full transition-smooth
            ${checked
              ? 'left-[22px] bg-accent-blue shadow-lg shadow-accent-blue/30'
              : 'left-0.5 bg-white/30'
            }
          `}
        />
      </div>
    </div>
  );
}

function Checkbox({ checked, onChange, label, description }) {
  return (
    <div
      className="flex items-center gap-3 p-3 rounded-xl bg-white/[0.02] border border-white/[0.05] cursor-pointer hover:bg-white/[0.04] transition-smooth"
      onClick={() => onChange(!checked)}
    >
      <div
        className={`
          w-5 h-5 rounded-md flex-shrink-0 flex items-center justify-center
          transition-smooth border
          ${checked
            ? 'bg-accent-blue/20 border-accent-blue/50'
            : 'bg-white/[0.04] border-white/[0.1]'
          }
        `}
      >
        {checked && (
          <svg className="w-3 h-3 text-accent-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
          </svg>
        )}
      </div>
      <div className="flex-1">
        <div className="text-sm font-medium text-white/70">{label}</div>
        {description && (
          <div className="text-[11px] text-white/30 mt-0.5">{description}</div>
        )}
      </div>
    </div>
  );
}
