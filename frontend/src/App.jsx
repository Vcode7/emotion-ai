import { useState } from 'react';
import useAppStore from './store/useAppStore';
import AvatarScene from './components/Avatar/AvatarScene';
import ChatArea from './components/Chat/ChatArea';
import ModePanel from './components/Settings/ModePanel';
import EmotionPlayground from './components/Playground/EmotionPlayground';

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const outputModes = useAppStore((s) => s.outputModes);
  const toggleSettingsPanel = useAppStore((s) => s.toggleSettingsPanel);
  const messages = useAppStore((s) => s.messages);
  const llmApiKey = useAppStore((s) => s.llmApiKey);
  const llmProvider = useAppStore((s) => s.llmProvider);

  const showAvatar = outputModes.avatar;

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-navy-900 bg-grid-pattern relative">
      {/* ── Header Bar ── */}
      <header className="flex-shrink-0 flex items-center justify-between px-5 py-3 border-b border-white/5 z-30 bg-navy-900/80 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center shadow-lg">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="8" r="5" />
              <path d="M3 21v-2a7 7 0 0 1 7-7h4a7 7 0 0 1 7 7v2" />
              <circle cx="9" cy="7" r="1" fill="white" />
              <circle cx="15" cy="7" r="1" fill="white" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-bold text-gradient leading-tight">DMESR</h1>
            <p className="text-[11px] text-white/40 leading-tight">Emotion-Aware Conversational AI</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-1 bg-white/5 p-1 rounded-xl border border-white/5">
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-smooth ${
              activeTab === 'chat'
                ? 'bg-gradient-to-br from-accent-blue to-accent-purple text-white shadow-lg'
                : 'text-white/60 hover:text-white/95 hover:bg-white/5'
            }`}
          >
            Chat Mode
          </button>
          <button
            onClick={() => setActiveTab('playground')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-smooth ${
              activeTab === 'playground'
                ? 'bg-gradient-to-br from-accent-blue to-accent-purple text-white shadow-lg'
                : 'text-white/60 hover:text-white/95 hover:bg-white/5'
            }`}
          >
            Emotion Playground
          </button>
        </div>

        <div className="flex items-center gap-3">
          {/* API Key status indicator */}
          {!llmApiKey && (
            <button
              onClick={toggleSettingsPanel}
              className="flex items-center gap-1.5 text-xs text-sky-400/80 bg-sky-400/10 hover:bg-sky-400/20 px-2.5 py-1 rounded-full transition-smooth"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
              Server Key (.env)
            </button>
          )}

          {/* Settings button */}
          <button
            id="settings-toggle"
            onClick={toggleSettingsPanel}
            className="w-10 h-10 rounded-xl glass-panel flex items-center justify-center transition-smooth hover:bg-white/10 hover:scale-105 active:scale-95"
            aria-label="Open settings"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white/70">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
            </svg>
          </button>
        </div>
      </header>

      {/* ── Main Content ── */}
      <main className="flex-1 flex flex-col min-h-0 overflow-hidden">
        {activeTab === 'chat' ? (
          <>
            {/* Avatar Area — top ~45% */}
            {showAvatar && (
              <div className="flex-shrink-0 relative" style={{ height: '45%' }}>
                <AvatarScene />
                {/* Gradient fade at bottom */}
                <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-navy-900 to-transparent pointer-events-none z-10" />
              </div>
            )}

            {/* Chat Area — fills remaining space */}
            <div className={`flex-1 min-h-0 ${showAvatar ? '' : 'h-full'}`}>
              <ChatArea />
            </div>
          </>
        ) : (
          <EmotionPlayground />
        )}
      </main>

      {/* ── Settings Panel (renders its own backdrop) ── */}
      <ModePanel />
    </div>
  );
}
