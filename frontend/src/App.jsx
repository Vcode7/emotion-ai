import { useState } from 'react';
import useAppStore from './store/useAppStore';
import AvatarScene from './components/Avatar/AvatarScene';
import ChatArea from './components/Chat/ChatArea';
import ModePanel from './components/Settings/ModePanel';
import EmotionPlayground from './components/Playground/EmotionPlayground';
import VisualChatTab from './components/VisualChat/VisualChatTab';
import FaceEmotionTestTab from './components/FaceEmotion/FaceEmotionTestTab';

export default function App() {
  const [activeTab, setActiveTab] = useState('visual-chat');
  const outputModes = useAppStore((s) => s.outputModes);
  const toggleSettingsPanel = useAppStore((s) => s.toggleSettingsPanel);
  const messages = useAppStore((s) => s.messages);
  const llmApiKey = useAppStore((s) => s.llmApiKey);
  const llmProvider = useAppStore((s) => s.llmProvider);

  const showAvatar = outputModes.avatar;

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-navy-900 bg-grid-pattern relative pt-10">
      {/* ── Fixed Header Bar ── */}
      <header className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-3 sm:px-5 py-2 sm:py-2.5 border-b border-white/[0.08] bg-navy-900/90 backdrop-blur-xl shadow-lg shadow-black/25">
        <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0">
          <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center shadow-md shadow-accent-blue/25">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="8" r="5" />
              <path d="M3 21v-2a7 7 0 0 1 7-7h4a7 7 0 0 1 7 7v2" />
              <circle cx="9" cy="7" r="1" fill="white" />
              <circle cx="15" cy="7" r="1" fill="white" />
            </svg>
          </div>
          <div>
            <h1 className="text-base sm:text-lg font-bold text-gradient leading-tight tracking-tight">DMESR</h1>
            <p className="text-[10px] text-white/40 leading-tight hidden md:block">Emotion-Aware AI</p>
          </div>
        </div>

        {/* Navigation Tabs — Desktop Header Only (Hidden on mobile) */}
        <div className="hidden sm:flex items-center gap-0.5 sm:gap-1 bg-white/[0.04] p-1 rounded-xl border border-white/[0.06] backdrop-blur-sm">
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-all flex items-center gap-1.5 ${
              activeTab === 'chat'
                ? 'bg-gradient-to-br from-accent-blue to-accent-purple text-white shadow-md shadow-accent-blue/20'
                : 'text-white/60 hover:text-white/90 hover:bg-white/5'
            }`}
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
            </svg>
            <span>Chat</span>
            <span className="hidden sm:inline">Mode</span>
          </button>
          <button
            id="visual-chat-tab"
            onClick={() => setActiveTab('visual-chat')}
            className={`px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-all flex items-center gap-1.5 ${
              activeTab === 'visual-chat'
                ? 'bg-gradient-to-br from-accent-blue to-accent-purple text-white shadow-md shadow-accent-blue/20'
                : 'text-white/60 hover:text-white/90 hover:bg-white/5'
            }`}
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"
              />
            </svg>
            <span>Visual</span>
            <span className="hidden sm:inline">Chat</span>
          </button>
          <button
            id="face-test-tab"
            onClick={() => setActiveTab('face-test')}
            className={`px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-all flex items-center gap-1.5 ${
              activeTab === 'face-test'
                ? 'bg-gradient-to-br from-purple-500 to-indigo-600 text-white shadow-md shadow-purple-500/20'
                : 'text-white/60 hover:text-white/90 hover:bg-white/5'
            }`}
          >
            <span>👁️</span>
            <span>Face Test</span>
          </button>
          <button
            onClick={() => setActiveTab('playground')}
            className={`px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-all flex items-center gap-1.5 ${
              activeTab === 'playground'
                ? 'bg-gradient-to-br from-accent-blue to-accent-purple text-white shadow-md shadow-accent-blue/20'
                : 'text-white/60 hover:text-white/90 hover:bg-white/5'
            }`}
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.942a4.5 4.5 0 01-4.66.012L12 15M19.8 15.3c.725.435 1.2 1.218 1.2 2.062v1.888A2.25 2.25 0 0118.75 21.5H5.25A2.25 2.25 0 013 19.25v-1.888c0-.844.475-1.627 1.2-2.062L5 14.5" />
            </svg>
            <span className="hidden sm:inline">Playground</span>
          </button>
        </div>

        <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0">
          {/* API Key status indicator */}
          {!llmApiKey && (
            <button
              onClick={toggleSettingsPanel}
              className="flex items-center gap-1.5 text-xs text-sky-400/80 bg-sky-400/10 hover:bg-sky-400/20 px-2 sm:px-2.5 py-1 rounded-full transition-smooth cursor-pointer"
              title="Server API Key active"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-pulse" />
              <span className="hidden sm:inline">Server Key (.env)</span>
              <span className="sm:hidden text-[10px]">Key</span>
            </button>
          )}

          {/* Settings button */}
          <button
            id="settings-toggle"
            onClick={toggleSettingsPanel}
            className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl glass-panel flex items-center justify-center transition-smooth hover:bg-white/10 hover:scale-105 active:scale-95 cursor-pointer"
            aria-label="Open settings"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white/70">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
            </svg>
          </button>
        </div>
      </header>

      {/* ── Main Content Area (padding compensates for fixed header + mobile bottom nav) ── */}
      <main className="flex-1 flex flex-col min-h-0 pt-[49px] sm:pt-[54px] pb-[54px] sm:pb-0 overflow-y-auto sm:overflow-hidden">
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
        ) : activeTab === 'visual-chat' ? (
          <VisualChatTab />
        ) : activeTab === 'face-test' ? (
          <FaceEmotionTestTab />
        ) : (
          <EmotionPlayground />
        )}
      </main>

      {/* ── Mobile Bottom Navigation Bar (Chat, Visual, Lab) ── */}
      <nav className="sm:hidden fixed bottom-0 left-0 right-0 z-40 bg-navy-900/95 backdrop-blur-xl border-t border-white/[0.08] px-3 py-1 flex items-center justify-around shadow-[0_-4px_24px_rgba(0,0,0,0.5)]">
        <button
          onClick={() => setActiveTab('chat')}
          className={`flex-1 flex flex-col items-center justify-center py-1 rounded-xl transition-all cursor-pointer ${
            activeTab === 'chat'
              ? 'text-accent-cyan font-bold scale-105'
              : 'text-white/40 hover:text-white/70'
          }`}
        >
          <div className={`p-1 rounded-lg transition-all ${activeTab === 'chat' ? 'bg-accent-blue/20 text-accent-cyan shadow-sm shadow-accent-blue/25' : ''}`}>
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
            </svg>
          </div>
          <span className="text-[10px] mt-0.5 tracking-wide">Chat</span>
        </button>

        <button
          id="visual-chat-tab-mobile"
          onClick={() => setActiveTab('visual-chat')}
          className={`flex-1 flex flex-col items-center justify-center py-1 rounded-xl transition-all cursor-pointer ${
            activeTab === 'visual-chat'
              ? 'text-accent-cyan font-bold scale-105'
              : 'text-white/40 hover:text-white/70'
          }`}
        >
          <div className={`p-1 rounded-lg transition-all ${activeTab === 'visual-chat' ? 'bg-accent-blue/20 text-accent-cyan shadow-sm shadow-accent-blue/25' : ''}`}>
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"
              />
            </svg>
          </div>
          <span className="text-[10px] mt-0.5 tracking-wide">Visual</span>
        </button>

        <button
          id="face-test-tab-mobile"
          onClick={() => setActiveTab('face-test')}
          className={`flex-1 flex flex-col items-center justify-center py-1 rounded-xl transition-all cursor-pointer ${
            activeTab === 'face-test'
              ? 'text-purple-400 font-bold scale-105'
              : 'text-white/40 hover:text-white/70'
          }`}
        >
          <div className={`p-1 rounded-lg transition-all ${activeTab === 'face-test' ? 'bg-purple-500/20 text-purple-300 shadow-sm shadow-purple-500/25' : ''}`}>
            <span className="text-lg leading-none">👁️</span>
          </div>
          <span className="text-[10px] mt-0.5 tracking-wide">Face Test</span>
        </button>

        <button
          onClick={() => setActiveTab('playground')}
          className={`flex-1 flex flex-col items-center justify-center py-1 rounded-xl transition-all cursor-pointer ${
            activeTab === 'playground'
              ? 'text-accent-cyan font-bold scale-105'
              : 'text-white/40 hover:text-white/70'
          }`}
        >
          <div className={`p-1 rounded-lg transition-all ${activeTab === 'playground' ? 'bg-accent-blue/20 text-accent-cyan shadow-sm shadow-accent-blue/25' : ''}`}>
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.942a4.5 4.5 0 01-4.66.012L12 15M19.8 15.3c.725.435 1.2 1.218 1.2 2.062v1.888A2.25 2.25 0 0118.75 21.5H5.25A2.25 2.25 0 013 19.25v-1.888c0-.844.475-1.627 1.2-2.062L5 14.5" />
            </svg>
          </div>
          <span className="text-[10px] mt-0.5 tracking-wide">Lab</span>
        </button>
      </nav>

      {/* ── Settings Panel (renders its own backdrop) ── */}
      <ModePanel />
    </div>
  );
}
