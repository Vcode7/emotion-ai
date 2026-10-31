import { useRef, useEffect, useState, useCallback } from 'react';
import { useProgress } from '@react-three/drei';
import VisualChatAvatar from './VisualChatAvatar';
import WebcamPanel from './WebcamPanel';
import useVisualChat from './useVisualChat';

/* ── Status badge helpers ── */
const STATUS_CONFIG = {
  idle: { label: 'Ready to start', color: 'text-white/40', dot: 'bg-white/20' },
  listening: { label: 'Listening…', color: 'text-emerald-400', dot: 'bg-emerald-400 animate-pulse' },
  processing: { label: 'Generating Answer…', color: 'text-accent-cyan', dot: 'bg-accent-cyan animate-pulse' },
  speaking: { label: 'Speaking…', color: 'text-accent-cyan', dot: 'bg-accent-cyan animate-ping' },
};

/* ── 4-Modality Emotion Inspector Panel ── */
function EmotionModalitiesPanel({ multimodalEmotions, isProcessing, compact = false }) {
  if (!multimodalEmotions) {
    return (
      <div className={`flex-shrink-0 ${compact ? 'px-3 py-1.5' : 'mx-3 sm:mx-5 mt-2 px-3 sm:px-4 py-2'} rounded-xl bg-white/[0.02] border border-white/[0.05] flex items-center justify-between`}>
        <div className="flex items-center gap-2 text-xs text-white/30 truncate">
          <span className="w-2 h-2 rounded-full bg-accent-blue/40 flex-shrink-0" />
          <span className="font-medium text-white/40 hidden xs:inline">4-Modality Engine:</span>
          <span className="text-white/25 truncate">Text (RoBERTa) · Voice (Wav2Vec2) · Face (ViT) · DSMR → fused</span>
        </div>
        <span className="text-[10px] text-white/25 uppercase tracking-widest font-mono flex-shrink-0 ml-2">Standby</span>
      </div>
    );
  }

  const modalities = [
    {
      id: 'text',
      label: 'Text Emotion',
      source: 'RoBERTa GoEmotions',
      data: multimodalEmotions.text,
      icon: '📝',
      tagColor: 'text-blue-400',
      barColor: 'from-blue-500 to-cyan-400',
    },
    {
      id: 'audio',
      label: 'Voice Emotion',
      source: 'Wav2Vec2 SER',
      data: multimodalEmotions.audio,
      icon: '🎙️',
      tagColor: 'text-emerald-400',
      barColor: 'from-emerald-500 to-teal-400',
    },
    {
      id: 'face',
      label: 'Face Emotion',
      source: 'ViT (HardlyHumans)',
      data: multimodalEmotions.face,
      icon: '👁️',
      tagColor: 'text-purple-400',
      barColor: 'from-purple-500 to-pink-400',
      specialText: multimodalEmotions.face?.face_detected === false ? 'No face detected' : null,
    },
    {
      id: 'dsmr',
      label: 'DSMR Sentiment',
      source: 'MOSEI sentiment',
      data: multimodalEmotions.dsmr,
      // DMESR predicts a sentiment scalar, not class probabilities — show valence.
      formatValue: (d) => (d?.valence != null
        ? `${d.valence >= 0 ? '+' : ''}${Number(d.valence).toFixed(2)}`
        : null),
      icon: '🧠',
      tagColor: 'text-amber-400',
      barColor: 'from-amber-500 to-orange-400',
    },
  ];

  return (
    <div className={`flex-shrink-0 ${compact ? 'p-2' : 'mx-3 sm:mx-5 mt-2 sm:mt-2.5 p-2 sm:p-2.5'} rounded-2xl bg-white/[0.025] border border-white/[0.08] backdrop-blur-md`}>
      <div className="flex items-center justify-between mb-1.5 sm:mb-2 px-1">
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse" />
          <span className="text-[10px] font-bold uppercase tracking-widest text-white/60">
            Multimodal Emotion Recognition (4 Models)
          </span>
        </div>
        {!isProcessing && multimodalEmotions.fused?.confidence > 0 && (
          <span className="text-[10px] text-white/70 font-medium truncate ml-2">
            Overall:{' '}
            <strong className="text-accent-cyan capitalize">{multimodalEmotions.fused.emotion}</strong>{' '}
            {Math.round(multimodalEmotions.fused.confidence * 100)}% · {multimodalEmotions.fused.agreement} agree
            {multimodalEmotions.fused.conflict ? ' · mismatch' : ''}
          </span>
        )}
        {isProcessing && (
          <span className="text-[10px] text-accent-blue font-medium animate-pulse">
            Analyzing…
          </span>
        )}
      </div>

      {/* Responsive grid: 2 cols on mobile, 4 cols on desktop */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-1.5 sm:gap-2">
        {modalities.map((m) => {
          const emotion = m.data?.emotion || 'neutral';
          const confidence = Math.round((m.data?.confidence || 0) * 100);
          const isAvail = m.data?.available !== false;
          const displayLabel = m.specialText || (!isAvail ? 'Unavailable' : emotion);

          return (
            <div
              key={m.id}
              className="p-2 sm:p-2.5 rounded-xl bg-white/[0.03] border border-white/[0.05] flex flex-col justify-between transition-all hover:border-white/15"
            >
              <div className="flex items-center justify-between text-[10px] text-white/40 mb-1">
                <span className="flex items-center gap-1 font-medium truncate">
                  <span className="text-xs">{m.icon}</span>
                  <span className="truncate">{m.label}</span>
                </span>
                <span className="text-[8px] sm:text-[9px] text-white/25 font-mono hidden xs:inline">{m.source}</span>
              </div>

              <div className="flex items-baseline justify-between mt-0.5">
                <span className={`text-xs font-semibold uppercase tracking-wider capitalize truncate ${isAvail ? m.tagColor : 'text-white/40'}`}>
                  {displayLabel}
                </span>
                {isAvail && !m.specialText && (
                  <span className="text-[10px] font-mono text-white/40 ml-1 flex-shrink-0">
                    {m.formatValue?.(m.data) ?? `${confidence}%`}
                  </span>
                )}
              </div>

              {/* Confidence progress bar */}
              <div className="w-full bg-white/5 rounded-full h-1 mt-1.5 overflow-hidden">
                <div
                  className={`bg-gradient-to-r ${m.barColor} h-full rounded-full transition-all duration-500`}
                  style={{ width: `${(!isAvail || m.specialText) ? 0 : confidence}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(() =>
    typeof window !== 'undefined' ? window.innerWidth < 768 : false
  );

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mql = window.matchMedia('(max-width: 767px)');
    const onChange = (e) => setIsMobile(e.matches);
    setIsMobile(mql.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);

  return isMobile;
}

/* ── Full Page 3D Avatar Loading Overlay ── */
function AvatarLoadingOverlay({ progress, onDismiss, showTimeout }) {
  const roundedProgress = Math.min(100, Math.max(5, Math.round(progress)));

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col items-center justify-center select-none p-6 text-center animate-fade-in"
      style={{
        background: 'radial-gradient(circle at 50% 40%, rgba(11, 19, 38, 0.98) 0%, rgba(3, 7, 18, 0.99) 70%, #020617 100%)',
        backdropFilter: 'blur(28px)',
        WebkitBackdropFilter: 'blur(28px)',
      }}
    >
      {/* Ambient background glow orbs */}
      <div className="absolute w-[500px] h-[500px] rounded-full bg-accent-blue/15 blur-[120px] pointer-events-none -top-24 -left-24 animate-pulse" />
      <div className="absolute w-[500px] h-[500px] rounded-full bg-accent-purple/15 blur-[120px] pointer-events-none -bottom-24 -right-24 animate-pulse" />

      {/* Holographic 3D Avatar Scanner Ring */}
      <div className="relative mb-6 sm:mb-8 flex items-center justify-center">
        {/* Pulsing outer concentric rings */}
        <div className="absolute -inset-6 rounded-full border border-accent-cyan/20 animate-ping opacity-25" />
        <div className="absolute -inset-3 rounded-full border border-accent-blue/40 animate-pulse" />

        {/* Central glowing icon container */}
        <div className="w-20 h-20 sm:w-24 sm:h-24 rounded-3xl bg-gradient-to-br from-white/[0.08] to-white/[0.02] border border-white/20 backdrop-blur-xl flex items-center justify-center shadow-[0_0_50px_rgba(6,182,212,0.35)] relative overflow-hidden">
          {/* Laser scanning line sweeping down */}
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-accent-cyan/40 to-transparent animate-loading-sweep pointer-events-none" />

          {/* 3D Avatar SVG */}
          <svg className="w-10 h-10 sm:w-12 sm:h-12 text-accent-cyan drop-shadow-[0_0_12px_rgba(6,182,212,0.8)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.4}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 2a7 7 0 00-7 7c0 2.38 1.19 4.47 3 5.74V17a2 2 0 002 2h4a2 2 0 002-2v-2.26c1.81-1.27 3-3.36 3-5.74a7 7 0 00-7-7z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 13c.5.5 1.5 1 3 1s2.5-.5 3-1" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M10 9h.01M14 9h.01" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M8 21h8" />
          </svg>
        </div>
      </div>

      {/* Cyber Tag */}
      <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-accent-cyan/10 border border-accent-cyan/30 mb-3 shadow-[0_0_15px_rgba(6,182,212,0.2)]">
        <span className="w-2 h-2 rounded-full bg-accent-cyan animate-pulse" />
        <span className="text-[10px] sm:text-xs font-bold tracking-widest uppercase text-accent-cyan font-mono">
          3D Avatar Engine
        </span>
      </div>

      {/* Main Heading */}
      <h2 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight mb-2">
        Loading 3D Avatar…
      </h2>

      {/* Subtitle */}
      <p className="text-xs sm:text-sm text-white/60 max-w-md mb-8 leading-relaxed font-normal">
        Preparing 3D stylized avatar geometry, ARKit blendshapes, and audio-driven lip sync
      </p>

      {/* Progress Bar Container */}
      <div className="w-72 sm:w-96 max-w-full flex flex-col gap-2.5">
        <div className="relative w-full h-3.5 rounded-full bg-white/10 p-0.5 border border-white/20 overflow-hidden backdrop-blur-md shadow-inner">
          <div
            className="h-full rounded-full bg-gradient-to-r from-accent-cyan via-accent-blue to-accent-purple shadow-[0_0_16px_rgba(6,182,212,0.9)] transition-all duration-300 relative overflow-hidden"
            style={{ width: `${roundedProgress}%` }}
          >
            {/* Shimmer light sweep */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/50 to-transparent animate-loading-sweep" />
          </div>
        </div>

        {/* Progress % and Status Text */}
        <div className="flex items-center justify-between text-xs text-white/50 px-1 font-mono">
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-ping" />
            <span className="text-white/70">
              {roundedProgress >= 100 ? 'Avatar Ready!' : 'Loading Avatar Model…'}
            </span>
          </span>
          <span className="font-bold text-accent-cyan">{roundedProgress}%</span>
        </div>
      </div>

      {/* Feature Pills */}
      <div className="mt-8 flex flex-wrap items-center justify-center gap-2 max-w-lg">
        <span className="text-[10px] sm:text-xs px-3 py-1 rounded-lg bg-white/[0.04] border border-white/[0.08] text-white/50">
          66 ARKit Facial Expressions
        </span>
        <span className="text-[10px] sm:text-xs px-3 py-1 rounded-lg bg-white/[0.04] border border-white/[0.08] text-white/50">
          Real-Time Audio Lip Sync
        </span>
        <span className="text-[10px] sm:text-xs px-3 py-1 rounded-lg bg-white/[0.04] border border-white/[0.08] text-white/50">
          Fast In-Memory Cache
        </span>
      </div>

      {/* Optional fallback button if slow connection */}
      {showTimeout && (
        <div className="mt-6 flex flex-col items-center gap-2 animate-fade-in">
          <p className="text-xs text-amber-400/90 font-medium">Loading is taking longer than expected.</p>
          <button
            onClick={onDismiss}
            className="px-4 py-1.5 rounded-xl bg-white/10 hover:bg-white/20 border border-white/25 text-xs font-semibold text-white transition-all cursor-pointer shadow-md"
          >
            Enter Visual Chat
          </button>
        </div>
      )}
    </div>
  );
}

export default function VisualChatTab() {
  const conversationEndRef = useRef(null);
  const fullModeChatEndRef = useRef(null);
  const isInitialMount = useRef(true);

  // Full Screen / Video Call Mode toggle
  const [isFullMode, setIsFullMode] = useState(false);
  // View swap between main video and corner video
  const [swappedViews, setSwappedViews] = useState(false);
  // Emotion button & expanded card toggle state
  const [showEmotionButton, setShowEmotionButton] = useState(false);
  const [isEmotionCardExpanded, setIsEmotionCardExpanded] = useState(false);

  const handleBrainClick = () => {
    setShowEmotionButton((prev) => {
      const next = !prev;
      if (!next) setIsEmotionCardExpanded(false);
      return next;
    });
  };

  // 3D Avatar Loading State & Progress
  const [isAvatarLoaded, setIsAvatarLoaded] = useState(false);
  const [showLoadingOverlay, setShowLoadingOverlay] = useState(true);
  const [displayProgress, setDisplayProgress] = useState(15);
  const [loadTimeout, setLoadTimeout] = useState(false);
  const { progress: loadingProgress } = useProgress();

  const handleAvatarLoaded = useCallback(() => {
    setIsAvatarLoaded(true);
    setDisplayProgress(100);
    setTimeout(() => {
      setShowLoadingOverlay(false);
    }, 450);
  }, []);

  useEffect(() => {
    if (isAvatarLoaded) {
      setDisplayProgress(100);
      return;
    }
    if (loadingProgress > 0) {
      setDisplayProgress((prev) => Math.max(prev, Math.round(loadingProgress)));
    } else {
      const timer = setInterval(() => {
        setDisplayProgress((prev) => {
          if (prev >= 88) return prev;
          return prev + 4;
        });
      }, 160);
      return () => clearInterval(timer);
    }
  }, [isAvatarLoaded, loadingProgress]);

  // Safety timeout in case network stalls
  useEffect(() => {
    if (isAvatarLoaded) return;
    const timer = setTimeout(() => {
      setLoadTimeout(true);
    }, 18000);
    return () => clearTimeout(timer);
  }, [isAvatarLoaded]);

  const isMobile = useIsMobile();

  const {
    isActive,
    status,
    transcript,
    conversation,
    multimodalEmotions,
    webcamStream,
    callError,
    setCallError,
    startConversation,
    stopConversation,
  } = useVisualChat();

  const statusInfo = STATUS_CONFIG[status] || STATUS_CONFIG.idle;

  // Ensure the page strictly remains at the top on initial load
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
    const mainEl = document.querySelector('main');
    if (mainEl) mainEl.scrollTop = 0;
  }, []);

  // Scroll only the inner conversation log container when messages arrive, never scrolling the main page
  useEffect(() => {
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }

    if (conversation.length > 0 || (transcript && status === 'processing')) {
      const chatContainer = conversationEndRef.current?.parentElement;
      if (chatContainer && chatContainer.scrollHeight > chatContainer.clientHeight) {
        chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
      }

      const fullChatContainer = fullModeChatEndRef.current?.parentElement;
      if (fullChatContainer && fullChatContainer.scrollHeight > fullChatContainer.clientHeight) {
        fullChatContainer.scrollTo({ top: fullChatContainer.scrollHeight, behavior: 'smooth' });
      }
    }
  }, [conversation, transcript, status]);

  const handleToggle = () => {
    if (isActive) {
      stopConversation();
    } else {
      startConversation();
    }
  };

  // ──────────────────────────────────────────────────────────────────────────
  //  FULL MODE: IMMERSIVE AI VIDEO CALL WITH SMALL USER CORNER DISPLAY
  // ──────────────────────────────────────────────────────────────────────────
  if (isFullMode) {
    return (
      <div className="fixed inset-0 z-50 flex flex-col overflow-hidden bg-black select-none">
        {/* Full Page 3D Avatar Loading Overlay */}
        {showLoadingOverlay && (
          <AvatarLoadingOverlay
            progress={displayProgress}
            onDismiss={() => setShowLoadingOverlay(false)}
            showTimeout={loadTimeout}
          />
        )}

        {/* ── AI Avatar View (Main or Corner PiP) ── */}
        <div
          className={`transition-all duration-300 ${
            !swappedViews
              ? 'absolute inset-0 z-0 bg-gradient-to-b from-navy-950 via-navy-900 to-black'
              : 'absolute top-14 right-3.5 z-40 w-24 h-32 sm:w-28 sm:h-38 rounded-2xl overflow-hidden shadow-2xl border-2 border-white/30 backdrop-blur-md active:scale-95 cursor-pointer hover:border-accent-cyan/70 hover:scale-105'
          }`}
          onClick={swappedViews ? () => setSwappedViews(false) : undefined}
          title={swappedViews ? 'Tap to make AI avatar main view' : undefined}
        >
          <div className="w-full h-full">
            <VisualChatAvatar
              status={status}
              isFullMode={!swappedViews}
              isPip={swappedViews}
              onLoaded={handleAvatarLoaded}
            />
          </div>
          {swappedViews && (
            <div className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded-md bg-black/75 text-[8px] text-white/70 flex items-center gap-0.5 font-medium pointer-events-none">
              <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
              </svg>
              <span>Swap</span>
            </div>
          )}
        </div>

        {/* ── Webcam View (Main or Corner PiP) ── */}
        <div
          className={`transition-all duration-300 ${
            swappedViews
              ? 'absolute inset-0 z-0 bg-gradient-to-b from-navy-950 via-navy-900 to-black'
              : 'absolute top-14 right-3.5 z-40 w-24 h-32 sm:w-28 sm:h-38 rounded-2xl overflow-hidden shadow-2xl border-2 border-white/30 backdrop-blur-md active:scale-95 cursor-pointer hover:border-accent-cyan/70 hover:scale-105'
          }`}
          onClick={!swappedViews ? () => setSwappedViews(true) : undefined}
          title={!swappedViews ? 'Tap to make camera main view' : undefined}
        >
          <WebcamPanel
            stream={webcamStream}
            isActive={isActive}
            isPip={!swappedViews}
            className={swappedViews ? 'rounded-none border-0' : 'rounded-2xl'}
          />
          {!swappedViews && (
            <div className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded-md bg-black/75 text-[8px] text-white/70 flex items-center gap-0.5 font-medium pointer-events-none">
              <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
              </svg>
              <span>Swap</span>
            </div>
          )}
        </div>

        {/* ── Top Floating Header Overlay ── */}
        <div className="absolute top-0 left-0 right-0 z-30 flex items-center justify-between px-3.5 py-3 bg-gradient-to-b from-black/85 via-black/40 to-transparent">
          <div className="flex items-center gap-2">
            <div className={`w-2.5 h-2.5 rounded-full ${statusInfo.dot}`} />
            <span className="text-xs font-semibold text-white/90 drop-shadow-md">
              {statusInfo.label}
            </span>
            {isActive && (
              <span className="px-2 py-0.5 rounded-full bg-red-500/20 border border-red-500/40 text-[9px] text-red-400 font-bold uppercase tracking-wider animate-pulse ml-1">
                LIVE
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* Brain icon toggle button */}
            <button
              onClick={handleBrainClick}
              className={`px-2.5 py-1.5 rounded-full text-xs font-semibold flex items-center gap-1.5 backdrop-blur-md border transition-all cursor-pointer shadow-lg active:scale-95 ${
                showEmotionButton
                  ? 'bg-accent-blue/40 border-accent-blue/80 text-white shadow-accent-blue/20'
                  : 'bg-black/50 hover:bg-black/70 border-white/20 text-white/80'
              }`}
              title="Emotions"
            >
              <span className="text-xs">🧠</span>
              <span className="hidden xs:inline text-[11px]">Emotions</span>
            </button>

            {/* Exit Full View Button */}
            <button
              onClick={() => setIsFullMode(false)}
              className="px-2.5 py-1.5 rounded-full bg-black/60 hover:bg-black/80 border border-white/20 text-white/80 hover:text-white text-xs font-semibold flex items-center gap-1.5 backdrop-blur-md shadow-lg transition-all active:scale-95 cursor-pointer"
              title="Exit Full View"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 9V4.5M9 9H4.5M9 9L3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5l5.25 5.25" />
              </svg>
              <span className="text-[11px] font-semibold">Exit Full View</span>
            </button>
          </div>
        </div>

        {/* ── When brain icon is clicked, show a button labeled "Show Emotions" ── */}
        {showEmotionButton && (
          <div className="absolute top-14 left-3 right-3 sm:right-36 z-35 flex flex-col items-center gap-2 animate-fade-in">
            <button
              onClick={() => setIsEmotionCardExpanded(!isEmotionCardExpanded)}
              className="px-4 py-2 rounded-2xl bg-gradient-to-r from-accent-blue/40 via-accent-purple/40 to-accent-cyan/40 hover:from-accent-blue/50 hover:to-accent-purple/50 border border-accent-cyan/50 text-white font-semibold text-xs sm:text-sm flex items-center gap-2 shadow-[0_4px_24px_rgba(6,182,212,0.35)] backdrop-blur-2xl transition-all active:scale-95 cursor-pointer"
            >
              <span>🧠</span>
              <span className="font-bold tracking-wide">{isEmotionCardExpanded ? 'Hide Emotions' : 'Show Emotions'}</span>
              <span className="text-[10px] text-accent-cyan">{isEmotionCardExpanded ? '▲' : '▼'}</span>
            </button>

            {isEmotionCardExpanded && (
              <div className="w-full max-h-[48vh] overflow-y-auto rounded-2xl shadow-2xl animate-scale-in">
                <EmotionModalitiesPanel
                  multimodalEmotions={multimodalEmotions}
                  isProcessing={status === 'processing'}
                  compact={true}
                />
              </div>
            )}
          </div>
        )}

        {/* ── Error Banner inside Full Mode ── */}
        {callError && (
          <div className="absolute top-14 left-3 right-3 sm:right-36 z-45 p-2.5 rounded-xl bg-red-500/90 text-white text-xs flex items-center justify-between shadow-xl backdrop-blur-md animate-fade-in">
            <span>⚠️ {callError}</span>
            <button onClick={() => setCallError(null)} className="ml-2 font-bold px-2 py-0.5 bg-black/30 rounded">✕</button>
          </div>
        )}

        {/* ── Floating Live Subtitles / Conversation Card (Overlay above bottom controls) ── */}
        <div className="absolute bottom-20 left-3 right-3 sm:left-6 sm:right-6 z-30 max-h-36 overflow-y-auto rounded-2xl bg-black/60 backdrop-blur-xl border border-white/15 p-3 shadow-2xl transition-all">
          {conversation.length === 0 && !transcript ? (
            <div className="text-center py-1">
              <p className="text-xs text-white/50 font-medium">
                {isActive ? 'Speak naturally — voice, face, text & DSMR evaluate each turn' : 'Tap Start Call below to begin AI conversation'}
              </p>
            </div>
          ) : (
            <div className="space-y-1.5 text-xs">
              {/* Show most recent conversation turn */}
              {conversation.slice(-2).map((msg) => (
                <div key={msg.id} className="leading-snug">
                  {msg.role === 'user' ? (
                    <p className="text-white/70">
                      <span className="text-white/40 font-semibold uppercase text-[9px] mr-1.5">You:</span>
                      {msg.text}
                    </p>
                  ) : (
                    <div>
                      {msg.emotion && (
                        <span className="inline-flex items-center gap-1 text-[9px] text-accent-cyan font-bold uppercase tracking-wider bg-accent-cyan/20 px-2 py-0.5 rounded-full border border-accent-cyan/40 mb-1">
                          <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse" />
                          AI: {msg.emotion}
                        </span>
                      )}
                      <p className="text-white/95 font-medium">{msg.text}</p>
                    </div>
                  )}
                </div>
              ))}

              {/* Real-time transcript while speaking */}
              {transcript && (
                <p className="text-xs text-white/90 italic pt-1 border-t border-white/10">
                  <span className="text-white/40 not-italic font-semibold text-[9px] uppercase mr-1.5">You said:</span>
                  {transcript}
                </p>
              )}

              {/* Generating Answer loading bar in full-screen subtitles card */}
              {status === 'processing' && (
                <div className="flex items-center gap-2 pt-1.5 border-t border-white/10 animate-fade-in">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-cyan opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-accent-cyan" />
                  </span>
                  <span className="text-[11px] font-semibold text-accent-cyan flex-shrink-0">Generating answer…</span>
                  <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden relative border border-white/10">
                    <div className="h-full w-full bg-gradient-to-r from-accent-cyan via-accent-blue to-accent-purple animate-loading-sweep rounded-full shadow-[0_0_8px_rgba(6,182,212,0.5)]" />
                  </div>
                </div>
              )}
              <div ref={fullModeChatEndRef} />
            </div>
          )}
        </div>

        {/* ── Floating Bottom Call Control Bar ── */}
        <div className="absolute bottom-3 left-0 right-0 z-30 flex items-center justify-center gap-3 px-4">
          <div className="flex items-center gap-3 bg-black/70 backdrop-blur-xl border border-white/20 p-2 rounded-full shadow-2xl">

            {/* Swap View Button */}
            <button
              onClick={() => setSwappedViews(!swappedViews)}
              className="w-10 h-10 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-white/80 hover:text-white transition-all active:scale-95 cursor-pointer"
              title="Swap main & corner camera"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
              </svg>
            </button>

            {/* Central Primary Call Toggle Button */}
            <button
              onClick={handleToggle}
              className={`
                px-5 py-2.5 rounded-full flex items-center gap-2 shadow-xl cursor-pointer transition-all active:scale-95
                ${isActive
                  ? 'bg-red-500 hover:bg-red-600 text-white shadow-red-500/40 border border-red-400/50 scale-105'
                  : 'bg-gradient-to-r from-accent-blue to-accent-purple hover:brightness-110 text-white shadow-accent-blue/30 border border-white/20'
                }
              `}
              aria-label={isActive ? 'End Call' : 'Start Call'}
            >
              {isActive ? (
                <>
                  <span className="w-3 h-3 rounded-sm bg-white" />
                  <span className="text-xs font-bold tracking-wide uppercase">End Call</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round"
                      d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z"
                    />
                  </svg>
                  <span className="text-xs font-bold tracking-wide uppercase">Start Call</span>
                </>
              )}
            </button>

            {/* Toggle HUD button */}
            <button
              onClick={handleBrainClick}
              className={`w-10 h-10 rounded-full flex items-center justify-center transition-all active:scale-95 cursor-pointer ${
                showEmotionButton ? 'bg-accent-blue text-white shadow-lg shadow-accent-blue/40' : 'bg-white/10 hover:bg-white/20 text-white/80'
              }`}
              title="Emotions"
            >
              <span className="text-sm">🧠</span>
            </button>

          </div>
        </div>

      </div>
    );
  }

  // ──────────────────────────────────────────────────────────────────────────
  //  STANDARD DASHBOARD VIEW (Split Screen & Conversation Log)
  // ──────────────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col md:h-full overflow-y-auto md:overflow-hidden bg-navy-900 relative">
      {/* Full Page 3D Avatar Loading Overlay */}
      {showLoadingOverlay && (
        <AvatarLoadingOverlay
          progress={displayProgress}
          onDismiss={() => setShowLoadingOverlay(false)}
          showTimeout={loadTimeout}
        />
      )}

      {/* ── Sub-header / Status Bar ── */}
      {/* <div className="flex-shrink-0 flex items-center justify-between px-3 sm:px-5 py-2 sm:py-2.5 border-b border-white/[0.05] bg-navy-900/40">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${statusInfo.dot}`} />
          <span className={`text-xs sm:text-sm font-medium ${statusInfo.color}`}>{statusInfo.label}</span>
          {isActive && (
            <div className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-500/10 border border-red-500/20 ml-1">
              <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
              <span className="text-[9px] text-red-400 font-medium">LIVE</span>
            </div>
          )}
        </div>

        {/* Mode Switcher: Standard vs Full View 
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setIsFullMode(true)}
            className="px-2.5 py-1 rounded-lg text-xs font-semibold bg-gradient-to-r from-accent-blue/20 to-accent-purple/20 hover:from-accent-blue/30 hover:to-accent-purple/30 border border-accent-blue/40 text-accent-cyan flex items-center gap-1.5 shadow-sm transition-all active:scale-95 cursor-pointer"
            title="Switch to Full View"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
            </svg>
            <span className="text-[11px] font-bold">Full View</span>
          </button>
        </div>
      </div> */}

      {/* ── Optional Call Error Banner ── */}
      {callError && (
        <div className="mx-3 sm:mx-5 mt-2 p-2.5 rounded-xl bg-red-500/15 border border-red-500/30 flex items-center justify-between text-xs text-red-300 animate-fade-in">
          <div className="flex items-center gap-2">
            <span>⚠️</span>
            <span>{callError}</span>
          </div>
          <button onClick={() => setCallError(null)} className="text-red-400 font-bold ml-2 hover:text-white">✕</button>
        </div>
      )}

      {/* ── Main Video Stage: Adaptive Split (Desktop) vs Hero + PiP (Mobile) ── */}
      {!isMobile ? (
        /* DESKTOP VIEW (md: and up) */
        <div className="flex flex-shrink-0 gap-4 px-5 pt-3" style={{ height: '52%', minHeight: '250px', maxHeight: '420px' }}>
          {/* Left: AI Avatar (Real-time 3D Avatar) */}
          <div className="flex-1 relative">
            <VisualChatAvatar
              status={status}
              isPip={false}
              onLoaded={handleAvatarLoaded}
              onToggleFullMode={() => setIsFullMode(true)}
            />
          </div>

          {/* Center: Start/Stop button */}
          <div className="flex-shrink-0 flex flex-col items-center justify-center gap-3">
            <button
              id="visual-chat-toggle"
              onClick={handleToggle}
              className={`
                relative w-14 h-14 rounded-full flex items-center justify-center
                transition-all duration-300 cursor-pointer shadow-lg
                ${isActive
                  ? 'bg-red-500/20 border-2 border-red-500/60 hover:bg-red-500/30 shadow-red-500/20 scale-105'
                  : 'bg-gradient-to-br from-accent-blue/20 to-accent-purple/20 border-2 border-accent-blue/40 hover:border-accent-blue/70 hover:scale-105 shadow-accent-blue/20'
                }
              `}
              aria-label={isActive ? 'Stop conversation' : 'Start conversation'}
            >
              {isActive ? (
                /* Stop icon */
                <div className="w-5 h-5 rounded bg-red-400" />
              ) : (
                /* Play / mic icon */
                <svg className="w-6 h-6 text-accent-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round"
                    d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z"
                  />
                </svg>
              )}

              {isActive && (
                <span className="absolute inset-0 rounded-full border-2 border-red-400/30 animate-ping" />
              )}
            </button>

            <span className="text-[10px] text-white/30 font-medium text-center max-w-[72px] leading-tight">
              {isActive ? 'End Call' : 'Start Call'}
            </span>
          </div>

          {/* Right: Webcam */}
          <div className="flex-1 relative">
            <WebcamPanel stream={webcamStream} isActive={isActive} isPip={false} />
          </div>
        </div>
      ) : (
        /* MOBILE VIEW (< md) — FaceTime / Video Call Hero Layout with persistent elements */
        <div
          className="flex-shrink-0 px-3 relative"
          style={{ height: '70vh', minHeight: '440px', marginTop: '1%' }}
        >
          <div className="relative w-full h-full rounded-3xl overflow-hidden border border-white/[0.12] shadow-2xl bg-navy-950/80">

            {/* Avatar container (Main by default) */}
            <div
              className={`transition-all duration-300 ${
                !swappedViews
                  ? 'absolute inset-0 z-10 w-full h-full'
                  : 'absolute bottom-3 right-3 z-30 w-24 h-32 sm:w-28 sm:h-36 rounded-2xl overflow-hidden shadow-2xl border-2 border-white/30 backdrop-blur-md active:scale-95 cursor-pointer hover:border-accent-cyan/70 hover:scale-105'
              }`}
              onClick={swappedViews ? () => setSwappedViews(false) : undefined}
              title={swappedViews ? 'Tap to make AI avatar main view' : undefined}
            >
              <div className="w-full h-full">
                <VisualChatAvatar
                  status={status}
                  isPip={swappedViews}
                  onLoaded={handleAvatarLoaded}
                  onToggleFullMode={() => setIsFullMode(true)}
                  onToggleEmotions={handleBrainClick}
                  showEmotions={showEmotionButton}
                />
              </div>
              {swappedViews && (
                <div className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded-md bg-black/75 text-[8px] text-white/70 flex items-center gap-0.5 font-medium pointer-events-none">
                  <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
                  </svg>
                  <span>Swap</span>
                </div>
              )}
            </div>

            {/* Webcam container (Small corner PiP by default, video call layout) */}
            <div
              className={`transition-all duration-300 ${
                swappedViews
                  ? 'absolute inset-0 z-10 w-full h-full'
                  : 'absolute bottom-3 right-3 z-30 w-24 h-32 sm:w-28 sm:h-36 rounded-2xl overflow-hidden shadow-2xl border-2 border-white/30 backdrop-blur-md active:scale-95 cursor-pointer hover:border-accent-cyan/70 hover:scale-105'
              }`}
              onClick={!swappedViews ? () => setSwappedViews(true) : undefined}
              title={!swappedViews ? 'Tap to make camera main view' : undefined}
            >
              <WebcamPanel
                stream={webcamStream}
                isActive={isActive}
                isPip={!swappedViews}
                className={swappedViews ? 'rounded-none border-0' : 'rounded-2xl'}
              />
              {!swappedViews && (
                <div className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded-md bg-black/75 text-[8px] text-white/70 flex items-center gap-0.5 font-medium pointer-events-none">
                  <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
                  </svg>
                  <span>Swap</span>
                </div>
              )}
            </div>

            {/* Top Bar for Webcam when swappedViews is true (webcam is main view) */}
            {swappedViews && (
              <>
                <button
                  onClick={handleBrainClick}
                  className={`absolute top-3 left-3 z-40 p-2 rounded-xl backdrop-blur-md border transition-all flex items-center gap-1.5 cursor-pointer shadow-lg active:scale-95 ${
                    showEmotionButton
                      ? 'bg-accent-blue/50 border-accent-blue/80 text-white shadow-accent-blue/30'
                      : 'bg-black/60 hover:bg-black/80 border-white/20 text-white/80'
                  }`}
                  title="Emotions"
                >
                  <span className="text-sm">🧠</span>
                </button>

                <button
                  onClick={() => setIsFullMode(true)}
                  className="absolute top-3 right-3 z-40 px-2.5 py-1.5 rounded-xl bg-black/60 hover:bg-black/80 backdrop-blur-md border border-white/20 text-white/90 hover:text-white transition-all flex items-center gap-1.5 cursor-pointer shadow-lg active:scale-95"
                  title="Full View"
                >
                  <svg className="w-3.5 h-3.5 text-accent-cyan" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
                  </svg>
                  <span className="text-[11px] font-bold">Full View</span>
                </button>
              </>
            )}

            {/* When brain icon is clicked, show a button labeled "Show Emotions" */}
            {showEmotionButton && (
              <div className="absolute top-14 left-3 right-3 z-40 flex flex-col items-center gap-2 animate-fade-in">
                <button
                  onClick={() => setIsEmotionCardExpanded(!isEmotionCardExpanded)}
                  className="px-4 py-2 rounded-2xl bg-black/85 hover:bg-black border border-accent-cyan/50 text-white font-semibold text-xs flex items-center gap-2 shadow-[0_4px_24px_rgba(6,182,212,0.35)] backdrop-blur-2xl transition-all active:scale-95 cursor-pointer"
                >
                  <span className="text-sm">🧠</span>
                  <span className="font-bold tracking-wide">{isEmotionCardExpanded ? 'Hide Emotions' : 'Show Emotions'}</span>
                  <span className="text-[10px] text-accent-cyan">{isEmotionCardExpanded ? '▲' : '▼'}</span>
                </button>

                {isEmotionCardExpanded && (
                  <div className="w-full max-h-[46vh] overflow-y-auto rounded-2xl shadow-2xl animate-scale-in">
                    <EmotionModalitiesPanel
                      multimodalEmotions={multimodalEmotions}
                      isProcessing={status === 'processing'}
                      compact={true}
                    />
                  </div>
                )}
              </div>
            )}

            {/* Floating Mobile Call Action Pill Button */}
            <div className="absolute bottom-3 left-3 z-40 flex items-center gap-2">
              <button
                id="visual-chat-toggle-mobile"
                onClick={handleToggle}
                className={`
                  px-3.5 py-2 rounded-full flex items-center gap-2 shadow-xl backdrop-blur-md cursor-pointer transition-all active:scale-95
                  ${isActive
                    ? 'bg-red-500/90 hover:bg-red-600 text-white shadow-red-500/40 border border-red-400/50'
                    : 'bg-gradient-to-r from-accent-blue to-accent-purple hover:brightness-110 text-white shadow-accent-blue/30 border border-white/20'
                  }
                `}
                aria-label={isActive ? 'End Conversation' : 'Start Conversation'}
              >
                {isActive ? (
                  <>
                    <span className="w-2.5 h-2.5 rounded-sm bg-white" />
                    <span className="text-xs font-bold tracking-wide">End Call</span>
                  </>
                ) : (
                  <>
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round"
                        d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z"
                      />
                    </svg>
                    <span className="text-xs font-bold tracking-wide">Start Call</span>
                  </>
                )}
              </button>
            </div>

          </div>
        </div>
      )}

      {/* ── 4-Modality Emotion Inspector Panel (Desktop only; on mobile accessed via "Show Emotions") ── */}
      {!isMobile && (
        <EmotionModalitiesPanel
          multimodalEmotions={multimodalEmotions}
          isProcessing={status === 'processing'}
        />
      )}

      {/* ── Live transcript strip ── */}
      <div className="flex-shrink-0 mx-3 sm:mx-5 mt-2 px-3 sm:px-4 py-2 rounded-xl bg-white/[0.02] border border-white/[0.05] min-h-[36px] flex items-center">
        {transcript ? (
          <p className="text-xs sm:text-sm text-white/80 italic animate-fade-in truncate">
            <span className="text-white/30 mr-1.5 not-italic font-medium text-[9px] sm:text-[10px] uppercase tracking-wider">You said:</span>
            {transcript}
          </p>
        ) : (
          <p className="text-[11px] sm:text-xs text-white/25 truncate">
            {isActive ? 'Speak naturally — VAD evaluates your voice & face…' : 'Tap Start Call to begin conversation'}
          </p>
        )}
      </div>

      {/* ── Conversation log ── */}
      <div
        className="flex-1 min-h-[180px] md:min-h-0 overflow-y-auto md:overflow-y-auto px-3 sm:px-5 py-2 space-y-2 scroll-smooth"
        style={{ scrollBehavior: 'smooth' }}
      >
        {conversation.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center animate-fade-in py-4">
            <div className="w-10 h-10 sm:w-12 sm:h-12 mb-2 rounded-2xl bg-gradient-to-br from-accent-blue/10 to-accent-purple/10 border border-white/5 flex items-center justify-center">
              <svg className="w-5 h-5 sm:w-6 sm:h-6 text-accent-blue/40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z"
                />
              </svg>
            </div>
            <p className="text-xs sm:text-sm text-white/35 font-medium">Conversation log will appear here</p>
            <p className="text-[10px] sm:text-xs text-white/20 mt-0.5">RoBERTa, Wav2Vec2, FER+ & DSMR evaluate each turn</p>
          </div>
        )}

        {conversation.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-2 animate-fade-in ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {msg.role === 'assistant' && (
              <div className="w-6 h-6 sm:w-7 sm:h-7 flex-shrink-0 rounded-full bg-gradient-to-br from-accent-blue/20 to-accent-purple/20 border border-white/10 flex items-center justify-center mt-0.5">
                <span className="text-[9px] sm:text-[10px] text-accent-blue font-bold">AI</span>
              </div>
            )}
            <div
              className={`
                max-w-[85%] sm:max-w-[75%] px-3 sm:px-3.5 py-2 sm:py-2.5 rounded-2xl text-xs sm:text-sm leading-relaxed
                ${msg.role === 'user'
                  ? 'bg-accent-blue/10 border border-accent-blue/20 text-white/90 rounded-br-sm'
                  : 'bg-white/[0.04] border border-white/[0.06] text-white/80 rounded-bl-sm'
                }
              `}
            >
              {/* User 4-modality emotion breakdown */}
              {msg.role === 'user' && msg.multimodalEmotions && (
                <div className="flex flex-wrap gap-1 justify-end mb-1.5">
                  {msg.multimodalEmotions.text?.emotion && (
                    <span className="text-[8px] sm:text-[9px] px-1.5 py-0.5 rounded bg-blue-500/10 border border-blue-500/20 text-blue-300">
                      Text: {msg.multimodalEmotions.text.emotion} ({Math.round((msg.multimodalEmotions.text.confidence || 0) * 100)}%)
                    </span>
                  )}
                  {msg.multimodalEmotions.audio?.emotion && (
                    <span className="text-[8px] sm:text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">
                      Voice: {msg.multimodalEmotions.audio.emotion} ({Math.round((msg.multimodalEmotions.audio.confidence || 0) * 100)}%)
                    </span>
                  )}
                  {msg.multimodalEmotions.face?.emotion && (
                    <span className="text-[8px] sm:text-[9px] px-1.5 py-0.5 rounded bg-purple-500/10 border border-purple-500/20 text-purple-300">
                      Face: {msg.multimodalEmotions.face.face_detected === false ? 'None' : msg.multimodalEmotions.face.emotion}
                    </span>
                  )}
                  {msg.multimodalEmotions.dsmr?.emotion && (
                    <span className="text-[8px] sm:text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/20 text-amber-300 font-semibold">
                      DSMR: {msg.multimodalEmotions.dsmr.emotion}
                    </span>
                  )}
                  {msg.multimodalEmotions.fused?.confidence > 0 && (
                    <span className="text-[8px] sm:text-[9px] px-1.5 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 font-semibold">
                      Overall: {msg.multimodalEmotions.fused.emotion} ({Math.round(msg.multimodalEmotions.fused.confidence * 100)}%)
                    </span>
                  )}
                </div>
              )}

              {/* Assistant message with AI response emotion */}
              {msg.role === 'assistant' && (
                <div>
                  {msg.emotion && (
                    <div className="flex items-center gap-1.5 mb-1.5">
                      <span className="inline-flex items-center gap-1 text-[9px] sm:text-[10px] text-accent-cyan font-bold uppercase tracking-wider bg-accent-cyan/15 px-2 py-0.5 rounded-full border border-accent-cyan/30">
                        <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse" />
                        AI Emotion: {msg.emotion}
                      </span>
                    </div>
                  )}
                  <p className="text-white/90 leading-relaxed">{msg.text}</p>
                </div>
              )}

              {msg.role === 'user' && msg.text}
            </div>
            {msg.role === 'user' && (
              <div className="w-6 h-6 sm:w-7 sm:h-7 flex-shrink-0 rounded-full bg-white/[0.06] border border-white/10 flex items-center justify-center mt-0.5">
                <svg className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-white/40" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z" />
                </svg>
              </div>
            )}
          </div>
        ))}

        {/* Generating Answer loading card in standard conversation log */}
        {status === 'processing' && (
          <div className="flex gap-2 animate-fade-in justify-start">
            <div className="w-6 h-6 sm:w-7 sm:h-7 flex-shrink-0 rounded-full bg-gradient-to-br from-accent-blue/20 to-accent-purple/20 border border-white/10 flex items-center justify-center mt-0.5">
              <span className="text-[9px] sm:text-[10px] text-accent-cyan font-bold">AI</span>
            </div>
            <div className="max-w-[85%] sm:max-w-[75%] px-3.5 py-2.5 rounded-2xl bg-white/[0.04] border border-accent-cyan/30 text-white/80 rounded-bl-sm flex flex-col gap-2 min-w-[200px] shadow-lg shadow-accent-blue/10">
              <div className="flex items-center justify-between">
                <span className="inline-flex items-center gap-1.5 text-[10px] text-accent-cyan font-bold uppercase tracking-wider">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-ping" />
                  Generating answer…
                </span>
                <span className="text-[9px] font-mono text-white/40">Qwen 27B</span>
              </div>
              <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden relative border border-white/10">
                <div className="h-full w-full bg-gradient-to-r from-accent-cyan via-accent-blue to-accent-purple animate-loading-sweep rounded-full shadow-[0_0_8px_rgba(6,182,212,0.6)]" />
              </div>
            </div>
          </div>
        )}
        <div ref={conversationEndRef} />
      </div>
    </div>
  );
}
