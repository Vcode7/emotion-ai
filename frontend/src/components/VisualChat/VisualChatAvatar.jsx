import { Suspense, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { Environment } from '@react-three/drei';
import SkyAvatar from './SkyAvatar';
import useAppStore from '../../store/useAppStore';
import * as THREE from 'three';
import { normalizeEmotion } from '../../utils/emotionLabels';

const EMOTION_COLORS = {
  neutral:     { ring: '#22d3ee', glow: 'rgba(34,211,238,0.25)',  label: 'Neutral' },
  happy:       { ring: '#f59e0b', glow: 'rgba(245,158,11,0.3)',   label: 'Happy' },
  sad:         { ring: '#60a5fa', glow: 'rgba(96,165,250,0.25)',  label: 'Sad' },
  angry:       { ring: '#f87171', glow: 'rgba(248,113,113,0.28)', label: 'Angry' },
  surprised:   { ring: '#e879f9', glow: 'rgba(232,121,249,0.28)', label: 'Surprised' },
  fear:        { ring: '#818cf8', glow: 'rgba(129,140,248,0.28)', label: 'Fear' },
  fearful:     { ring: '#818cf8', glow: 'rgba(129,140,248,0.28)', label: 'Fear' },
  disgust:     { ring: '#34d399', glow: 'rgba(52,211,153,0.28)',  label: 'Disgust' },
  concerned:   { ring: '#fb923c', glow: 'rgba(251,146,60,0.28)',  label: 'Concerned' },
  thinking:    { ring: '#facc15', glow: 'rgba(250,204,21,0.25)',  label: 'Thinking' },
  empathetic:  { ring: '#a78bfa', glow: 'rgba(167,139,250,0.28)', label: 'Empathetic' },
  compassion:  { ring: '#f472b6', glow: 'rgba(244,114,182,0.28)', label: 'Compassion' },
  supportive:  { ring: '#4ade80', glow: 'rgba(74,222,128,0.28)',  label: 'Supportive' },
  encouraging: { ring: '#f59e0b', glow: 'rgba(245,158,11,0.3)',   label: 'Encouraging' },
};

function getEmoColors(emotion) {
  return EMOTION_COLORS[normalizeEmotion(emotion, EMOTION_COLORS)] || EMOTION_COLORS.neutral;
}

function WaveformBars({ color }) {
  return (
    <div className="flex items-end gap-[3px] h-4">
      {[0,1,2,3,4,5].map((i) => (
        <div
          key={i}
          className="rounded-full animate-bounce"
          style={{
            width: '2.5px',
            backgroundColor: color,
            height: `${5 + (i % 3) * 5}px`,
            animationDelay: `${i * 0.09}s`,
            animationDuration: '0.6s',
          }}
        />
      ))}
    </div>
  );
}

function Avatar3DLoading({ label = 'Loading Avatar…' }) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center bg-navy-950/80 backdrop-blur-sm z-10">
      <div className="w-10 h-10 rounded-full border-2 border-accent-blue/30 border-t-accent-blue animate-spin mb-3" />
      <span className="text-[11px] font-semibold text-white/60 uppercase tracking-widest">{label}</span>
    </div>
  );
}

/**
 * StudioLighting — 4-point studio rig tuned for human skin rendering.
 */
function StudioLighting() {
  return (
    <>
      <Environment preset="sunset" background={false} environmentIntensity={0.35} />

      {/* Main key light — reduced intensity for natural portrait shading without blowout */}
      <directionalLight
        position={[-1.4, 2.0, 2.8]}
        intensity={0.95}
        color={new THREE.Color(1.0, 0.94, 0.88)}
        castShadow={false}
      />

      {/* Soft warm fill light */}
      <directionalLight
        position={[2.2, 0.6, 1.8]}
        intensity={0.30}
        color={new THREE.Color(0.88, 0.74, 0.62)}
      />

      {/* Subtle frontal portrait fill */}
      <directionalLight
        position={[0, -0.2, 3.8]}
        intensity={0.22}
        color={new THREE.Color(0.96, 0.96, 0.98)}
      />

      {/* Hair & shoulder rim separation light */}
      <directionalLight
        position={[0.2, 3.6, -1.2]}
        intensity={0.50}
        color={new THREE.Color(0.95, 0.90, 0.84)}
      />

      {/* Low bounce light */}
      <pointLight
        position={[0, -2.0, 1.4]}
        intensity={0.12}
        color={new THREE.Color(0.85, 0.68, 0.55)}
      />

      {/* Soft ambient light */}
      <ambientLight intensity={0.18} color={new THREE.Color(0.42, 0.40, 0.45)} />
    </>
  );
}

const AVATAR_OPTIONS = [
  { id: 'sky',  label: 'Sky',   tag: 'ARKit 66' },
  { id: 'sky2', label: 'Sky 2', tag: 'ARKit 66' },
];

export default function VisualChatAvatar({
  emotion: emotionProp,
  isSpeaking: isSpeakingProp,
  status = 'idle',
  isFullMode = false,
  isPip = false,
  onLoaded,
  onToggleFullMode,
  onToggleEmotions,
  showEmotions = false,
}) {
  const [showPicker, setShowPicker] = useState(false);
  const [showControls, setShowControls] = useState(false);

  // Model transform controls: pan (up/down/left/right), rotate, zoom
  const [modelOffset, setModelOffset] = useState({ x: 0, y: 0, z: 0 });
  const [modelRotationY, setModelRotationY] = useState(0);
  const [modelZoom, setModelZoom] = useState(1.0);

  const storeIsSpeaking   = useAppStore((s) => s.isSpeaking);
  const storeEmotion      = useAppStore((s) => s.currentEmotion);
  const storeIsProcessing = useAppStore((s) => s.isProcessing);
  const rawAvatar         = useAppStore((s) => s.selectedAvatar);
  const setSelectedAvatar = useAppStore((s) => s.setSelectedAvatar);

  // Strictly constrain avatar to 'sky' | 'sky2'
  const selectedAvatar = rawAvatar === 'sky2' ? 'sky2' : 'sky';

  const currentEmotion    = emotionProp ?? storeEmotion ?? 'neutral';
  const currentIsSpeaking = isSpeakingProp !== undefined
    ? isSpeakingProp
    : (status === 'speaking' || storeIsSpeaking);
  const isGenerating = (status === 'processing' || storeIsProcessing) && !currentIsSpeaking;
  const effectiveEmotion = isGenerating && currentEmotion === 'neutral' ? 'thinking' : currentEmotion;
  const colors = getEmoColors(effectiveEmotion);

  const activeAvatarName = selectedAvatar === 'sky2' ? 'Sky 2' : 'Sky';

  // Model control actions
  const handleMoveUp = () => setModelOffset((p) => ({ ...p, y: +(p.y + 0.15).toFixed(2) }));
  const handleMoveDown = () => setModelOffset((p) => ({ ...p, y: +(p.y - 0.15).toFixed(2) }));
  const handleMoveLeft = () => setModelOffset((p) => ({ ...p, x: +(p.x - 0.15).toFixed(2) }));
  const handleMoveRight = () => setModelOffset((p) => ({ ...p, x: +(p.x + 0.15).toFixed(2) }));
  const handleRotateLeft = () => setModelRotationY((r) => +(r - 0.15).toFixed(3));
  const handleRotateRight = () => setModelRotationY((r) => +(r + 0.15).toFixed(3));
  const handleZoomIn = () => setModelZoom((z) => Math.min(+(z + 0.15).toFixed(2), 3.0));
  const handleZoomOut = () => setModelZoom((z) => Math.max(+(z - 0.15).toFixed(2), 0.4));
  const handleResetPosition = () => setModelOffset({ x: 0, y: 0, z: 0 });
  const handleResetAll = () => {
    setModelOffset({ x: 0, y: 0, z: 0 });
    setModelRotationY(0);
    setModelZoom(1.0);
  };

  return (
    <div
      className={`relative w-full h-full overflow-hidden flex items-center justify-center select-none ${
        isFullMode ? 'rounded-none border-0 p-0' : 'rounded-2xl border border-white/[0.08] p-1 sm:p-2'
      }`}
      style={{
        background: isFullMode
          ? 'radial-gradient(circle at 48% 44%, #121622 0%, #090c14 55%, #030408 100%)'
          : 'radial-gradient(circle at 48% 42%, #111520 0%, #080a12 60%, #030408 100%)',
      }}
    >
      <div
        className={`relative w-full h-full overflow-hidden transition-all duration-700 flex items-center justify-center ${
          isFullMode ? 'rounded-none' : 'rounded-2xl'
        }`}
        style={{
          boxShadow: isFullMode
            ? `inset 0 0 70px ${colors.glow}, inset 0 -30px 50px rgba(0,0,0,0.85)`
            : `0 0 0 1.5px ${colors.ring}60, 0 0 28px 4px ${colors.glow}, 0 8px 32px rgba(0,0,0,0.7)`,
          transition: 'box-shadow 0.8s ease',
        }}
      >
        {/* Top-edge neon laser scanner bar during answer generation */}
        {isGenerating && (
          <div className="absolute top-0 left-0 right-0 h-[2.5px] sm:h-[3px] z-30 overflow-hidden bg-white/5 pointer-events-none">
            <div className="h-full bg-gradient-to-r from-transparent via-accent-cyan via-accent-blue to-transparent animate-loading-sweep shadow-[0_0_12px_rgba(6,182,212,0.9)]" />
          </div>
        )}

        <Suspense fallback={<Avatar3DLoading label={`Loading ${activeAvatarName}…`} />}>
          <Canvas
            camera={{ position: [0, 0, 4.4], fov: 32 }}
            dpr={[1, 2]}
            gl={{
              antialias: true,
              alpha: true,
              powerPreference: 'high-performance',
              toneMapping: THREE.ACESFilmicToneMapping,
              toneMappingExposure: 0.88,
            }}
            shadows={false}
            style={{ width: '100%', height: '100%', pointerEvents: 'none' }}
          >
            <StudioLighting />
            <SkyAvatar
              key={selectedAvatar}
              avatarType={selectedAvatar}
              modelPath={selectedAvatar === 'sky2' ? '/models/sky2.glb' : '/models/sky.glb'}
              emotion={effectiveEmotion}
              isSpeaking={currentIsSpeaking}
              status={status}
              modelOffset={modelOffset}
              modelRotationY={modelRotationY}
              modelZoom={modelZoom}
              onLoaded={onLoaded}
            />
          </Canvas>
        </Suspense>

        {/* ── Top Bar Controls Overlay (hidden in PiP mode) ── */}
        {!isPip && (
          <div
            className={`absolute ${
              isFullMode ? 'top-14 sm:top-16' : 'top-2 sm:top-3'
            } left-2 sm:left-3 right-2 sm:right-3 z-20 flex items-center justify-between pointer-events-none gap-1 sm:gap-2`}
          >
            {/* Top-Left: Brain Emotions button (if enabled) + Emotion Badge */}
            <div className="flex items-center gap-1 sm:gap-1.5 pointer-events-auto flex-shrink-0">
              {onToggleEmotions && (
                <button
                  id="avatar-emotions-btn"
                  onClick={onToggleEmotions}
                  className={`p-1.5 sm:px-2.5 sm:py-1 rounded-full backdrop-blur-md border transition-all flex items-center gap-1 cursor-pointer shadow-lg active:scale-95 flex-shrink-0 ${
                    showEmotions
                      ? 'bg-accent-blue/50 border-accent-blue/80 text-white shadow-accent-blue/30'
                      : 'bg-black/65 hover:bg-black/85 border-white/10 text-white/80 hover:text-white'
                  }`}
                  title="Emotions Breakdown"
                >
                  <span className="text-xs">🧠</span>
                  <span className="hidden sm:inline text-[10px] font-semibold">Emotions</span>
                </button>
              )}

              {/* Emotion Badge + Preview Picker Dropdown */}
              <div className="relative flex items-center gap-1 p-0.5 sm:p-1 rounded-full bg-black/65 backdrop-blur-md border border-white/10 shadow-lg flex-shrink-0">
                <button
                  onClick={() => setShowPicker((p) => !p)}
                  className="flex items-center gap-1 sm:gap-1.5 px-2 py-0.5 rounded-full hover:bg-white/10 transition-colors cursor-pointer"
                  title="Preview emotions"
                >
                  <div className="w-2 h-2 rounded-full animate-pulse flex-shrink-0" style={{ backgroundColor: colors.ring }} />
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-white/90">{colors.label}</span>
                  <span className="text-[8px] text-white/40 ml-0.5">▼</span>
                </button>
                {showPicker && (
                  <div className="absolute top-full left-0 mt-1.5 p-1.5 rounded-2xl bg-black/90 backdrop-blur-xl border border-white/15 shadow-2xl flex flex-wrap gap-1 z-40 animate-fade-in w-56 sm:w-64">
                    {['neutral','happy','sad','angry','surprised','fear','disgust'].map((emo) => (
                      <button
                        key={emo}
                        onClick={() => {
                          useAppStore.getState().setCurrentEmotion(emo);
                          setShowPicker(false);
                        }}
                        className={`px-2 py-0.5 text-[9px] font-semibold rounded-full uppercase tracking-wider transition-all cursor-pointer ${
                          currentEmotion.toLowerCase() === emo
                            ? 'bg-accent-blue/40 border border-accent-blue/60 text-white shadow-sm'
                            : 'text-white/60 hover:text-white hover:bg-white/10'
                        }`}
                      >
                        {emo}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Top-Right: Controls Button + Avatar Switcher (Sky vs Sky2) + Full View Toggle */}
            <div className="flex items-center gap-1 sm:gap-1.5 pointer-events-auto flex-shrink-0">
              {/* Controls Button: opens/expands panel with up/down/left/right/rotate/zoom */}
              <button
                id="model-controls-btn"
                onClick={() => setShowControls((v) => !v)}
                className={`flex items-center gap-1 sm:gap-1.5 px-2 sm:px-2.5 py-1 rounded-full backdrop-blur-md border text-[10px] sm:text-[11px] font-semibold tracking-wide transition-all shadow-lg active:scale-95 cursor-pointer flex-shrink-0 ${
                  showControls
                    ? 'bg-accent-cyan/25 border-accent-cyan text-white shadow-[0_0_12px_rgba(6,182,212,0.4)]'
                    : 'bg-black/65 hover:bg-black/85 border-white/10 text-white/80 hover:text-white'
                }`}
                title="Toggle Model Controls (Pan, Rotate, Zoom)"
              >
                <svg className="w-3.5 h-3.5 text-accent-cyan flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
                </svg>
                <span className="hidden sm:inline">Controls</span>
                <span className="text-[8px] text-white/50">{showControls ? '▲' : '▼'}</span>
              </button>

              {/* Avatar Switcher: Sky & Sky 2 */}
              <div className="flex items-center p-0.5 sm:p-1 rounded-full bg-black/65 backdrop-blur-md border border-white/10 shadow-lg flex-shrink-0">
                <div className="flex items-center gap-0.5 sm:gap-1">
                  {AVATAR_OPTIONS.map((av) => {
                    const isSelected = selectedAvatar === av.id;
                    return (
                      <button
                        key={av.id}
                        id={`avatar-btn-${av.id}`}
                        onClick={() => setSelectedAvatar(av.id)}
                        className={`px-2 sm:px-2.5 py-0.5 rounded-full text-[10px] sm:text-[11px] font-semibold tracking-wide transition-all cursor-pointer whitespace-nowrap ${
                          isSelected
                            ? 'bg-gradient-to-r from-accent-cyan/80 to-accent-blue/80 text-white shadow-md shadow-accent-cyan/20 scale-[1.02]'
                            : 'text-white/60 hover:text-white hover:bg-white/10'
                        }`}
                        title={`Switch avatar to ${av.label}`}
                      >
                        <span>{av.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Full View Button (only shown in normal view mode) */}
              {onToggleFullMode && !isFullMode && (
                <button
                  id="toggle-full-view-btn"
                  onClick={onToggleFullMode}
                  className="p-1 sm:px-2.5 sm:py-1 rounded-full bg-black/65 hover:bg-black/85 backdrop-blur-md border border-white/10 hover:border-accent-cyan/40 text-white/80 hover:text-white text-[10px] sm:text-[11px] font-semibold flex items-center gap-1 shadow-lg transition-all active:scale-95 cursor-pointer flex-shrink-0"
                  title="Expand to Full View"
                >
                  <svg className="w-3.5 h-3.5 text-accent-cyan flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
                  </svg>
                  <span className="hidden sm:inline">Full</span>
                </button>
              )}
            </div>
          </div>
        )}

        {/* ── Expandable Model Controls Panel ── */}
        {!isPip && showControls && (
          <div
            id="model-controls-panel"
            className={`absolute ${
              isFullMode ? 'top-24 sm:top-28' : 'top-12 sm:top-14'
            } right-2 sm:right-3 z-30 p-2.5 sm:p-3 rounded-2xl bg-black/85 backdrop-blur-xl border border-accent-cyan/30 shadow-[0_8px_32px_rgba(0,0,0,0.8),0_0_20px_rgba(6,182,212,0.15)] flex flex-col gap-2 sm:gap-2.5 animate-scale-in text-white select-none w-48 sm:w-56 max-w-[calc(100vw-1.5rem)]`}
          >
            {/* Panel Header */}
            <div className="flex items-center justify-between border-b border-white/10 pb-1.5">
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse" />
                <span className="text-[10px] font-bold uppercase tracking-wider text-accent-cyan">Model Controls</span>
              </div>
              <button
                id="btn-close-controls"
                onClick={() => setShowControls(false)}
                className="w-5 h-5 rounded-full hover:bg-white/10 flex items-center justify-center text-[11px] text-white/50 hover:text-white transition-colors cursor-pointer"
                title="Close controls panel"
              >
                ✕
              </button>
            </div>

            {/* Position / Pan Pad: Up, Down, Left, Right */}
            <div className="flex flex-col items-center gap-1">
              <span className="text-[9px] uppercase tracking-wider text-white/40 font-mono font-medium">Position</span>
              <div className="grid grid-cols-3 gap-1 w-28">
                <div />
                <button
                  id="btn-control-up"
                  onClick={handleMoveUp}
                  className="h-8 rounded-lg bg-white/10 hover:bg-white/20 active:bg-accent-blue/40 border border-white/10 hover:border-accent-cyan/40 flex items-center justify-center text-xs font-bold text-white transition-all cursor-pointer shadow-sm active:scale-95"
                  title="Move Model Up"
                >
                  ▲
                </button>
                <div />

                <button
                  id="btn-control-left"
                  onClick={handleMoveLeft}
                  className="h-8 rounded-lg bg-white/10 hover:bg-white/20 active:bg-accent-blue/40 border border-white/10 hover:border-accent-cyan/40 flex items-center justify-center text-xs font-bold text-white transition-all cursor-pointer shadow-sm active:scale-95"
                  title="Move Model Left"
                >
                  ◀
                </button>
                <button
                  id="btn-control-reset-center"
                  onClick={handleResetPosition}
                  className="h-8 rounded-lg bg-white/5 hover:bg-white/15 border border-white/5 flex items-center justify-center text-[9px] font-mono text-white/60 hover:text-white transition-all cursor-pointer"
                  title="Center Position"
                >
                  ●
                </button>
                <button
                  id="btn-control-right"
                  onClick={handleMoveRight}
                  className="h-8 rounded-lg bg-white/10 hover:bg-white/20 active:bg-accent-blue/40 border border-white/10 hover:border-accent-cyan/40 flex items-center justify-center text-xs font-bold text-white transition-all cursor-pointer shadow-sm active:scale-95"
                  title="Move Model Right"
                >
                  ▶
                </button>

                <div />
                <button
                  id="btn-control-down"
                  onClick={handleMoveDown}
                  className="h-8 rounded-lg bg-white/10 hover:bg-white/20 active:bg-accent-blue/40 border border-white/10 hover:border-accent-cyan/40 flex items-center justify-center text-xs font-bold text-white transition-all cursor-pointer shadow-sm active:scale-95"
                  title="Move Model Down"
                >
                  ▼
                </button>
                <div />
              </div>
            </div>

            {/* Rotation: Rotate Left & Rotate Right */}
            <div className="flex flex-col gap-1 border-t border-white/10 pt-2">
              <div className="flex items-center justify-between text-[9px] uppercase tracking-wider text-white/40 font-mono font-medium">
                <span>Rotation</span>
                <span className="text-accent-cyan/80">{Math.round((modelRotationY * 180) / Math.PI)}°</span>
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  id="btn-control-rotate-left"
                  onClick={handleRotateLeft}
                  className="h-7 px-2 rounded-lg bg-white/10 hover:bg-white/20 active:bg-accent-blue/40 border border-white/10 hover:border-accent-cyan/40 flex items-center justify-center gap-1 text-[11px] font-semibold text-white transition-all cursor-pointer shadow-sm active:scale-95"
                  title="Rotate Left"
                >
                  <span>↺</span>
                  <span className="text-[10px]">Left</span>
                </button>
                <button
                  id="btn-control-rotate-right"
                  onClick={handleRotateRight}
                  className="h-7 px-2 rounded-lg bg-white/10 hover:bg-white/20 active:bg-accent-blue/40 border border-white/10 hover:border-accent-cyan/40 flex items-center justify-center gap-1 text-[11px] font-semibold text-white transition-all cursor-pointer shadow-sm active:scale-95"
                  title="Rotate Right"
                >
                  <span className="text-[10px]">Right</span>
                  <span>↻</span>
                </button>
              </div>
            </div>

            {/* Zoom: Zoom Out & Zoom In */}
            <div className="flex flex-col gap-1 border-t border-white/10 pt-2">
              <div className="flex items-center justify-between text-[9px] uppercase tracking-wider text-white/40 font-mono font-medium">
                <span>Zoom</span>
                <span className="text-accent-cyan/80">{Math.round(modelZoom * 100)}%</span>
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  id="btn-control-zoom-out"
                  onClick={handleZoomOut}
                  className="h-7 px-2 rounded-lg bg-white/10 hover:bg-white/20 active:bg-accent-blue/40 border border-white/10 hover:border-accent-cyan/40 flex items-center justify-center gap-1 text-[11px] font-semibold text-white transition-all cursor-pointer shadow-sm active:scale-95"
                  title="Zoom Out"
                >
                  <span className="text-sm font-bold leading-none">−</span>
                  <span className="text-[10px]">Out</span>
                </button>
                <button
                  id="btn-control-zoom-in"
                  onClick={handleZoomIn}
                  className="h-7 px-2 rounded-lg bg-white/10 hover:bg-white/20 active:bg-accent-blue/40 border border-white/10 hover:border-accent-cyan/40 flex items-center justify-center gap-1 text-[11px] font-semibold text-white transition-all cursor-pointer shadow-sm active:scale-95"
                  title="Zoom In"
                >
                  <span className="text-sm font-bold leading-none">+</span>
                  <span className="text-[10px]">In</span>
                </button>
              </div>
            </div>

            {/* Reset All */}
            <button
              id="btn-control-reset-all"
              onClick={handleResetAll}
              className="mt-1 h-6 rounded-lg bg-white/5 hover:bg-white/15 border border-white/10 text-[10px] font-mono text-white/60 hover:text-white flex items-center justify-center gap-1 transition-all cursor-pointer active:scale-95"
              title="Reset all model controls"
            >
              <span>⟲</span>
              <span>Reset View</span>
            </button>
          </div>
        )}

        {/* Speaking / Listening / Generating Answer Badge & Loading Bar */}
        {!isPip && (
          <div
            className={`absolute ${
              isFullMode ? (isGenerating ? 'bottom-40' : 'bottom-3') : 'bottom-3'
            } left-1/2 -translate-x-1/2 z-20 pointer-events-none transition-all duration-300 max-w-[92%] sm:max-w-md w-auto`}
          >
            {currentIsSpeaking && (
              <div
                className="flex items-center gap-2 px-3 py-1 rounded-full border backdrop-blur-md shadow-lg mx-auto w-fit"
                style={{ backgroundColor: `${colors.ring}20`, borderColor: `${colors.ring}60` }}
              >
                <WaveformBars color={colors.ring} />
                <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: colors.ring }}>Speaking</span>
              </div>
            )}
            {!currentIsSpeaking && status === 'listening' && (
              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/15 border border-emerald-500/40 backdrop-blur-md shadow-lg mx-auto w-fit">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-400">Listening</span>
              </div>
            )}
            {!currentIsSpeaking && isGenerating && (
              <div className="flex flex-col gap-2 p-3 sm:p-3.5 rounded-2xl bg-black/80 backdrop-blur-2xl border border-accent-cyan/40 shadow-[0_0_30px_rgba(6,182,212,0.25)] min-w-[240px] sm:min-w-[290px] animate-fade-in pointer-events-auto">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-2.5 w-2.5">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-cyan opacity-75" />
                      <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-accent-cyan" />
                    </span>
                    <span className="text-xs sm:text-sm font-bold tracking-wide text-white drop-shadow">
                      Generating answer…
                    </span>
                  </div>
                  <span className="text-[9px] font-mono font-semibold uppercase tracking-wider text-accent-cyan bg-accent-cyan/15 px-2 py-0.5 rounded-full border border-accent-cyan/30 animate-pulse">
                    AI Thinking
                  </span>
                </div>

                <div className="relative w-full h-2 sm:h-2.5 bg-white/10 rounded-full overflow-hidden border border-white/15 p-[1px]">
                  <div className="h-full w-full rounded-full bg-gradient-to-r from-accent-cyan via-accent-blue to-accent-purple animate-gradient-flow" />
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/80 to-transparent animate-loading-sweep" />
                </div>

                <div className="flex items-center justify-between text-[9px] sm:text-[10px] text-white/50 font-mono">
                  <span className="flex items-center gap-1.5 truncate">
                    <span className="w-1.5 h-1.5 rounded-full bg-accent-blue animate-pulse" />
                    Multimodal Emotion & Reasoning
                  </span>
                  <span className="text-accent-cyan/90 font-semibold flex-shrink-0 ml-1">
                    Qwen 27B
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Minimal indicator in PiP mode when speaking */}
        {isPip && currentIsSpeaking && (
          <div
            className="absolute bottom-1.5 left-1/2 -translate-x-1/2 z-20 px-2 py-0.5 rounded-full border backdrop-blur-md shadow-md pointer-events-none"
            style={{ backgroundColor: `${colors.ring}30`, borderColor: `${colors.ring}80` }}
          >
            <span className="text-[8px] font-bold uppercase tracking-wider text-white">AI Speaking</span>
          </div>
        )}

        {/* Watermark */}
        {!isPip && (
          <div className="absolute bottom-1.5 right-2.5 z-20 pointer-events-none opacity-40 hover:opacity-80 transition-opacity">
            <span className="text-[8px] font-mono tracking-wider text-white uppercase">
              {activeAvatarName} · ARKit 66
            </span>
          </div>
        )}
      </div>
    </div>
  );
}