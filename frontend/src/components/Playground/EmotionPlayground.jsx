import { useState, useRef, useEffect, useCallback, Suspense, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import RobotAvatar from '../Avatar/RobotAvatar';
import { synthesizeSpeech } from '../../api/client';
import useAppStore from '../../store/useAppStore';

const PRESET_PHRASES = [
  "Hello there! I am your conversational assistant, ready to assist you in any emotional register.",
  "I am absolutely thrilled to share this project with you! It is going to change everything!",
  "Oh no... I am really sorry to hear that. That sounds incredibly difficult and sad.",
  "What? Are you serious? I cannot believe this is actually happening right now!",
  "Please do not worry. Take a deep breath, and let's work through this together step by step.",
  "I'm feeling a bit confused. Let me think about how we can resolve this problem."
];

const EMOTIONS = [
  {
    id: 'neutral',
    name: 'Neutral',
    color: '#06b6d4',
    bgClass: 'border-accent-cyan/20 text-accent-cyan hover:bg-accent-cyan/5',
    activeBgClass: 'bg-accent-cyan/10 border-accent-cyan/50 text-accent-cyan glow-cyan',
    accentColor: '#06b6d4',
    description: 'Standard, conversational tone. Balanced features, cyan lighting.',
    voiceParams: { speed: 1.0, pitch: 1.0 }
  },
  {
    id: 'happy',
    name: 'Happy',
    color: '#f59e0b',
    bgClass: 'border-accent-amber/20 text-accent-amber hover:bg-accent-amber/5',
    activeBgClass: 'bg-accent-amber/10 border-accent-amber/50 text-accent-amber glow-amber',
    accentColor: '#f59e0b',
    description: 'Upbeat, high energy voice. Happy squint eyes, broad smile.',
    voiceParams: { speed: 1.15, pitch: 1.15 }
  },
  {
    id: 'sad',
    name: 'Sad',
    color: '#3b82f6',
    bgClass: 'border-accent-blue/20 text-accent-blue hover:bg-accent-blue/5',
    activeBgClass: 'bg-accent-blue/10 border-accent-blue/50 text-accent-blue glow-blue',
    accentColor: '#3b82f6',
    description: 'Slow, low-pitched voice. Drooped eyes and mouth curve.',
    voiceParams: { speed: 0.75, pitch: 0.75 }
  },
  {
    id: 'angry',
    name: 'Angry',
    color: '#ef4444',
    bgClass: 'border-accent-red/20 text-accent-red hover:bg-accent-red/5',
    activeBgClass: 'bg-accent-red/10 border-accent-red/50 text-accent-red glow-red',
    accentColor: '#ef4444',
    description: 'Fast, sharp voice. Flat mouth, red illumination.',
    voiceParams: { speed: 1.25, pitch: 0.80 }
  },
  {
    id: 'surprised',
    name: 'Surprised',
    color: '#ec4899',
    bgClass: 'border-accent-pink/20 text-accent-pink hover:bg-accent-pink/5',
    activeBgClass: 'bg-accent-pink/10 border-accent-pink/50 text-accent-pink glow-pink',
    accentColor: '#ec4899',
    description: 'Very fast, high-pitched voice. Wide open eyes, deep mouth drop.',
    voiceParams: { speed: 1.30, pitch: 1.25 }
  },
  {
    id: 'compassion',
    name: 'Compassion',
    color: '#8b5cf6',
    bgClass: 'border-accent-purple/20 text-accent-purple hover:bg-accent-purple/5',
    activeBgClass: 'bg-accent-purple/10 border-accent-purple/50 text-accent-purple glow-purple',
    accentColor: '#8b5cf6',
    description: 'Warm, slow, soft tone. Broad gentle smile, violet lighting.',
    voiceParams: { speed: 0.80, pitch: 1.05 }
  },
  {
    id: 'concerned',
    name: 'Concerned',
    color: '#f97316',
    bgClass: 'border-accent-orange/20 text-accent-orange hover:bg-accent-orange/5',
    activeBgClass: 'bg-accent-orange/10 border-accent-orange/50 text-accent-orange glow-orange',
    accentColor: '#f97316',
    description: 'Caring, moderate speed. Empathetic tilt, orange glow.',
    voiceParams: { speed: 0.85, pitch: 0.90 }
  },
  {
    id: 'encouraging',
    name: 'Encouraging',
    color: '#10b981',
    bgClass: 'border-accent-emerald/20 text-accent-emerald hover:bg-accent-emerald/5',
    activeBgClass: 'bg-accent-emerald/10 border-accent-emerald/50 text-accent-emerald glow-emerald',
    accentColor: '#10b981',
    description: 'Bright, motivating tone. Upward smile, emerald lighting.',
    voiceParams: { speed: 1.15, pitch: 1.10 }
  },
  {
    id: 'supportive',
    name: 'Supportive',
    color: '#34d399',
    bgClass: 'border-accent-emerald/20 text-accent-emerald hover:bg-accent-emerald/5',
    activeBgClass: 'bg-accent-emerald/10 border-accent-emerald/50 text-accent-emerald glow-emerald',
    accentColor: '#34d399',
    description: 'Kind, steady cadence. Broad curve, soft green glow.',
    voiceParams: { speed: 0.90, pitch: 1.00 }
  },
  {
    id: 'thinking',
    name: 'Thinking',
    color: '#eab308',
    bgClass: 'border-accent-amber/20 text-accent-amber hover:bg-accent-amber/5',
    activeBgClass: 'bg-accent-amber/10 border-accent-amber/50 text-accent-amber glow-amber',
    accentColor: '#eab308',
    description: 'Slow, hesitant pace. Wavy mouth curve, yellow lights.',
    voiceParams: { speed: 0.80, pitch: 0.92 }
  },
  {
    id: 'fearful',
    name: 'Fearful',
    color: '#a78bfa',
    bgClass: 'border-accent-purple/20 text-accent-purple hover:bg-accent-purple/5',
    activeBgClass: 'bg-accent-purple/10 border-accent-purple/50 text-accent-purple glow-purple',
    accentColor: '#a78bfa',
    description: 'Fast, trembling speed. Tremulous smile and purple aura.',
    voiceParams: { speed: 1.20, pitch: 0.85 }
  }
];

function LoadingFallback() {
  return (
    <mesh>
      <sphereGeometry args={[0.3, 16, 16]} />
      <meshStandardMaterial color="#3b82f6" emissive="#3b82f6" emissiveIntensity={0.5} wireframe />
    </mesh>
  );
}

function SceneBackground() {
  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight position={[3, 4, 5]} intensity={0.8} color="#e2e8f0" />
      <directionalLight position={[-3, 2, 3]} intensity={0.3} color="#3b82f6" />
      <directionalLight position={[0, -2, -4]} intensity={0.2} color="#8b5cf6" />
      <mesh position={[0, 0, -3]} scale={[12, 12, 1]}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial color="#000000" />
      </mesh>
      <gridHelper args={[10, 20, '#1e293b', '#111827']} position={[0, -2, 0]} />
    </>
  );
}

export default function EmotionPlayground() {
  const [text, setText] = useState(PRESET_PHRASES[0]);
  const [selectedEmotion, setSelectedEmotion] = useState('neutral');
  const [speedSlider, setSpeedSlider] = useState(1.0);
  const [pitchSlider, setPitchSlider] = useState(1.0);

  // Store actions
  const setIsSpeaking = useAppStore((s) => s.setIsSpeaking);

  // Runtime states
  const [loadingEmotion, setLoadingEmotion] = useState(null);
  const [playingEmotion, setPlayingEmotion] = useState(null);
  const [sequentialMode, setSequentialMode] = useState(false);

  const audioRef = useRef(null);
  const abortRef = useRef(false);

  // Sync sliders to selected emotion default when selectedEmotion changes
  useEffect(() => {
    const emoData = EMOTIONS.find((e) => e.id === selectedEmotion);
    if (emoData) {
      setSpeedSlider(emoData.voiceParams.speed);
      setPitchSlider(emoData.voiceParams.pitch);
    }
  }, [selectedEmotion]);

  const stopPlayback = useCallback(() => {
    abortRef.current = true;
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
      audioRef.current = null;
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    setPlayingEmotion(null);
    setLoadingEmotion(null);
    setSequentialMode(false);
    setIsSpeaking(false);
  }, [setIsSpeaking]);

  const playWithWebSpeech = useCallback((phrase, emotionId, speed, pitch) => {
    return new Promise((resolve) => {
      if (!window.speechSynthesis) return resolve();
      const utterance = new SpeechSynthesisUtterance(phrase);
      utterance.rate = speed;
      utterance.pitch = pitch;
      utterance.volume = 1.0;
      utterance.onend = () => resolve(true);
      utterance.onerror = () => resolve(false);
      window.speechSynthesis.speak(utterance);
    });
  }, []);

  const playEmotionAudio = useCallback(async (emotionId, speedVal, pitchVal, skipReset = false) => {
    if (!skipReset) {
      stopPlayback();
    }
    abortRef.current = false;
    setSelectedEmotion(emotionId);
    setLoadingEmotion(emotionId);

    const speed = speedVal ?? speedSlider;
    const pitch = pitchVal ?? pitchSlider;

    try {
      const voiceParams = {
        speed,
        pitch,
        warmth: emotionId === 'compassion' || emotionId === 'supportive' ? 0.9 : 0.6
      };

      const result = await synthesizeSpeech(text, emotionId, voiceParams);

      if (abortRef.current) return;

      if (result && result.audio_url) {
        setLoadingEmotion(null);
        setPlayingEmotion(emotionId);
        setIsSpeaking(true);

        const audio = new Audio(result.audio_url);
        audioRef.current = audio;

        await new Promise((resolve) => {
          audio.onended = () => resolve(true);
          audio.onerror = () => resolve(false);
          audio.play().catch(() => resolve(false));
        });
      } else {
        throw new Error(result?.error || 'No audio url');
      }
    } catch (err) {
      console.warn('Fallback to web speech synthesis', err);
      if (!abortRef.current) {
        setLoadingEmotion(null);
        setPlayingEmotion(emotionId);
        setIsSpeaking(true);
        await playWithWebSpeech(text, emotionId, speed, pitch);
      }
    } finally {
      setIsSpeaking(false);
      if (!abortRef.current) {
        setPlayingEmotion(null);
      }
    }
  }, [text, speedSlider, pitchSlider, stopPlayback, playWithWebSpeech, setIsSpeaking]);

  const playSequentialMix = useCallback(async () => {
    stopPlayback();
    abortRef.current = false;
    setSequentialMode(true);

    const targetEmotions = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'compassion'];

    for (const emoId of targetEmotions) {
      if (abortRef.current) break;
      const emoData = EMOTIONS.find((e) => e.id === emoId);
      if (emoData) {
        setSelectedEmotion(emoId);
        await playEmotionAudio(
          emoId,
          emoData.voiceParams.speed,
          emoData.voiceParams.pitch,
          true
        );
      }
      // Brief pause between segments
      if (!abortRef.current) {
        await new Promise((r) => setTimeout(r, 600));
      }
    }

    if (!abortRef.current) {
      setSequentialMode(false);
    }
  }, [playEmotionAudio, stopPlayback]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      abortRef.current = true;
      if (audioRef.current) {
        audioRef.current.pause();
      }
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
      setIsSpeaking(false);
    };
  }, [setIsSpeaking]);

  const selectedEmotionData = useMemo(() => {
    return EMOTIONS.find((e) => e.id === selectedEmotion) || EMOTIONS[0];
  }, [selectedEmotion]);

  return (
    <div className="flex-1 flex flex-col md:flex-row min-h-0 h-full overflow-hidden bg-navy-950/20">
      {/* ── Left Column: 3D Robot Visualizer ── */}
      <div className="w-full md:w-[45%] flex flex-col border-b md:border-b-0 md:border-r border-white/5 relative min-h-[300px] md:min-h-0 bg-navy-900/40">
        
        {/* Render Canvas */}
        <div className="flex-1 relative">
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-navy-900/90 pointer-events-none z-10" />
          
          <div className="absolute top-4 left-4 z-20 flex flex-col gap-1.5">
            <span className="text-[10px] text-white/40 font-bold uppercase tracking-widest">
              Live Avatar Engine
            </span>
            <div className="flex items-center gap-2">
              <span 
                className="w-2.5 h-2.5 rounded-full transition-smooth"
                style={{ 
                  backgroundColor: selectedEmotionData.color,
                  boxShadow: `0 0 10px ${selectedEmotionData.color}` 
                }}
              />
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                {selectedEmotionData.name} Robot state
              </h2>
            </div>
          </div>

          {/* Reset Orbit Controls tooltip */}
          <span className="absolute bottom-4 right-4 text-[10px] text-white/30 z-20 pointer-events-none">
            Drag to rotate • Pinch to zoom
          </span>

          <Canvas
            camera={{ position: [0, 0, 3.5], fof: 45 }}
            dpr={[1, 2]}
            gl={{ antialias: true, alpha: true }}
            style={{ background: 'transparent' }}
          >
            <SceneBackground />
            <Suspense fallback={<LoadingFallback />}>
              <RobotAvatar emotion={selectedEmotion} />
            </Suspense>
            <OrbitControls
              enableZoom={true}
              enablePan={false}
              minPolarAngle={Math.PI / 3}
              maxPolarAngle={Math.PI / 1.8}
              minAzimuthAngle={-Math.PI / 4}
              maxAzimuthAngle={Math.PI / 4}
              rotateSpeed={0.5}
            />
          </Canvas>
        </div>

        {/* Emotion Description Overlay */}
        <div className="p-5 border-t border-white/5 z-20 bg-navy-900/60 backdrop-blur-md">
          <div className="glass-panel p-4 flex flex-col gap-2">
            <div className="flex justify-between items-center border-b border-white/5 pb-2">
              <span className="text-[11px] font-bold text-white/50 uppercase">Expression parameters</span>
              <span className="text-[11px] px-2 py-0.5 rounded bg-white/5 border border-white/10 font-mono text-white/80" style={{ color: selectedEmotionData.color }}>
                {selectedEmotionData.id}
              </span>
            </div>
            <p className="text-xs text-white/70 leading-relaxed mt-1">
              {selectedEmotionData.description}
            </p>
            <div className="grid grid-cols-2 gap-3 mt-3 pt-3 border-t border-white/5">
              <div>
                <span className="block text-[10px] text-white/40 font-medium">Default Speed</span>
                <span className="text-xs font-mono text-white font-bold">{selectedEmotionData.voiceParams.speed}x</span>
              </div>
              <div>
                <span className="block text-[10px] text-white/40 font-medium">Default Pitch</span>
                <span className="text-xs font-mono text-white font-bold">{selectedEmotionData.voiceParams.pitch}x</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Right Column: Configuration & Emotion Grid ── */}
      <div className="flex-1 flex flex-col min-h-0 overflow-y-auto p-5 md:p-6 gap-6">
        
        {/* Playback Controls & Header */}
        <div className="flex flex-col gap-4">
          <div className="flex justify-between items-start md:items-center flex-col md:flex-row gap-3">
            <div>
              <h2 className="text-xl font-extrabold text-white tracking-tight">Emotion Playground</h2>
              <p className="text-xs text-white/50 mt-0.5">Explore how the same text dynamically outputs different audio accents and visual configurations.</p>
            </div>

            <div className="flex items-center gap-2 w-full md:w-auto">
              {playingEmotion || loadingEmotion || sequentialMode ? (
                <button
                  onClick={stopPlayback}
                  className="w-full md:w-auto flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl bg-accent-red/20 border border-accent-red/40 hover:bg-accent-red/30 transition-smooth text-accent-red text-xs font-bold btn-hover"
                >
                  <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <rect x="9" y="9" width="6" height="6" fill="currentColor" />
                  </svg>
                  Stop Playback
                </button>
              ) : (
                <button
                  onClick={playSequentialMix}
                  className="w-full md:w-auto flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl bg-gradient-to-r from-accent-blue to-accent-purple text-white shadow-lg shadow-accent-purple/15 text-xs font-bold transition-smooth btn-hover"
                >
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" />
                  </svg>
                  Play Sequential Mix
                </button>
              )}
            </div>
          </div>

          {/* Text Input Block */}
          <div className="glass-panel p-4 flex flex-col gap-3">
            <span className="text-[11px] font-bold text-white/40 uppercase tracking-wider">Test Text Phrase</span>
            <textarea
              className="w-full bg-navy-950/60 border border-white/5 rounded-xl p-3 text-sm text-white focus:border-accent-purple/40 transition-smooth min-h-[70px] resize-none"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type anything you want the robot to say..."
              maxLength={400}
            />

            {/* Presets */}
            <div className="flex flex-col gap-1.5">
              <span className="text-[10px] text-white/30 font-medium">Quick Presets:</span>
              <div className="flex flex-wrap gap-2">
                {PRESET_PHRASES.map((preset, index) => (
                  <button
                    key={index}
                    onClick={() => setText(preset)}
                    className={`text-[11px] px-2.5 py-1.5 rounded-lg border transition-smooth text-left max-w-[200px] truncate ${
                      text === preset 
                        ? 'bg-white/10 border-white/20 text-white font-medium'
                        : 'bg-white/0 border-white/5 text-white/60 hover:bg-white/5 hover:text-white/80'
                    }`}
                  >
                    Preset {index + 1}: "{preset}"
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Voice Fine-tuning Sliders */}
          <div className="glass-panel p-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="flex flex-col gap-2">
              <div className="flex justify-between items-center text-xs font-bold">
                <span className="text-white/40 uppercase tracking-wider">Voice Speed</span>
                <span className="font-mono text-accent-cyan">{speedSlider.toFixed(2)}x</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="1.6"
                step="0.05"
                value={speedSlider}
                onChange={(e) => setSpeedSlider(parseFloat(e.target.value))}
                className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-accent-cyan"
              />
              <span className="text-[9px] text-white/30">Adjust the tempo of synthesis (0.5x slowest - 1.6x fastest)</span>
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex justify-between items-center text-xs font-bold">
                <span className="text-white/40 uppercase tracking-wider">Voice Pitch</span>
                <span className="font-mono text-accent-purple">{pitchSlider.toFixed(2)}x</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="1.5"
                step="0.05"
                value={pitchSlider}
                onChange={(e) => setPitchSlider(parseFloat(e.target.value))}
                className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-accent-purple"
              />
              <span className="text-[9px] text-white/30">Shift vocal tone frequency (0.5x lowest - 1.5x highest)</span>
            </div>
          </div>
        </div>

        {/* Grid of Emotions */}
        <div className="flex flex-col gap-3">
          <span className="text-[11px] font-bold text-white/40 uppercase tracking-wider">Available Emotion Profiles ({EMOTIONS.length})</span>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {EMOTIONS.map((emo) => {
              const isSelected = selectedEmotion === emo.id;
              const isLoading = loadingEmotion === emo.id;
              const isPlaying = playingEmotion === emo.id;
              
              return (
                <div
                  key={emo.id}
                  onClick={() => {
                    setSelectedEmotion(emo.id);
                  }}
                  className={`border rounded-2xl p-4 flex items-center justify-between cursor-pointer transition-smooth relative overflow-hidden ${
                    isSelected ? emo.activeBgClass : emo.bgClass
                  } bg-navy-900/20`}
                >
                  <div className="flex-1 min-w-0 pr-3">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: emo.color }} />
                      <h4 className="text-sm font-bold text-white tracking-wide">{emo.name}</h4>
                      {isPlaying && (
                        <span className="flex items-center gap-0.5 ml-1">
                          <span className="w-0.5 h-3 bg-white animate-bounce" style={{ animationDelay: '0.1s' }} />
                          <span className="w-0.5 h-4 bg-white animate-bounce" style={{ animationDelay: '0.2s' }} />
                          <span className="w-0.5 h-2.5 bg-white animate-bounce" style={{ animationDelay: '0.3s' }} />
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-white/50 leading-relaxed mt-1">{emo.description}</p>
                  </div>

                  <div className="flex-shrink-0">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (isPlaying || isLoading) {
                          stopPlayback();
                        } else {
                          playEmotionAudio(emo.id);
                        }
                      }}
                      className="w-10 h-10 rounded-xl glass-panel-dense flex items-center justify-center transition-smooth hover:bg-white/10 hover:scale-105 active:scale-95 border border-white/5"
                    >
                      {isLoading ? (
                        <svg className="w-4 h-4 animate-spin text-white" viewBox="0 0 24 24" fill="none">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                      ) : isPlaying ? (
                        <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <rect x="6" y="6" width="12" height="12" fill="currentColor" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4 text-white/80 hover:text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" />
                        </svg>
                      )}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
