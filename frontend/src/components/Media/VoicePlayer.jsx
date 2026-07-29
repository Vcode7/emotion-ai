import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { synthesizeSpeech } from '../../api/client';
import useAppStore from '../../store/useAppStore';

/**
 * VoicePlayer — plays TTS audio for response segments.
 * Falls back to Web Speech API when the TTS endpoint is unavailable.
 * Dynamically updates the 3D Robot Avatar emotion during playback.
 */
export default function VoicePlayer({
  segments,
  content,
  emotion,
  voiceParams,
  avatarCommands,
  autoPlay = false,
  onDone,
}) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentSegmentIndex, setCurrentSegmentIndex] = useState(-1);
  const audioRef = useRef(null);
  const abortRef = useRef(false);
  const autoPlayedRef = useRef(false);

  const setIsSpeaking = useAppStore((s) => s.setIsSpeaking);
  const setCurrentEmotion = useAppStore((s) => s.setCurrentEmotion);
  const setCurrentAvatarCommands = useAppStore((s) => s.setCurrentAvatarCommands);

  // Normalize segments array
  const playableSegments = useMemo(() => {
    if (Array.isArray(segments) && segments.length > 0) {
      return segments;
    }
    if (content) {
      return [{ text: content, emotion: emotion || 'neutral', avatarCommands }];
    }
    return [];
  }, [segments, content, emotion, avatarCommands]);

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
    setIsPlaying(false);
    setCurrentSegmentIndex(-1);
    setIsSpeaking(false);
  }, [setIsSpeaking]);

  const playWithWebSpeech = useCallback((text, emo) => {
    return new Promise((resolve) => {
      if (!window.speechSynthesis) return resolve();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = voiceParams?.speed || 0.95;
      utterance.pitch = voiceParams?.pitch || (emo === 'happy' ? 1.2 : emo === 'sad' ? 0.8 : 1.0);
      utterance.volume = 1.0;
      utterance.onend = () => resolve(true);
      utterance.onerror = () => resolve(false);
      window.speechSynthesis.speak(utterance);
    });
  }, [voiceParams]);

  const playSegments = useCallback(async () => {
    if (!playableSegments || playableSegments.length === 0) return;

    setIsPlaying(true);
    setIsSpeaking(true);
    abortRef.current = false;

    for (let i = 0; i < playableSegments.length; i++) {
      if (abortRef.current) break;

      const segment = playableSegments[i];
      const text = segment.text || segment.content || '';
      const segEmotion = segment.emotion || emotion || 'neutral';
      const segAvatarCmds = segment.avatarCommands || avatarCommands || null;

      if (!text.trim()) continue;

      setCurrentSegmentIndex(i);

      // Sync active visual emotion/commands for this spoken segment
      setCurrentEmotion(segEmotion);
      if (segAvatarCmds) {
        setCurrentAvatarCommands(segAvatarCmds);
      }

      try {
        const ttsResult = await synthesizeSpeech(text, segEmotion, voiceParams);

        if (abortRef.current) break;

        if (ttsResult && ttsResult.audio_url) {
          const audio = new Audio(ttsResult.audio_url);
          audioRef.current = audio;

          await new Promise((resolve) => {
            audio.onended = () => resolve(true);
            audio.onerror = () => resolve(false);
            audio.play().catch(() => resolve(false));
          });
        } else {
          throw new Error(ttsResult?.error || 'TTS unavailable');
        }
      } catch (_err) {
        if (!abortRef.current && window.speechSynthesis) {
          await playWithWebSpeech(text, segEmotion);
        }
      }
    }

    setIsSpeaking(false);

    if (!abortRef.current) {
      setIsPlaying(false);
      setCurrentSegmentIndex(-1);
      onDone?.();
    }
  }, [
    playableSegments,
    emotion,
    avatarCommands,
    voiceParams,
    onDone,
    playWithWebSpeech,
    setIsSpeaking,
    setCurrentEmotion,
    setCurrentAvatarCommands,
  ]);

  // Handle autoPlay on mount
  useEffect(() => {
    if (autoPlay && !autoPlayedRef.current) {
      autoPlayedRef.current = true;
      playSegments();
    }
  }, [autoPlay, playSegments]);

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

  if (!playableSegments || playableSegments.length === 0) return null;

  return (
    <div className="flex items-center gap-2 mt-2">
      {!isPlaying ? (
        <button
          onClick={playSegments}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-smooth btn-hover text-white/60 hover:text-white/90"
        >
          <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M11.5 3.05a.75.75 0 011.28-.53l5.25 5.25a.75.75 0 010 1.06l-5.25 5.25a.75.75 0 01-1.28-.53V9.31c-3.77.35-6.48 2.13-8.14 4.37a.75.75 0 01-1.36-.56c1.26-5.36 5.5-8.53 9.5-9.17V3.05z" />
          </svg>
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072M17.95 6.05a8 8 0 010 11.9M6.5 8.8v6.4a.8.8 0 00.8.8h2.4l3.3 3.3V5.5l-3.3 3.3H7.3a.8.8 0 00-.8.8z" />
          </svg>
          <span className="text-[11px] font-medium">Listen</span>
        </button>
      ) : (
        <button
          onClick={stopPlayback}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-accent-blue/15 border border-accent-blue/30 transition-smooth btn-hover text-accent-blue"
        >
          <div className="relative w-4 h-4">
            <svg className="w-4 h-4 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072M17.95 6.05a8 8 0 010 11.9M6.5 8.8v6.4a.8.8 0 00.8.8h2.4l3.3 3.3V5.5l-3.3 3.3H7.3a.8.8 0 00-.8.8z" />
            </svg>
          </div>
          <span className="text-[11px] font-medium">
            Playing {currentSegmentIndex + 1}/{playableSegments.length}
          </span>
        </button>
      )}
    </div>
  );
}
