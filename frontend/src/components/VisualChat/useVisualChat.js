import { useRef, useCallback, useState } from 'react';
import useAppStore from '../../store/useAppStore';
import useVoiceActivityDetection from '../../hooks/useVoiceActivityDetection';
import { chatVisual, synthesizeSpeech, logCallComplete } from '../../api/client';
import { attachAudioElement } from './audioAnalyser';
import { getFemaleVoice } from '../../utils/speech';
import { normalizeEmotion } from '../../utils/emotionLabels';
import { v4 as uuidv4 } from 'uuid';

// Webcam frames sampled while the user is speaking (the face at rest after
// speech ends says little). Frames are downscaled before upload.
const FRAME_INTERVAL_MS = 700;
const MAX_SPEECH_FRAMES = 4;
const FRAME_MAX_WIDTH = 480;

/**
 * useVisualChat — Orchestration hook for the Visual Chat pipeline.
 *
 * State machine:
 *   idle → starting → listening → processing → speaking → listening → …
 *
 * Pipeline:
 *   1. While the user speaks, sample a webcam frame every ~700 ms (max 4),
 *      plus one more at speech end.
 *   2. Send audio blob + frames to POST /chat/visual.
 *   3. Backend gates out non-speech (returns no_speech → stay listening),
 *      runs Whisper + text / voice / face emotion + DSMR, and fuses them.
 *   4. Backend calls the LLM with the fused read + conversation history.
 *   5. Backend returns emotion(sentence) as llm_emotion + llm_sentence.
 *   6. Update avatar with llm_emotion (AI response emotion, NOT user emotion).
 *   7. TTS on llm_sentence → play audio.
 *   8. Store the user emotion signals (+ fused read) in Zustand for the UI panel.
 */
export default function useVisualChat() {
  const store = useAppStore;

  const [isActive, setIsActive] = useState(false);
  const [status, setStatus] = useState('idle'); // 'idle'|'listening'|'processing'|'speaking'
  const [transcript, setTranscript] = useState('');
  const [conversation, setConversation] = useState([]);
  const [multimodalEmotions, setMultimodalEmotions] = useState(null);
  const [webcamStream, setWebcamStream] = useState(null);
  const [callError, setCallError] = useState(null);

  const processingRef = useRef(false);
  const speakingRef = useRef(false);   // suppress VAD while AI speaks
  const isBusyRef = useRef(false);     // suppress VAD while backend processing OR speaking
  const activeRef = useRef(false);
  const audioRef = useRef(null);       // current playing Audio element
  const webcamStreamRef = useRef(null);
  const offscreenVideoRef = useRef(null); // dedicated offscreen video for reliable frame capture
  const callStartTimeRef = useRef(null);
  const callTurnsRef = useRef([]);
  const speechFramesRef = useRef([]);       // frames captured during the current utterance
  const frameTimerRef = useRef(null);

  // ── Get fresh store values ────────────────────────────────────────────────
  const getStoreValues = () => {
    const s = store.getState();
    return {
      sessionId:    s.sessionId,
      llmProvider:  s.llmProvider,
      llmApiKey:    s.llmApiKey,
      sttProvider:  s.sttProvider,
      groqApiKey:   s.groqApiKey,
    };
  };

  // ── Capture webcam snapshot as JPEG Blob (downscaled to ≤ 480 px wide) ────
  const captureWebcamSnapshot = useCallback(() => {
    return new Promise((resolve) => {
      try {
        const stream = webcamStreamRef.current;

        if (!stream || !stream.active) {
          console.log('[VisualChat] No active webcam stream for snapshot');
          resolve(null);
          return;
        }

        const videoTrack = stream.getVideoTracks()[0];
        if (!videoTrack || videoTrack.readyState !== 'live') {
          console.log('[VisualChat] Video track not live');
          resolve(null);
          return;
        }

        // Try offscreen video first, fall back to any playing DOM video rendering this stream
        let video = offscreenVideoRef.current;
        if (!video || video.readyState < 2 || video.videoWidth === 0) {
          if (typeof document !== 'undefined') {
            const domVideos = document.querySelectorAll('video');
            for (const v of domVideos) {
              if (v.srcObject === stream && v.videoWidth > 0 && v.readyState >= 2) {
                video = v;
                break;
              }
            }
          }
        }

        const srcWidth = video?.videoWidth || 0;
        const srcHeight = video?.videoHeight || 0;

        if (!video || srcWidth === 0 || srcHeight === 0) {
          console.log('[VisualChat] Video frame not ready yet — proceeding without image');
          resolve(null);
          return;
        }

        const scale = Math.min(1, FRAME_MAX_WIDTH / srcWidth);
        const width = Math.round(srcWidth * scale);
        const height = Math.round(srcHeight * scale);
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, width, height);

        canvas.toBlob((blob) => {
          if (blob) {
            console.log(`[VisualChat] Captured webcam snapshot (${blob.size} bytes, ${width}x${height})`);
          }
          resolve(blob || null);
        }, 'image/jpeg', 0.85);
      } catch (err) {
        console.warn('[VisualChat] Webcam snapshot failed:', err);
        resolve(null);
      }
    });
  }, []);

  // ── Web Speech fallback if backend TTS fails or is unavailable ────────────
  const playSpeechSynthesis = useCallback((text, emotion, voiceParams) => {
    return new Promise((resolve) => {
      if (typeof window === 'undefined' || !window.speechSynthesis || !activeRef.current) {
        return resolve();
      }

      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      const femaleVoice = getFemaleVoice();
      if (femaleVoice) {
        utterance.voice = femaleVoice;
      }
      utterance.rate = voiceParams?.speed || 0.95;
      const norm = normalizeEmotion(emotion);
      const basePitch = voiceParams?.pitch || (norm === 'happy' ? 1.15 : norm === 'sad' ? 0.95 : 1.05);
      utterance.pitch = Math.max(1.0, basePitch);
      utterance.volume = 1.0;

      speakingRef.current = true;
      isBusyRef.current = true;
      store.getState().setIsSpeaking(true);
      setStatus('speaking');

      let completed = false;
      const onFinish = () => {
        if (completed) return;
        completed = true;
        speakingRef.current = false;
        store.getState().setIsSpeaking(false);
        isBusyRef.current = processingRef.current || false;
        if (activeRef.current && !processingRef.current) setStatus('listening');
        resolve();
      };

      utterance.onend = onFinish;
      utterance.onerror = (e) => {
        console.warn('[WebSpeech] Synthesis error:', e);
        onFinish();
      };

      window.speechSynthesis.speak(utterance);
    });
  }, []);

  // ── Sample webcam frames while the user is speaking ───────────────────────
  const stopFrameSampling = useCallback(() => {
    if (frameTimerRef.current) {
      clearInterval(frameTimerRef.current);
      frameTimerRef.current = null;
    }
  }, []);

  const handleSpeechStart = useCallback(() => {
    stopFrameSampling();
    speechFramesRef.current = [];
    if (!webcamStreamRef.current) return;

    const grab = async () => {
      if (speechFramesRef.current.length >= MAX_SPEECH_FRAMES) {
        stopFrameSampling();
        return;
      }
      const blob = await captureWebcamSnapshot();
      if (blob) speechFramesRef.current.push(blob);
    };
    grab();
    frameTimerRef.current = setInterval(grab, FRAME_INTERVAL_MS);
  }, [captureWebcamSnapshot, stopFrameSampling]);

  // ── Handle speech end from VAD ────────────────────────────────────────────
  const handleSpeechEnd = useCallback(async (audioBlob, chunkId = `chunk_${Date.now()}`) => {
    stopFrameSampling();
    const speechFrames = speechFramesRef.current;
    speechFramesRef.current = [];

    if (!activeRef.current || processingRef.current || speakingRef.current) {
      console.log(`[VAD] [${chunkId}] Ignored — active: ${activeRef.current}, busy: ${processingRef.current || speakingRef.current}`);
      return;
    }
    processingRef.current = true;
    isBusyRef.current = true;
    setStatus('processing');
    store.getState().setIsProcessing(true);

    const sv = getStoreValues();
    console.log(`[VAD] [${chunkId}] Dispatching speech to 4-modality visual chat pipeline`);

    try {
      // 1. Frames captured during speech + one at the end
      const endFrame = await captureWebcamSnapshot();
      const imageBlobs = [...speechFrames, ...(endFrame ? [endFrame] : [])];
      console.log(`[VisualChat] [${chunkId}] Frames: ${imageBlobs.length} (${speechFrames.length} during speech)`);

      // 2. Send to POST /chat/visual
      const effectiveSttProvider = sv.sttProvider || 'groq';
      const effectiveGroqKey = sv.groqApiKey || '';
      const effectiveLlmApiKey = sv.llmApiKey || (sv.llmProvider === 'groq' ? sv.groqApiKey : '');
      let response = null;
      try {
        console.log(
          `[Whisper] 📤 Sending ${audioBlob.size} bytes — ` +
          `session: ${sv.sessionId?.substring(0, 8)} — provider: ${sv.llmProvider} — ${new Date().toISOString()}`
        );
        console.log(`[VisualChat] [${chunkId}] Calling /chat/visual (provider: ${sv.llmProvider}, STT: ${effectiveSttProvider})`);
        response = await chatVisual(
          audioBlob,
          imageBlobs,
          sv.sessionId,
          sv.llmProvider,
          effectiveLlmApiKey,
          effectiveSttProvider,
          effectiveGroqKey,
        );
      } catch (err) {
        console.error(`[VisualChat] [${chunkId}] /chat/visual failed:`, err);
        const errMsg = err?.response?.data?.detail || err?.message || 'Visual chat request failed';
        setCallError(errMsg);
      }

      if (!activeRef.current || !response) {
        console.log(`[VisualChat] [${chunkId}] Pipeline cancelled or empty response`);
        processingRef.current = false;
        if (activeRef.current) setStatus('listening');
        store.getState().setIsProcessing(false);
        return;
      }

      // 2b. Speech gate: the backend found no real speech (noise, echo, or a
      //     Whisper hallucination like "Thank you.") — stay quietly listening.
      if (response.no_speech) {
        console.log(
          `[VisualChat] [${chunkId}] No speech (${response.no_speech_reason || 'unknown'}) — ` +
          `ignored transcript "${response.transcript || ''}"`
        );
        return;
      }

      // 3. Extract transcript
      const text = response.transcript?.trim() || '(inaudible)';
      console.log(
        `[Whisper] ✅ Transcription returned: "${text.substring(0, 100)}" — ${new Date().toISOString()}`
      );
      setTranscript(text);

      // 4. Extract 4 user emotion signals
      const newMultimodalEmotions = {
        text:  response.text_emotion  || { emotion: 'neutral', confidence: 0, available: false },
        audio: response.audio_emotion || { emotion: 'neutral', confidence: 0, available: false },
        face:  response.face_emotion  || { emotion: 'neutral', confidence: 0, available: false, face_detected: false },
        dsmr:  response.dsmr_emotion  || { emotion: 'neutral', confidence: 0 },
        fused: response.fused_emotion || null,
      };
      setMultimodalEmotions(newMultimodalEmotions);
      store.getState().setMultimodalEmotions(newMultimodalEmotions);

      // 5. Extract AI response emotion (LLM reasoning over 4 signals)
      const llmEmotion  = response.llm_emotion  || 'neutral';
      const llmSentence = response.llm_sentence  || '';

      // ── Record Call Turn and Output Structured Log ─────────────────────────
      const turnNumber = callTurnsRef.current.length + 1;
      const turnRecord = {
        turnNumber,
        userSaid: text,
        userEmotion: {
          text: newMultimodalEmotions.text?.emotion || 'neutral',
          textConfidence: newMultimodalEmotions.text?.confidence || 0,
          audio: newMultimodalEmotions.audio?.emotion || 'neutral',
          audioConfidence: newMultimodalEmotions.audio?.confidence || 0,
          face: newMultimodalEmotions.face?.face_detected === false
            ? 'No face detected'
            : (newMultimodalEmotions.face?.emotion || 'neutral'),
          faceConfidence: newMultimodalEmotions.face?.confidence || 0,
          dsmr: newMultimodalEmotions.dsmr?.emotion || 'neutral',
          dsmrConfidence: newMultimodalEmotions.dsmr?.confidence || 0,
          overall: newMultimodalEmotions.fused?.emotion || 'neutral',
          overallConfidence: newMultimodalEmotions.fused?.confidence || 0,
        },
        aiResponded: llmSentence,
        aiEmotion: llmEmotion,
        timestamp: new Date().toISOString(),
      };
      callTurnsRef.current.push(turnRecord);

      console.log(
        `%c==================== 📞 CALL TURN #${turnNumber} COMPLETE ====================\n` +
        `%c[USER SAID]     : %c"${text}"\n` +
        `%c[USER EMOTIONS]  :\n` +
        `%c  • Text Emotion  : %c${turnRecord.userEmotion.text} (${Math.round(turnRecord.userEmotion.textConfidence * 100)}%)\n` +
        `%c  • Voice Emotion : %c${turnRecord.userEmotion.audio} (${Math.round(turnRecord.userEmotion.audioConfidence * 100)}%)\n` +
        `%c  • Face Emotion  : %c${turnRecord.userEmotion.face} (${Math.round(turnRecord.userEmotion.faceConfidence * 100)}%)\n` +
        `%c  • DSMR Sentiment: %c${turnRecord.userEmotion.dsmr}\n` +
        `%c  • Overall       : %c${turnRecord.userEmotion.overall} (${Math.round(turnRecord.userEmotion.overallConfidence * 100)}%)\n` +
        `%c---------------------------------------------------------------\n` +
        `%c[AI RESPONDED]  : %c"${llmSentence}"\n` +
        `%c[AI EMOTION]    : %c${llmEmotion.toUpperCase()}\n` +
        `%c================================================================`,
        'color: #38bdf8; font-weight: bold; font-size: 13px;',
        'color: #93c5fd; font-weight: bold;', 'color: #ffffff;',
        'color: #86efac; font-weight: bold;',
        'color: #a7f3d0;', 'color: #ffffff;',
        'color: #a7f3d0;', 'color: #ffffff;',
        'color: #a7f3d0;', 'color: #ffffff;',
        'color: #a7f3d0;', 'color: #ffffff;',
        'color: #a7f3d0;', 'color: #ffffff;',
        'color: #64748b;',
        'color: #f472b6; font-weight: bold;', 'color: #ffffff;',
        'color: #fde047; font-weight: bold;', 'color: #fde047; font-weight: bold;',
        'color: #38bdf8; font-weight: bold;'
      );

      // 6. Add user message to conversation
      const userMsgId = uuidv4();
      setConversation((prev) => [...prev, {
        role: 'user',
        text,
        id: userMsgId,
        multimodalEmotions: newMultimodalEmotions,
      }]);
      store.getState().addMessage({
        id: userMsgId,
        role: 'user',
        content: text,
        multimodalEmotions: newMultimodalEmotions,
        timestamp: Date.now(),
      });

      // 7. Update avatar with AI response emotion (NOT user emotion)
      const avatarCommands = {
        ...(response.avatar_commands || {}),
        expression: llmEmotion,
      };
      store.getState().setCurrentEmotion(llmEmotion);
      store.getState().setCurrentAvatarCommands(avatarCommands);

      // 8. Add assistant message to conversation
      const assistantMsgId = uuidv4();
      setConversation((prev) => [...prev, {
        role: 'assistant',
        text: llmSentence,
        emotion: llmEmotion,
        id: assistantMsgId,
        multimodalEmotions: newMultimodalEmotions,
        latency: response.latency || {},
      }]);
      store.getState().addMessage({
        id: assistantMsgId,
        role: 'assistant',
        content: llmSentence,
        emotion: llmEmotion,
        emotionState: response.emotion_state || null,
        avatarCommands,
        voiceParams: response.voice_params || null,
        timestamp: Date.now(),
      });

      // 9. TTS synthesis + audio playback
      if (llmSentence && activeRef.current) {
        console.log(`[TTS] [${chunkId}] Synthesizing: emotion="${llmEmotion}"`);
        let played = false;
        try {
          const ttsResult = await synthesizeSpeech(llmSentence, llmEmotion, response.voice_params);
          if (activeRef.current && ttsResult?.audio_url) {
            // Backend returns relative paths like /generated/xxx.mp3
            // The Vite proxy routes /api → backend, so we must prefix with /api
            const audioUrl = ttsResult.audio_url.startsWith('/api')
              ? ttsResult.audio_url
              : `/api${ttsResult.audio_url}`;
            console.log(`[TTS] [${chunkId}] Audio URL: ${audioUrl}`);
            await playAudio(audioUrl, chunkId);
            played = true;
            console.log(`[TTS] [${chunkId}] Finished`);
          } else {
            console.warn(`[TTS] [${chunkId}] No audio_url in TTS result:`, ttsResult);
          }
        } catch (err) {
          console.warn(`[TTS] [${chunkId}] Backend TTS failed, using Web Speech fallback:`, err);
        }

        // Web Speech API fallback if backend TTS was unavailable or failed
        if (!played && activeRef.current) {
          console.log(`[WebSpeech] [${chunkId}] Speaking via browser SpeechSynthesis`);
          await playSpeechSynthesis(llmSentence, llmEmotion, response.voice_params);
        }
      }

    } catch (err) {
      console.error(`[VisualChat] [${chunkId}] Pipeline exception:`, err);
    } finally {
      processingRef.current = false;
      store.getState().setIsProcessing(false);
      isBusyRef.current = speakingRef.current || false;
      if (activeRef.current && !speakingRef.current) setStatus('listening');
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [captureWebcamSnapshot, playSpeechSynthesis, stopFrameSampling]);

  // ── Play TTS audio, block VAD while playing ───────────────────────────────
  const playAudio = useCallback((audioUrl, chunkId = '') => {
    return new Promise((resolve) => {
      if (!activeRef.current) {
        console.log(`[TTS] [${chunkId}] Skipped — session inactive`);
        return resolve();
      }

      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }

      speakingRef.current = true;
      isBusyRef.current = true;
      store.getState().setIsSpeaking(true);
      setStatus('speaking');

      let resolved = false;
      const onDone = () => {
        if (resolved) return;
        resolved = true;
        speakingRef.current = false;
        store.getState().setIsSpeaking(false);
        audioRef.current = null;
        isBusyRef.current = processingRef.current || false;
        if (activeRef.current && !processingRef.current) setStatus('listening');
        resolve();
      };

      // Create audio element WITHOUT crossOrigin — the /generated files are
      // served from the same origin via the Vite proxy, so crossOrigin is
      // not needed and causes CORS preflight failures.
      const audio = new Audio(audioUrl);
      audioRef.current = audio;

      audio.onended = () => {
        console.log(`[TTS] [${chunkId}] Audio ended`);
        onDone();
      };

      audio.onerror = (e) => {
        console.warn(`[TTS] [${chunkId}] Audio element error:`, e);
        onDone();
      };

      audio.onpause = () => {
        if (!activeRef.current) {
          onDone();
        }
      };

      // Hook up the Web Audio analyser for lip-sync (best-effort).
      // We do this AFTER setting src so createMediaElementSource works reliably.
      try {
        attachAudioElement(audio);
      } catch (_) {
        // Non-critical — lip sync simply falls back to cadence mode
      }

      // Resume AudioContext first (browser autoplay policy may suspend it)
      const resumeAndPlay = async () => {
        try {
          // Resume the globally stored AudioContext if suspended
          const ctx = window._emotionAIAudioCtx;
          if (ctx && ctx.state === 'suspended') {
            await ctx.resume();
            console.log(`[TTS] [${chunkId}] AudioContext resumed`);
          }
        } catch (_) {}

        try {
          await audio.play();
          console.log(`[TTS] [${chunkId}] Playing: ${audioUrl}`);
        } catch (err) {
          console.warn(`[TTS] [${chunkId}] play() failed (${err.name}): ${err.message}`);
          if (err.name === 'NotAllowedError') {
            console.warn(`[TTS] [${chunkId}] Autoplay blocked — user must interact with the page first.`);
          }
          onDone();
        }
      };

      resumeAndPlay();
    });
  }, []);

  // ── VAD hook ─────────────────────────────────────────────────────────────
  const { startListening, stopListening } = useVoiceActivityDetection({
    onSpeechEnd: handleSpeechEnd,
    onSpeechStart: handleSpeechStart,
    suppressRef: isBusyRef,
    silenceDurationMs: 900,
    speakThreshold: 0.015,
    silenceThreshold: 0.008,
    minUtteranceDurationMs: 500,
    postSuppressCooldownMs: 400,
  });

  // ── Stop conversation ─────────────────────────────────────────────────────
  const stopConversation = useCallback(async () => {
    activeRef.current = false;

    // ── Log Call Completion Session Summary ──
    const turns = [...callTurnsRef.current];
    const durationSeconds = callStartTimeRef.current
      ? parseFloat(((Date.now() - callStartTimeRef.current) / 1000).toFixed(1))
      : 0;

    if (turns.length > 0) {
      console.log(
        `%c================================================================\n` +
        `%c                     📞 CALL COMPLETED SUMMARY                  \n` +
        `%c================================================================\n` +
        `%cTotal Turns: ${turns.length} | Call Duration: ${durationSeconds}s\n` +
        turns.map((t, idx) => (
          `----------------------------------------------------------------\n` +
          `Turn #${idx + 1}:\n` +
          `  [USER SAID]    : "${t.userSaid}"\n` +
          `  [USER EMOTION] : Text: ${t.userEmotion.text} (${Math.round(t.userEmotion.textConfidence * 100)}%) | ` +
          `Voice: ${t.userEmotion.audio} (${Math.round(t.userEmotion.audioConfidence * 100)}%) | ` +
          `Face: ${t.userEmotion.face} | ` +
          `DSMR: ${t.userEmotion.dsmr} | ` +
          `Overall: ${t.userEmotion.overall} (${Math.round(t.userEmotion.overallConfidence * 100)}%)\n` +
          `  [AI RESPONDED] : "${t.aiResponded}"\n` +
          `  [AI EMOTION]   : ${t.aiEmotion.toUpperCase()}`
        )).join('\n') + '\n' +
        `================================================================`,
        'color: #22c55e; font-weight: bold;',
        'color: #22c55e; font-weight: bold; font-size: 14px;',
        'color: #22c55e; font-weight: bold;',
        'color: #cbd5e1; font-weight: bold;'
      );

      const { sessionId } = getStoreValues();
      logCallComplete(sessionId, turns, durationSeconds).catch(() => {});
    }

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }

    if (typeof window !== 'undefined' && window.speechSynthesis) {
      try { window.speechSynthesis.cancel(); } catch (_) {}
    }

    stopListening();
    stopFrameSampling();
    speechFramesRef.current = [];

    if (webcamStreamRef.current) {
      webcamStreamRef.current.getTracks().forEach((t) => {
        try { t.stop(); } catch (_) {}
      });
      webcamStreamRef.current = null;
    }

    if (offscreenVideoRef.current) {
      offscreenVideoRef.current.srcObject = null;
    }

    setWebcamStream(null);
    speakingRef.current = false;
    processingRef.current = false;
    isBusyRef.current = false;
    store.getState().setIsSpeaking(false);
    store.getState().setIsProcessing(false);
    store.getState().setVisualChatActive(false);

    setIsActive(false);
    setStatus('idle');
    setTranscript('');
  }, [stopListening, stopFrameSampling]);

  // ── Start conversation ────────────────────────────────────────────────────
  const startConversation = useCallback(async () => {
    if (activeRef.current) return;
    setCallError(null);

    // Verify mediaDevices support
    if (!navigator?.mediaDevices?.getUserMedia) {
      const msg = !window.isSecureContext
        ? 'Camera and microphone require HTTPS or localhost. If on a phone, open via localhost or use HTTPS.'
        : 'Camera and microphone are not supported in this browser.';
      setCallError(msg);
      console.error('[VisualChat]', msg);
      return;
    }

    // Unlock / resume AudioContext on user click gesture
    if (typeof window !== 'undefined') {
      const ctx = window._emotionAIAudioCtx;
      if (ctx && ctx.state === 'suspended') {
        ctx.resume().catch(() => {});
      }
    }

    activeRef.current = true;
    callStartTimeRef.current = Date.now();
    callTurnsRef.current = [];
    setIsActive(true);
    setStatus('listening');
    setConversation([]);
    setTranscript('');
    setMultimodalEmotions(null);
    store.getState().setVisualChatActive(true);
    store.getState().setMultimodalEmotions(null);

    // 1. Request webcam stream with progressive constraints
    let camStream = null;
    try {
      camStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
    } catch (e1) {
      try {
        camStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });
      } catch (e2) {
        try {
          camStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        } catch (e3) {
          console.warn('[VisualChat] Camera access failed on all constraints:', e3);
        }
      }
    }

    if (!activeRef.current) {
      if (camStream) {
        camStream.getTracks().forEach((t) => { try { t.stop(); } catch (_) {} });
      }
      return;
    }

    if (camStream) {
      webcamStreamRef.current = camStream;
      setWebcamStream(camStream);

      // Create or update dedicated offscreen video for instant frame snapshots
      if (!offscreenVideoRef.current && typeof document !== 'undefined') {
        const offVideo = document.createElement('video');
        offVideo.autoplay = true;
        offVideo.muted = true;
        offVideo.playsInline = true;
        offscreenVideoRef.current = offVideo;
      }
      if (offscreenVideoRef.current) {
        offscreenVideoRef.current.srcObject = camStream;
        offscreenVideoRef.current.play().catch(() => {});
      }
    } else {
      console.warn('[VisualChat] Continuing audio-only (no camera stream).');
    }

    // 2. Start VAD (microphone)
    try {
      await startListening();
    } catch (err) {
      console.error('[VisualChat] Microphone access failed:', err);
      setCallError('Microphone permission was denied. Please allow microphone access in your browser.');
      await stopConversation();
    }
  }, [startListening, stopConversation]);

  return {
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
  };
}
