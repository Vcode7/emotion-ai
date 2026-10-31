import { useRef, useCallback, useEffect } from 'react';

/**
 * useVoiceActivityDetection
 * =========================
 * Energy-based VAD using the Web Audio API AnalyserNode.
 *
 * How it works:
 *   1. Opens the microphone stream.
 *   2. Runs an AnalyserNode on the stream to measure RMS energy every ~50ms.
 *   3. When energy exceeds SPEAK_THRESHOLD → user is speaking.
 *   4. When energy drops below SILENCE_THRESHOLD for >= SILENCE_DURATION_MS →
 *      a "speech end" event fires with the accumulated audio blob.
 *   5. MediaRecorder collects audio chunks. Each utterance gets a FRESH MediaRecorder
 *      instance so that chunk 0 ALWAYS contains the valid EBML container header
 *      and Opus codec parameters.
 *   6. When suppressed (AI is processing or speaking), recording is stopped so
 *      microphone does not capture speaker echo. When suppressed drops, a fresh
 *      recorder starts with a clean EBML header.
 *
 * Logging:
 *   🎙️  Audio capture STARTED  — when recording begins for a new utterance
 *   ✅  Audio capture COMPLETE  — when VAD fires after detecting silence
 *   📤  Sending to Whisper      — just before dispatching the blob to STT
 *
 * @param {object} opts
 * @param {function} opts.onSpeechEnd  - Called with (audioBlob, chunkId) when silence is detected.
 * @param {function} opts.onSpeechStart - Called when speaking is first detected (optional).
 * @param {React.MutableRefObject<boolean>} opts.suppressRef - If true, pauses recording and VAD.
 * @param {number} [opts.silenceDurationMs=900]  - Silence required to trigger onSpeechEnd.
 * @param {number} [opts.speakThreshold=0.015]   - RMS energy level considered "speaking".
 * @param {number} [opts.silenceThreshold=0.008] - RMS energy level considered "silence".
 * @param {number} [opts.minUtteranceDurationMs=300] - Minimum VOICED length (first to last loud frame,
 *                                                     excluding the trailing silence) to fire.
 * @param {number} [opts.postSuppressCooldownMs=400] - After suppression ends (AI stopped speaking),
 *                                                     ignore energy for this long so the TTS echo
 *                                                     tail does not start an utterance.
 */
export default function useVoiceActivityDetection({
  onSpeechEnd,
  onSpeechStart,
  suppressRef,
  silenceDurationMs = 900,
  speakThreshold = 0.015,
  silenceThreshold = 0.008,
  minUtteranceDurationMs = 300,
  postSuppressCooldownMs = 400,
} = {}) {
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const animFrameRef = useRef(null);
  const isSpeakingRef = useRef(false);
  const silenceStartRef = useRef(null);
  const utteranceStartRef = useRef(null);
  const lastVoiceAtRef = useRef(null);
  const cooldownUntilRef = useRef(0);
  const activeRef = useRef(false);
  const mimeTypeRef = useRef('audio/webm');
  const wasSuppressedRef = useRef(false);
  const recorderStartTimeRef = useRef(null);

  // Keep stable callback references
  const onSpeechEndRef = useRef(onSpeechEnd);
  const onSpeechStartRef = useRef(onSpeechStart);
  useEffect(() => { onSpeechEndRef.current = onSpeechEnd; }, [onSpeechEnd]);
  useEffect(() => { onSpeechStartRef.current = onSpeechStart; }, [onSpeechStart]);

  const stopEverything = useCallback(() => {
    activeRef.current = false;
    wasSuppressedRef.current = false;

    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.ondataavailable = null;
      mediaRecorderRef.current.onstop = null;
      if (mediaRecorderRef.current.state !== 'inactive') {
        try { mediaRecorderRef.current.stop(); } catch (_) {}
      }
      mediaRecorderRef.current = null;
    }
    chunksRef.current = [];
    recorderStartTimeRef.current = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => {
        try { t.stop(); } catch (_) {}
      });
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      try { audioContextRef.current.close(); } catch (_) {}
      audioContextRef.current = null;
      analyserRef.current = null;
    }
    isSpeakingRef.current = false;
    silenceStartRef.current = null;
    utteranceStartRef.current = null;
    lastVoiceAtRef.current = null;
    cooldownUntilRef.current = 0;
  }, []);

  const _startRecorder = useCallback((stream) => {
    if (!activeRef.current || !stream) return;

    // Clean up any existing recorder first
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.ondataavailable = null;
      mediaRecorderRef.current.onstop = null;
      if (mediaRecorderRef.current.state !== 'inactive') {
        try { mediaRecorderRef.current.stop(); } catch (_) {}
      }
      mediaRecorderRef.current = null;
    }

    // Always start with an empty chunk array so index 0 is GUARANTEED to be
    // the EBML container header (1a 45 df a3) generated by MediaRecorder.start()
    chunksRef.current = [];

    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';
    mimeTypeRef.current = mimeType;

    try {
      const mr = new MediaRecorder(stream, { mimeType });
      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };
      mr.start(100); // collect data every 100ms
      recorderStartTimeRef.current = Date.now();
      console.log(`[VAD] 🎙️ Audio capture STARTED (fresh WebM container) — ${new Date().toISOString()}`);
      mediaRecorderRef.current = mr;
    } catch (err) {
      console.error('[VAD] Failed to start MediaRecorder:', err);
    }
  }, []);

  const _flushAndFire = useCallback(() => {
    const mr = mediaRecorderRef.current;
    if (!mr || mr.state === 'inactive') return;

    const chunkId = `chunk_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
    const mimeType = mimeTypeRef.current;

    // Voiced duration only: the trailing silence used to confirm the end of
    // speech must not count, or a single click/cough passes the minimum.
    const utteranceDuration = utteranceStartRef.current && lastVoiceAtRef.current
      ? lastVoiceAtRef.current - utteranceStartRef.current
      : 0;

    // Snapshot existing chunks for THIS utterance
    const capturedChunks = [...chunksRef.current];
    chunksRef.current = []; // fresh array for next session

    // Reset VAD state synchronously before stopping recorder
    isSpeakingRef.current = false;
    silenceStartRef.current = null;
    utteranceStartRef.current = null;
    lastVoiceAtRef.current = null;
    mediaRecorderRef.current = null;

    // Crucial: Route any trailing data flushed by mr.stop() into capturedChunks,
    // NEVER into chunksRef.current (which would corrupt the next recording's EBML header)
    mr.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        capturedChunks.push(e.data);
      }
    };

    mr.onstop = () => {
      if (!activeRef.current) return;

      if (capturedChunks.length > 0 && utteranceDuration >= minUtteranceDurationMs) {
        const blob = new Blob(capturedChunks, { type: mimeType });

        console.log(
          `[VAD] ✅ Audio capture COMPLETE — size: ${blob.size} bytes, ` +
          `duration: ${utteranceDuration}ms, chunks: ${capturedChunks.length} — ${new Date().toISOString()}`
        );

        if (blob.size > 1000 && onSpeechEndRef.current) {
          console.log(
            `[VAD] 📤 Sending ${blob.size} bytes to Whisper STT — ${new Date().toISOString()}`
          );
          onSpeechEndRef.current(blob, chunkId);
        } else {
          console.log(`[VAD] [${chunkId}] Blob too small (${blob.size} bytes) — skipped`);
        }
      } else {
        console.log(
          `[VAD] [${chunkId}] Ignored segment — below voiced-duration/size threshold ` +
          `(${utteranceDuration}ms, ${capturedChunks.length} chunks)`
        );
      }

      // If suppressRef is not currently active, restart recorder after brief delay
      const isSuppressed = suppressRef?.current ?? false;
      if (!isSuppressed && activeRef.current && streamRef.current) {
        setTimeout(() => {
          const stillSuppressed = suppressRef?.current ?? false;
          if (!stillSuppressed && activeRef.current && streamRef.current && !mediaRecorderRef.current) {
            _startRecorder(streamRef.current);
          }
        }, 50);
      }
    };

    mr.stop();
  }, [_startRecorder, minUtteranceDurationMs, suppressRef]);

  const _tick = useCallback(() => {
    if (!activeRef.current || !analyserRef.current) return;

    const bufferLength = analyserRef.current.fftSize;
    const dataArray = new Float32Array(bufferLength);
    analyserRef.current.getFloatTimeDomainData(dataArray);

    // Compute RMS energy
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
      sum += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(sum / bufferLength);

    const suppressed = suppressRef?.current ?? false;
    const now = Date.now();

    if (suppressed) {
      // While suppressed (AI processing or speaking):
      // 1. Stop recorder so mic doesn't capture speaker echo
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.ondataavailable = null;
        mediaRecorderRef.current.onstop = null;
        if (mediaRecorderRef.current.state !== 'inactive') {
          try { mediaRecorderRef.current.stop(); } catch (_) {}
        }
        mediaRecorderRef.current = null;
      }
      chunksRef.current = [];
      isSpeakingRef.current = false;
      silenceStartRef.current = null;
      utteranceStartRef.current = null;
      lastVoiceAtRef.current = null;
      wasSuppressedRef.current = true;

      animFrameRef.current = requestAnimationFrame(_tick);
      return;
    }

    // When suppressed transitions from true -> false (AI finished speaking):
    // Start a fresh MediaRecorder so chunk 0 has a valid EBML header
    if (wasSuppressedRef.current) {
      wasSuppressedRef.current = false;
      cooldownUntilRef.current = now + postSuppressCooldownMs;
      if (activeRef.current && streamRef.current && !mediaRecorderRef.current) {
        _startRecorder(streamRef.current);
      }
    }

    // Ignore the echo tail of the AI's own voice right after it stops speaking
    if (now < cooldownUntilRef.current) {
      animFrameRef.current = requestAnimationFrame(_tick);
      return;
    }

    // Ensure recorder is active when listening
    if (!mediaRecorderRef.current && streamRef.current) {
      _startRecorder(streamRef.current);
    }

    // Prevent buffer from growing indefinitely if user is idle for > 30s
    if (!isSpeakingRef.current && chunksRef.current.length > 300) {
      _startRecorder(streamRef.current);
    }

    if (rms >= speakThreshold) {
      // Speaking
      if (!isSpeakingRef.current) {
        isSpeakingRef.current = true;
        utteranceStartRef.current = now;
        lastVoiceAtRef.current = now;
        silenceStartRef.current = null;
        console.log(`[VAD] 🗣️ Speech detected — RMS: ${rms.toFixed(4)} — ${new Date().toISOString()}`);
        if (onSpeechStartRef.current) onSpeechStartRef.current();
      }
      lastVoiceAtRef.current = now;
      silenceStartRef.current = null;
    } else if (rms < silenceThreshold) {
      // Silence
      if (isSpeakingRef.current) {
        if (silenceStartRef.current === null) {
          silenceStartRef.current = now;
          console.log(`[VAD] 🤫 Pause detected — waiting ${silenceDurationMs}ms to confirm end`);
        } else if (now - silenceStartRef.current >= silenceDurationMs) {
          // Sustained silence after speech → fire
          console.log(`[VAD] 🔇 Silence confirmed (${now - silenceStartRef.current}ms) — finalizing utterance`);
          _flushAndFire();
          silenceStartRef.current = null;
        }
      }
    }

    animFrameRef.current = requestAnimationFrame(_tick);
  }, [speakThreshold, silenceThreshold, silenceDurationMs, postSuppressCooldownMs, suppressRef, _flushAndFire, _startRecorder]);

  const startListening = useCallback(async () => {
    if (activeRef.current) return;
    stopEverything();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      streamRef.current = stream;
      activeRef.current = true;
      wasSuppressedRef.current = false;

      // Build audio graph
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.3;
      source.connect(analyser);
      analyserRef.current = analyser;

      chunksRef.current = [];
      _startRecorder(stream);
      animFrameRef.current = requestAnimationFrame(_tick);

      console.log('[VAD] ✅ Microphone active — VAD running');
    } catch (err) {
      console.error('[VAD] Failed to start microphone:', err);
      stopEverything();
      throw err;
    }
  }, [stopEverything, _startRecorder, _tick]);

  const stopListening = useCallback(() => {
    console.log('[VAD] 🛑 Stopping microphone and VAD');
    stopEverything();
  }, [stopEverything]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopEverything();
    };
  }, [stopEverything]);

  return { startListening, stopListening };
}
