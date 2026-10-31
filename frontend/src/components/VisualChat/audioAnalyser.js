/**
 * audioAnalyser.js — Web Audio API Analyser & Lip Sync Controller
 * ================================================================
 *
 * Extracts real-time audio frequency data from the active TTS audio element
 * and maps spectral energy into FaceCap ARKit morph targets:
 *   - jawOpen
 *   - mouthFunnel
 *   - mouthPucker
 *   - mouthUpperUp_L / mouthUpperUp_R
 *   - mouthLowerDown_L / mouthLowerDown_R
 *   - mouthStretch_L / mouthStretch_R
 *
 * Lip-sync design — "open → close → open" pattern:
 *   - Uses a biphasic syllable wave that crosses ZERO between syllables,
 *     ensuring the jaw closes between words rather than staying open.
 *   - When real audio data is available from the AnalyserNode, the jaw
 *     tracks actual volume closely (low latency smoothing = 0.15).
 *   - When audio data is silent / unavailable, the cadence wave generates
 *     realistic-looking articulation at natural speech rhythm (4–6 Hz).
 */

let audioCtx = null;
let analyserNode = null;
let dataArray = null;
const connectedElements = new WeakSet();

/**
 * Lazily initialize the AudioContext and AnalyserNode.
 */
function getAudioContext() {
  if (!audioCtx && typeof window !== 'undefined') {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
      // Expose globally so playAudio can resume it if suspended by browser autoplay policy
      window._emotionAIAudioCtx = audioCtx;
      analyserNode = audioCtx.createAnalyser();
      // Higher fftSize = better frequency resolution
      analyserNode.fftSize = 512;
      // Low smoothing = fast response to audio changes (quicker open→close)
      analyserNode.smoothingTimeConstant = 0.15;
      dataArray = new Uint8Array(analyserNode.frequencyBinCount);
      // Always connect analyser to destination so audio reaches speakers
      analyserNode.connect(audioCtx.destination);
    }
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume().catch(() => {});
  }
  return { audioCtx, analyserNode, dataArray };
}

/**
 * Connect an HTMLAudioElement to the Web Audio Analyser.
 * Uses a WeakSet to ensure each element is hooked up only once.
 *
 * @param {HTMLAudioElement} audioElement
 */
export function attachAudioElement(audioElement) {
  if (!audioElement || typeof window === 'undefined') return;
  const { audioCtx: ctx, analyserNode: analyser } = getAudioContext();
  if (!ctx || !analyser) return;

  if (connectedElements.has(audioElement)) return;

  try {
    const source = ctx.createMediaElementSource(audioElement);
    source.connect(analyser);
    // analyser is already connected to ctx.destination in getAudioContext()
    connectedElements.add(audioElement);
  } catch (err) {
    // If already connected or cross-origin restrictions apply, log once gracefully
    console.warn('[audioAnalyser] Audio connection notice:', err.message || err);
  }
}

/**
 * Calculate current speech blendshapes.
 *
 * Lip-sync strategy:
 *   1. If real audio energy detected → scale jaw directly from volume (fast, accurate)
 *   2. If no audio data → use a biphasic wave that crosses zero so mouth CLOSES
 *      between syllables rather than staying open.
 *
 * @param {boolean} isSpeaking — true when TTS is playing
 * @param {number} time — elapsed time in seconds (for cadence)
 * @param {'facecap'|'metahuman'} [avatarType='facecap'] — naming convention
 * @returns {Record<string, number>}
 */
export function getSpeechMorphs(isSpeaking, time = 0, avatarType = 'sky') {
  const isARKitLR = avatarType === 'sky' || avatarType === 'sky2' || avatarType === 'metahuman' || avatarType === 'tufrilla';

  // ── Closed-mouth zero targets ────────────────────────────────────────
  if (!isSpeaking) {
    if (isARKitLR) {
      const base = {
        jawOpen: 0,
        mouthFunnel: 0,
        mouthPucker: 0,
        mouthUpperUpLeft: 0,
        mouthUpperUpRight: 0,
        mouthLowerDownLeft: 0,
        mouthLowerDownRight: 0,
        mouthStretchLeft: 0,
        mouthStretchRight: 0,
      };
      if (avatarType === 'sky' || avatarType === 'sky2') {
        base.aa = 0; base.oh = 0; base.ou = 0;
      }
      if (avatarType === 'tufrilla') {
        base.A = 0; base.O = 0; base.a = 0; base.o = 0;
      }
      return base;
    }
    return {
      jawOpen: 0,
      mouthFunnel: 0,
      mouthPucker: 0,
      mouthUpperUp_L: 0,
      mouthUpperUp_R: 0,
      mouthLowerDown_L: 0,
      mouthLowerDown_R: 0,
      mouthStretch_L: 0,
      mouthStretch_R: 0,
    };
  }

  let lowEnergy = 0;
  let midEnergy = 0;
  let highEnergy = 0;
  let totalVolume = 0;

  if (analyserNode && dataArray) {
    analyserNode.getByteFrequencyData(dataArray);

    const binCount = dataArray.length;

    // Low bins (approx 0–500 Hz): jaw/vowel fundamentals
    const lowEnd = Math.min(10, binCount);
    let lowSum = 0;
    for (let i = 0; i < lowEnd; i++) lowSum += dataArray[i];
    lowEnergy = lowSum / (lowEnd * 255);

    // Mid bins (approx 500–2000 Hz): vowel formants F1/F2
    const midEnd = Math.min(30, binCount);
    let midSum = 0;
    for (let i = lowEnd; i < midEnd; i++) midSum += dataArray[i];
    midEnergy = midSum / ((midEnd - lowEnd) * 255);

    // High bins (approx 2000–6000 Hz): consonants / fricatives / sibilants
    const highEnd = Math.min(60, binCount);
    let highSum = 0;
    for (let i = midEnd; i < highEnd; i++) highSum += dataArray[i];
    highEnergy = highSum / ((highEnd - midEnd) * 255);

    let sum = 0;
    for (let i = 0; i < binCount; i++) sum += dataArray[i];
    totalVolume = sum / (binCount * 255);
  }

  // ── Biphasic syllable cadence ────────────────────────────────────────
  // Crosses ZERO between peaks so mouth CLOSES between syllables.
  // Primary rate ~5 Hz (natural syllable rate), modulated at ~2 Hz (stress).
  const syllableRaw = Math.sin(time * 31.4) * 0.6  // ~5 Hz primary
                    + Math.sin(time * 18.8) * 0.35  // ~3 Hz secondary
                    + Math.sin(time * 12.6) * 0.2;  // ~2 Hz modulation
  // syllableRaw is in ~[-1, +1]; only keep POSITIVE portion (mouth open phase)
  const syllableOpen = Math.max(0, syllableRaw);      // 0 = closed, >0 = open

  // ── Route between real audio vs cadence ──────────────────────────────
  const hasAudio = totalVolume > 0.025;

  let jaw, funnel, pucker, stretch, lipUp, lipDown;

  if (hasAudio) {
    // Real audio path: track volume closely
    const speechIntensity = Math.min(1.0, totalVolume * 3.2 + lowEnergy * 0.8);
    jaw     = Math.min(0.65, speechIntensity * 0.60 + lowEnergy * 0.30);
    funnel  = Math.min(0.45, midEnergy * 0.55 + Math.sin(time * 11) * 0.10 * speechIntensity);
    pucker  = Math.min(0.35, midEnergy > 0.25 ? midEnergy * 0.38 : 0);
    stretch = Math.min(0.42, highEnergy * 0.65 + (jaw > 0.3 ? 0.10 : 0));
    lipUp   = Math.min(0.32, jaw * 0.38);
    lipDown = Math.min(0.32, jaw * 0.38);
  } else {
    // Cadence fallback: use biphasic wave (closes between syllables)
    const intensity = syllableOpen * 0.55;
    jaw     = intensity;
    funnel  = syllableOpen * 0.20;
    pucker  = 0;
    stretch = syllableOpen * 0.15;
    lipUp   = intensity * 0.35;
    lipDown = intensity * 0.35;
  }

  if (isARKitLR) {
    const base = {
      jawOpen:             jaw,
      mouthFunnel:         Math.max(0, funnel),
      mouthPucker:         Math.max(0, pucker),
      mouthUpperUpLeft:    lipUp,
      mouthUpperUpRight:   lipUp,
      mouthLowerDownLeft:  lipDown,
      mouthLowerDownRight: lipDown,
      mouthStretchLeft:    stretch,
      mouthStretchRight:   stretch,
    };
    if (avatarType === 'sky' || avatarType === 'sky2') {
      base.aa = jaw * 0.8;
      base.oh = Math.max(0, funnel) * 0.8;
      base.ou = Math.max(0, pucker) * 0.8;
    }
    if (avatarType === 'tufrilla') {
      base.A = jaw * 0.8;
      base.O = Math.max(0, funnel) * 0.8;
      base.a = jaw * 0.8;
      base.o = Math.max(0, funnel) * 0.8;
    }
    return base;
  }

  return {
    jawOpen:          jaw,
    mouthFunnel:      Math.max(0, funnel),
    mouthPucker:      Math.max(0, pucker),
    mouthUpperUp_L:   lipUp,
    mouthUpperUp_R:   lipUp,
    mouthLowerDown_L: lipDown,
    mouthLowerDown_R: lipDown,
    mouthStretch_L:   stretch,
    mouthStretch_R:   stretch,
  };
}
