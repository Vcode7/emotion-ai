/**
 * emotionLabels — normalise emotion labels coming from the backend.
 *
 * The backend uses several vocabularies (user emotion "Happy"/"Content",
 * DMESR avatar expressions "joyful"/"alert", voice labels "warm_positive",
 * LLM labels "empathetic"/"listening", …). Each UI map only knows a subset,
 * so unmapped labels used to fall through to neutral. Use normalizeEmotion()
 * before every lookup.
 */

// Synonyms → canonical label.
const ALIASES = {
  joy: 'happy',
  joyful: 'happy',
  happiness: 'happy',
  content: 'happy',
  excited: 'happy',
  warm_positive: 'happy',
  alert: 'surprised',
  surprise: 'surprised',
  calm_reassuring: 'compassion',
  compassionate: 'compassion',
  caring: 'empathetic',
  listening: 'neutral',
  speaking: 'neutral',
  calm: 'neutral',
  stressed: 'concerned',
  worried: 'concerned',
  anxious: 'concerned',
  fear: 'fearful',
  anger: 'angry',
  sadness: 'sad',
};

// When a map lacks the canonical label, try the closest one it does have.
const FALLBACKS = {
  empathetic: 'compassion',
  compassion: 'empathetic',
  fearful: 'fear',
  encouraging: 'happy',
  supportive: 'compassion',
  concerned: 'sad',
  surprised: 'happy',
};

/**
 * @param {string} label - Any emotion label from the backend.
 * @param {object|Set} [supported] - Map (object keys) or Set of labels the caller can render.
 * @returns {string} A lowercase label; when `supported` is given, one of its keys (or 'neutral').
 */
export function normalizeEmotion(label, supported) {
  let key = String(label ?? 'neutral').toLowerCase().trim().replace(/\s+/g, '_');
  key = ALIASES[key] || key || 'neutral';
  if (!supported) return key;

  const has = (k) =>
    supported instanceof Set ? supported.has(k) : Object.prototype.hasOwnProperty.call(supported, k);
  if (has(key)) return key;
  const fallback = FALLBACKS[key];
  if (fallback && has(fallback)) return fallback;
  return 'neutral';
}
