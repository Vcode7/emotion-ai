/**
 * speech.js — Web Speech API female voice utilities
 * ==================================================
 * Avatars Sky and Sky2 are female models.
 * This utility guarantees that any browser-based speech synthesis
 * selects a female voice and strictly avoids male voices.
 */

const FEMALE_VOICE_PATTERNS = [
  /zira/i,
  /jenny/i,
  /aria/i,
  /ava/i,
  /emma/i,
  /michelle/i,
  /samantha/i,
  /victoria/i,
  /karen/i,
  /susan/i,
  /hazel/i,
  /catherine/i,
  /linda/i,
  /heera/i,
  /veena/i,
  /female/i,
  /natural.*female/i,
];

const MALE_VOICE_PATTERNS = [
  /\bmale\b/i,
  /\bdavid\b/i,
  /\bmark\b/i,
  /\bgeorge\b/i,
  /\bguy\b/i,
  /\bchristopher\b/i,
  /\bandrew\b/i,
  /\brichard\b/i,
  /\bjames\b/i,
  /\bravi\b/i,
  /\bpaul\b/i,
  /\bsteffan\b/i,
];

function isMaleVoice(name) {
  return MALE_VOICE_PATTERNS.some((pattern) => pattern.test(name));
}

function isFemaleVoice(name) {
  return FEMALE_VOICE_PATTERNS.some((pattern) => pattern.test(name));
}

/**
 * Returns the best available female voice from the browser's SpeechSynthesis engine.
 * Never returns a voice matching known male patterns.
 *
 * @returns {SpeechSynthesisVoice | null}
 */
export function getFemaleVoice() {
  if (typeof window === 'undefined' || !window.speechSynthesis) {
    return null;
  }

  const voices = window.speechSynthesis.getVoices();
  if (!voices || voices.length === 0) {
    return null;
  }

  // 1. Highest priority: English voice matching known female names and not male
  for (const pattern of FEMALE_VOICE_PATTERNS) {
    const match = voices.find(
      (v) => (v.lang.startsWith('en') || v.lang.startsWith('en-')) && pattern.test(v.name) && !isMaleVoice(v.name)
    );
    if (match) return match;
  }

  // 2. Second priority: Any voice matching known female names and not male
  for (const pattern of FEMALE_VOICE_PATTERNS) {
    const match = voices.find((v) => pattern.test(v.name) && !isMaleVoice(v.name));
    if (match) return match;
  }

  // 3. Third priority: English voice that is NOT male
  const nonMaleEn = voices.find(
    (v) => (v.lang.startsWith('en') || v.lang.startsWith('en-')) && !isMaleVoice(v.name)
  );
  if (nonMaleEn) return nonMaleEn;

  // 4. Any voice that is not male
  const nonMale = voices.find((v) => !isMaleVoice(v.name));
  if (nonMale) return nonMale;

  return voices[0] || null;
}
