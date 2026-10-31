import { normalizeEmotion } from '../../utils/emotionLabels';

/**
 * emotionPresets.js — Centralized ARKit Morph Target Presets
 * ===========================================================
 *
 * Two naming conventions are supported:
 *
 * ① FaceCap (facecap.glb) — Apple ARKit suffix _L / _R
 *     browDown_L, eyeBlink_L, mouthSmile_L …
 *
 * ② MetaHuman (metaHumanHead.fbx) — Unreal Left / Right suffix
 *     browDownLeft, eyeBlinkLeft, mouthSmileLeft …
 *
 * Call getEmotionPreset(emotion) for FaceCap (default)
 * Call getEmotionPreset(emotion, 'metahuman') for MetaHuman
 */

export const EMOTION_PRESETS = {
  neutral: {},

  happy: {
    mouthSmile_L: 0.75,
    mouthSmile_R: 0.75,
    cheekSquint_L: 0.4,
    cheekSquint_R: 0.4,
    mouthDimple_L: 0.25,
    mouthDimple_R: 0.25,
    eyeSquint_L: 0.2,
    eyeSquint_R: 0.2,
  },

  sad: {
    mouthFrown_L: 0.65,
    mouthFrown_R: 0.65,
    browInnerUp: 0.65,
    browDown_L: 0.25,
    browDown_R: 0.25,
    mouthRollLower: 0.25,
    mouthShrugLower: 0.3,
  },

  angry: {
    browDown_L: 0.85,
    browDown_R: 0.85,
    eyeSquint_L: 0.5,
    eyeSquint_R: 0.5,
    noseSneer_L: 0.5,
    noseSneer_R: 0.5,
    mouthPress_L: 0.4,
    mouthPress_R: 0.4,
    jawForward: 0.25,
  },

  surprised: {
    eyeWide_L: 0.85,
    eyeWide_R: 0.85,
    browInnerUp: 0.8,
    browOuterUp_L: 0.75,
    browOuterUp_R: 0.75,
    jawOpen: 0.35,
    mouthFunnel: 0.25,
  },

  fear: {
    eyeWide_L: 0.8,
    eyeWide_R: 0.8,
    browInnerUp: 0.85,
    browDown_L: 0.4,
    browDown_R: 0.4,
    mouthStretch_L: 0.5,
    mouthStretch_R: 0.5,
    jawOpen: 0.25,
  },

  fearful: {
    eyeWide_L: 0.8,
    eyeWide_R: 0.8,
    browInnerUp: 0.85,
    browDown_L: 0.4,
    browDown_R: 0.4,
    mouthStretch_L: 0.5,
    mouthStretch_R: 0.5,
    jawOpen: 0.25,
  },

  disgust: {
    noseSneer_L: 0.85,
    noseSneer_R: 0.85,
    mouthFrown_L: 0.5,
    mouthFrown_R: 0.5,
    browDown_L: 0.6,
    browDown_R: 0.6,
    mouthShrugUpper: 0.4,
    cheekSquint_L: 0.35,
    cheekSquint_R: 0.35,
  },

  // Conversational aliases
  concerned: {
    browInnerUp: 0.5,
    mouthFrown_L: 0.3,
    mouthFrown_R: 0.3,
    eyeSquint_L: 0.15,
    eyeSquint_R: 0.15,
  },

  thinking: {
    browInnerUp: 0.35,
    browDown_R: 0.4,
    mouthPress_L: 0.3,
    eyeLookUp_L: 0.3,
    eyeLookUp_R: 0.3,
  },

  empathetic: {
    mouthSmile_L: 0.35,
    mouthSmile_R: 0.35,
    browInnerUp: 0.35,
    cheekSquint_L: 0.2,
    cheekSquint_R: 0.2,
  },

  compassion: {
    mouthSmile_L: 0.4,
    mouthSmile_R: 0.4,
    browInnerUp: 0.4,
    cheekSquint_L: 0.25,
    cheekSquint_R: 0.25,
  },

  supportive: {
    mouthSmile_L: 0.5,
    mouthSmile_R: 0.5,
    browInnerUp: 0.25,
    cheekSquint_L: 0.3,
    cheekSquint_R: 0.3,
  },

  encouraging: {
    mouthSmile_L: 0.6,
    mouthSmile_R: 0.6,
    browOuterUp_L: 0.25,
    browOuterUp_R: 0.25,
    cheekSquint_L: 0.35,
    cheekSquint_R: 0.35,
  },
};

// ─── MetaHuman Emotion Presets (Left / Right suffix — Unreal naming) ────────

export const MH_EMOTION_PRESETS = {
  neutral: {},

  happy: {
    mouthSmileLeft:    0.75,
    mouthSmileRight:   0.75,
    cheekSquintLeft:   0.4,
    cheekSquintRight:  0.4,
    mouthDimpleLeft:   0.25,
    mouthDimpleRight:  0.25,
    eyeSquintLeft:     0.2,
    eyeSquintRight:    0.2,
  },

  sad: {
    mouthFrownLeft:    0.65,
    mouthFrownRight:   0.65,
    browInnerUp:       0.65,
    browDownLeft:      0.25,
    browDownRight:     0.25,
    mouthRollLower:    0.25,
    mouthShrugLower:   0.3,
  },

  angry: {
    browDownLeft:      0.85,
    browDownRight:     0.85,
    eyeSquintLeft:     0.5,
    eyeSquintRight:    0.5,
    noseSneerLeft:     0.5,
    noseSneerRight:    0.5,
    mouthPressLeft:    0.4,
    mouthPressRight:   0.4,
    jawForward:        0.25,
  },

  surprised: {
    eyeWideLeft:       0.85,
    eyeWideRight:      0.85,
    browInnerUp:       0.8,
    browOuterUpLeft:   0.75,
    browOuterUpRight:  0.75,
    jawOpen:           0.35,
    mouthFunnel:       0.25,
  },

  fear: {
    eyeWideLeft:       0.8,
    eyeWideRight:      0.8,
    browInnerUp:       0.85,
    browDownLeft:      0.4,
    browDownRight:     0.4,
    mouthStretchLeft:  0.5,
    mouthStretchRight: 0.5,
    jawOpen:           0.25,
  },

  fearful: {
    eyeWideLeft:       0.8,
    eyeWideRight:      0.8,
    browInnerUp:       0.85,
    browDownLeft:      0.4,
    browDownRight:     0.4,
    mouthStretchLeft:  0.5,
    mouthStretchRight: 0.5,
    jawOpen:           0.25,
  },

  disgust: {
    noseSneerLeft:     0.85,
    noseSneerRight:    0.85,
    mouthFrownLeft:    0.5,
    mouthFrownRight:   0.5,
    browDownLeft:      0.6,
    browDownRight:     0.6,
    mouthShrugUpper:   0.4,
    cheekSquintLeft:   0.35,
    cheekSquintRight:  0.35,
  },

  concerned: {
    browInnerUp:       0.5,
    mouthFrownLeft:    0.3,
    mouthFrownRight:   0.3,
    eyeSquintLeft:     0.15,
    eyeSquintRight:    0.15,
  },

  thinking: {
    browInnerUp:       0.35,
    browDownRight:     0.4,
    mouthPressLeft:    0.3,
    eyeLookUpLeft:     0.3,
    eyeLookUpRight:    0.3,
  },

  empathetic: {
    mouthSmileLeft:    0.35,
    mouthSmileRight:   0.35,
    browInnerUp:       0.35,
    cheekSquintLeft:   0.2,
    cheekSquintRight:  0.2,
  },

  compassion: {
    mouthSmileLeft:    0.4,
    mouthSmileRight:   0.4,
    browInnerUp:       0.4,
    cheekSquintLeft:   0.25,
    cheekSquintRight:  0.25,
  },

  supportive: {
    mouthSmileLeft:    0.5,
    mouthSmileRight:   0.5,
    browInnerUp:       0.25,
    cheekSquintLeft:   0.3,
    cheekSquintRight:  0.3,
  },

  encouraging: {
    mouthSmileLeft:    0.6,
    mouthSmileRight:   0.6,
    browOuterUpLeft:   0.25,
    browOuterUpRight:  0.25,
    cheekSquintLeft:   0.35,
    cheekSquintRight:  0.35,
  },
};

// ─── Sky Emotion Presets (ARKit Left / Right suffix — stylized GLB) ────────
export const SKY_EMOTION_PRESETS = {
  ...MH_EMOTION_PRESETS,
  happy: {
    ...MH_EMOTION_PRESETS.happy,
    mouthSmileLeft: 0.8,
    mouthSmileRight: 0.8,
  },
};

// ─── Tufrilla Emotion Presets (VRM + ARKit blendshapes) ────────────────────
export const TUFRILLA_EMOTION_PRESETS = {
  neutral: {
    neutral: 1.0,
  },

  happy: {
    mouthSmileLeft:   0.75,
    mouthSmileRight:  0.75,
    cheekSquintLeft:  0.35,
    cheekSquintRight: 0.35,
    Joy:              0.8,
    joy:              0.8,
    Fun:              0.4,
    fun:              0.4,
  },

  sad: {
    mouthFrownLeft:   0.65,
    mouthFrownRight:  0.65,
    browInnerUp:      0.65,
    browDownLeft:     0.25,
    browDownRight:    0.25,
    Sorrow:           0.8,
    sorrow:           0.8,
  },

  angry: {
    browDownLeft:     0.85,
    browDownRight:    0.85,
    mouthPressLeft:   0.4,
    mouthPressRight:  0.4,
    Angry:            0.85,
    angry:            0.85,
    'Brows down':     0.6,
  },

  surprised: {
    eyeWideLeft:      0.85,
    eyeWideRight:     0.85,
    browInnerUp:      0.8,
    jawOpen:          0.35,
    mouthFunnel:      0.25,
    'Brows up':       0.7,
    fun:              0.3,
  },

  fear: {
    eyeWideLeft:      0.8,
    eyeWideRight:     0.8,
    browInnerUp:      0.85,
    browDownLeft:     0.35,
    browDownRight:    0.35,
    mouthStretchLeft: 0.45,
    mouthStretchRight:0.45,
    Sorrow:           0.5,
    sorrow:           0.5,
  },

  fearful: {
    eyeWideLeft:      0.8,
    eyeWideRight:     0.8,
    browInnerUp:      0.85,
    browDownLeft:     0.35,
    browDownRight:    0.35,
    mouthStretchLeft: 0.45,
    mouthStretchRight:0.45,
    Sorrow:           0.5,
    sorrow:           0.5,
  },

  disgust: {
    noseSneerLeft:    0.7,
    noseSneerRight:   0.7,
    mouthFrownLeft:   0.5,
    mouthFrownRight:  0.5,
    browDownLeft:     0.55,
    browDownRight:    0.55,
    Angry:            0.4,
    angry:            0.4,
  },

  concerned: {
    browInnerUp:      0.5,
    mouthFrownLeft:   0.3,
    mouthFrownRight:  0.3,
    Sorrow:           0.35,
    sorrow:           0.35,
  },

  thinking: {
    browInnerUp:      0.35,
    browDownRight:    0.4,
    mouthPressLeft:   0.3,
    'Look Up':        0.45,
    lookup:           0.45,
  },

  empathetic: {
    mouthSmileLeft:   0.35,
    mouthSmileRight:  0.35,
    browInnerUp:      0.35,
    Joy:              0.35,
    joy:              0.35,
  },

  compassion: {
    mouthSmileLeft:   0.4,
    mouthSmileRight:  0.4,
    browInnerUp:      0.4,
    Joy:              0.4,
    joy:              0.4,
  },

  supportive: {
    mouthSmileLeft:   0.5,
    mouthSmileRight:  0.5,
    Joy:              0.5,
    joy:              0.5,
  },

  encouraging: {
    mouthSmileLeft:   0.6,
    mouthSmileRight:  0.6,
    Joy:              0.6,
    joy:              0.6,
  },
};

/**
 * Returns the morph preset for an emotion name, with neutral fallback.
 * @param {string} emotion
 * @param {'facecap'|'metahuman'|'sky'|'sky2'|'tufrilla'} [avatarType='sky']
 * @returns {Record<string, number>}
 */
export function getEmotionPreset(emotion, avatarType = 'sky') {
  let presets = SKY_EMOTION_PRESETS;
  if (avatarType === 'sky' || avatarType === 'sky2') presets = SKY_EMOTION_PRESETS;
  else if (avatarType === 'tufrilla') presets = TUFRILLA_EMOTION_PRESETS;
  else if (avatarType === 'metahuman') presets = MH_EMOTION_PRESETS;
  else if (avatarType === 'facecap') presets = EMOTION_PRESETS;

  if (!emotion) return presets.neutral || {};
  return presets[normalizeEmotion(emotion, presets)] || presets.neutral || {};
}

