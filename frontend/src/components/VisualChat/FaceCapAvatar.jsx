import { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import { KTX2Loader } from 'three-stdlib';
import { getEmotionPreset } from './emotionPresets';
import { getSpeechMorphs } from './audioAnalyser';

/**
 * FaceCapAvatar — Real-time 3D Human Avatar using public/models/facecap.glb
 * =========================================================================
 *
 * Renders the FaceCap head mesh with 52 Apple ARKit blendshapes.
 *
 * Features:
 *   - Centralized Emotion Presets (Neutral, Happy, Sad, Angry, Surprised, Fear, Disgust)
 *   - Audio-driven Lip Sync (real-time jaw & mouth shapes from TTS audio)
 *   - Emotion + Speech Blending: Final = clamp(Emotion + Speech + Idle, 0..1)
 *   - Natural Idle Behaviour:
 *       • Procedural blinking with randomized interval (2.5s - 5.5s)
 *       • Subtle eye saccades / micro-gaze movement
 *       • Gentle organic head sway and breathing
 *   - 60+ FPS performance: all morph calculations in Three.js useFrame loop
 */
export default function FaceCapAvatar({
  emotion = 'neutral',
  isSpeaking = false,
  status = 'idle',
}) {
  const gl = useThree((state) => state.gl);
  const headGroupRef = useRef();
  const morphMeshRef = useRef(null);

  // Configure GLTF loader with KTX2 basis transcoder and Meshopt
  const gltf = useGLTF(
    '/models/facecap.glb',
    true,
    true,
    (loader) => {
      const ktx2Loader = new KTX2Loader().setTranscoderPath('/basis/').detectSupport(gl);
      loader.setKTX2Loader(ktx2Loader);
    }
  );

  // Clone scene so multiple instances (e.g. PiP or full mode) don't collide
  const scene = useMemo(() => gltf.scene.clone(true), [gltf.scene]);

  // Find the mesh with ARKit blendshapes and log available morph targets
  useEffect(() => {
    if (!scene) return;

    let targetMesh = null;
    scene.traverse((child) => {
      if (child.isMesh && child.morphTargetDictionary) {
        targetMesh = child;
        console.log(
          `[FaceCapAvatar] Loaded morph mesh "${child.name}":`,
          `${Object.keys(child.morphTargetDictionary).length} ARKit targets`,
          child.morphTargetDictionary
        );
      }
    });

    morphMeshRef.current = targetMesh;
  }, [scene]);

  // ── Idle Animation State Refs (Persistent across frames) ──────────────────
  const idleState = useRef({
    // Blinking
    nextBlinkTime: 2.0,
    blinkDuration: 0.22,
    blinkStartTime: 0,
    isBlinking: false,
    blinkVal: 0,

    // Gaze Saccades
    nextGazeTime: 1.5,
    targetGazeX: 0,
    targetGazeY: 0,
    currentGazeX: 0,
    currentGazeY: 0,
  });

  // ── Main Render Loop: Blending, Interpolation & Idle Behaviors ─────────────
  useFrame((state, delta) => {
    const time = state.clock.getElapsedTime();
    const idle = idleState.current;
    const mesh = morphMeshRef.current;

    // ── 1. Update Idle Blinking ─────────────────────────────────────────────
    if (!idle.isBlinking && time >= idle.nextBlinkTime) {
      idle.isBlinking = true;
      idle.blinkStartTime = time;
      idle.blinkDuration = 0.18 + Math.random() * 0.08; // 180-260 ms
    }

    if (idle.isBlinking) {
      const elapsedBlink = time - idle.blinkStartTime;
      const progress = elapsedBlink / idle.blinkDuration;
      if (progress >= 1.0) {
        idle.isBlinking = false;
        idle.blinkVal = 0;
        // Schedule next blink in 2.5 to 5.5 seconds (occasional double blink)
        const isDoubleBlink = Math.random() < 0.2;
        idle.nextBlinkTime = time + (isDoubleBlink ? 0.35 : 2.5 + Math.random() * 3.0);
      } else {
        // Quick close (first 40%), smooth open (remaining 60%)
        if (progress < 0.4) {
          idle.blinkVal = progress / 0.4;
        } else {
          idle.blinkVal = 1.0 - (progress - 0.4) / 0.6;
        }
      }
    }

    // ── 2. Update Idle Eye Saccades (Gentle, focused forward gaze) ─────────
    if (time >= idle.nextGazeTime) {
      idle.targetGazeX = (Math.random() - 0.5) * 0.12; // subtle gaze shift
      idle.targetGazeY = (Math.random() - 0.5) * 0.08; // subtle gaze shift
      idle.nextGazeTime = time + 2.5 + Math.random() * 3.5;
    }
    const gazeLerp = 1 - Math.exp(-6 * delta);
    idle.currentGazeX = THREE.MathUtils.lerp(idle.currentGazeX, idle.targetGazeX, gazeLerp);
    idle.currentGazeY = THREE.MathUtils.lerp(idle.currentGazeY, idle.targetGazeY, gazeLerp);

    // ── 3. Subtle Head Movement & Breathing Sway ────────────────────────────
    if (headGroupRef.current) {
      const listeningTilt = status === 'listening' ? 0.03 : 0;
      const swayX = Math.sin(time * 0.6) * 0.015 + idle.currentGazeY * 0.02;
      const swayY = Math.sin(time * 0.45) * 0.025 + idle.currentGazeX * 0.03 + listeningTilt;
      const swayZ = Math.sin(time * 0.35) * 0.008;
      const breatheY = -0.2 + Math.sin(time * 1.3) * 0.004;

      headGroupRef.current.rotation.x = THREE.MathUtils.lerp(headGroupRef.current.rotation.x, swayX, 0.08);
      headGroupRef.current.rotation.y = THREE.MathUtils.lerp(headGroupRef.current.rotation.y, swayY, 0.08);
      headGroupRef.current.rotation.z = THREE.MathUtils.lerp(headGroupRef.current.rotation.z, swayZ, 0.08);
      headGroupRef.current.position.y = THREE.MathUtils.lerp(headGroupRef.current.position.y, breatheY, 0.08);
    }

    // ── 4. Morph Target Influences: Emotion + Speech + Idle ─────────────────
    if (!mesh || !mesh.morphTargetDictionary || !mesh.morphTargetInfluences) return;

    const dict = mesh.morphTargetDictionary;
    const influences = mesh.morphTargetInfluences;

    // Retrieve active contributions
    const emotionWeights = getEmotionPreset(emotion);
    const speechWeights = getSpeechMorphs(isSpeaking, time);

    // Idle morph contributions (blinking + gentle gaze)
    // Add subtle baseline eyeLookOut to make eyes look naturally forward instead of converging
    const idleWeights = {
      eyeBlink_L: idle.blinkVal,
      eyeBlink_R: idle.blinkVal,
      eyeLookOut_L: 0.12,
      eyeLookOut_R: 0.12,
    };

    const gazeAmp = 0.35;
    if (idle.currentGazeX > 0) {
      idleWeights.eyeLookOut_R = (idleWeights.eyeLookOut_R || 0) + idle.currentGazeX * gazeAmp;
      idleWeights.eyeLookIn_L = idle.currentGazeX * gazeAmp;
    } else {
      idleWeights.eyeLookOut_L = (idleWeights.eyeLookOut_L || 0) + (-idle.currentGazeX * gazeAmp);
      idleWeights.eyeLookIn_R = -idle.currentGazeX * gazeAmp;
    }

    if (idle.currentGazeY > 0) {
      idleWeights.eyeLookUp_L = idle.currentGazeY * gazeAmp;
      idleWeights.eyeLookUp_R = idle.currentGazeY * gazeAmp;
    } else {
      idleWeights.eyeLookDown_L = -idle.currentGazeY * gazeAmp;
      idleWeights.eyeLookDown_R = -idle.currentGazeY * gazeAmp;
    }

    // Frame-rate independent exponential smoothing factor (~12-16 gives natural fluid transitions)
    const lerpFactor = 1 - Math.exp(-14 * delta);

    for (const [targetName, targetIndex] of Object.entries(dict)) {
      const eVal = emotionWeights[targetName] || 0;
      const sVal = speechWeights[targetName] || 0;
      const iVal = idleWeights[targetName] || 0;

      // Critical Rule: Blend Emotion + Speech + Idle without overwriting, clamped to [0, 1]
      const targetValue = THREE.MathUtils.clamp(eVal + sVal + iVal, 0.0, 1.0);

      // Smooth interpolation
      const currentValue = influences[targetIndex] || 0;
      influences[targetIndex] = THREE.MathUtils.lerp(currentValue, targetValue, lerpFactor);
    }
  });

  return (
    <group ref={headGroupRef} position={[0, -0.2, 0.35]}>
      <primitive object={scene} />
    </group>
  );
}
