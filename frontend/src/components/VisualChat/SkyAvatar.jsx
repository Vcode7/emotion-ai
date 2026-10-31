import { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import * as SkeletonUtils from 'three/examples/jsm/utils/SkeletonUtils.js';
import { getEmotionPreset } from './emotionPresets';
import { getSpeechMorphs } from './audioAnalyser';

/**
 * SkyAvatar — Real-time 3D Stylized Avatar with 66 ARKit & Viseme Blendshapes
 * ============================================================================
 *
 * Loads: /models/sky.glb
 *
 * Blendshape naming: Unreal / ARKit Left/Right convention
 *   eyeBlinkLeft, mouthSmileLeft, browDownLeft, jawOpen, aa, oh, ou …
 *
 * Features:
 *   - Clean portrait framing tuned to match MetaHuman scene camera
 *   - Emotion presets mapped to ARKit morph targets
 *   - Audio-driven lip sync with viseme support
 *   - Procedural blinking, micro-gaze saccades, head sway & breathing
 */
export default function SkyAvatar({
  avatarType = 'sky',
  modelPath,
  emotion = 'neutral',
  isSpeaking = false,
  status = 'idle',
  position = [0, 0, 0],
  modelOffset = { x: 0, y: 0, z: 0 },
  modelRotationY = 0,
  modelZoom = 1.0,
  onLoaded,
}) {
  const avatarGroupRef = useRef();
  const headBoneRef = useRef(null);
  const leftArmRef = useRef(null);
  const rightArmRef = useRef(null);
  const leftForeArmRef = useRef(null);
  const rightForeArmRef = useRef(null);
  const morphMeshesRef = useRef([]);

  const effectiveModelPath = modelPath || (avatarType === 'sky2' ? '/models/sky2.glb' : '/models/sky.glb');
  const gltf = useGLTF(effectiveModelPath);

  // Clone scene with proper skeleton rebinding and calibrate portrait framing
  const scene = useMemo(() => {
    if (!gltf?.scene) return null;
    const cloned = SkeletonUtils.clone(gltf.scene);

    // Frame portrait/bust view: head center is at Y ≈ 1.59m
    // Scale by 4.25 and offset Y by -6.72 to match portrait framing
    cloned.scale.setScalar(4.25);
    cloned.position.set(0, -6.72, 0);

    return cloned;
  }, [gltf.scene]);

  useEffect(() => {
    if (!scene) return;

    const meshes = [];
    let foundHeadBone = null;

    scene.traverse((child) => {
      // Natural resting arm pose (rotate local X by ~75 deg to lower arms down beside torso)
      if (child.name === 'LeftArm') {
        leftArmRef.current = child;
        child.rotation.x = 1.30;
        child.rotation.y = 0.05;
        child.rotation.z = -0.08;
      } else if (child.name === 'RightArm') {
        rightArmRef.current = child;
        child.rotation.x = 1.30;
        child.rotation.y = -0.05;
        child.rotation.z = 0.08;
      } else if (child.name === 'LeftForeArm') {
        leftForeArmRef.current = child;
        child.rotation.x = 0.22;
        child.rotation.z = 0.15;
      } else if (child.name === 'RightForeArm') {
        rightForeArmRef.current = child;
        child.rotation.x = 0.22;
        child.rotation.z = -0.15;
      }

      // Find head bone for subtle natural sway
      if (child.isBone && (child.name.toLowerCase() === 'head' || child.name === 'Head')) {
        foundHeadBone = child;
      }

      if (child.isMesh) {
        child.frustumCulled = false;

          const nameLower = (child.name || '').toLowerCase();

          // Hide redundant Cornea geometry domes (eliminates duplicate reflections and Z-shimmer)
          if (nameLower.includes('cornea')) {
            child.visible = false;
            return;
          }

          if (child.material) {
            child.material = child.material.clone();
            child.material.needsUpdate = true;

            const isEyelash = nameLower.includes('lash') || nameLower.includes('brow');
            const isEyeball = (nameLower.includes('eyeball') || nameLower.includes('eye')) && !isEyelash;

            if (isEyeball) {
              // Flatten micro-normal map to eliminate specular aliasing & sparkling glitter
              if (child.material.normalScale) {
                child.material.normalScale.set(0, 0);
              }
              child.material.normalMap = null;
              child.material.roughnessMap = null;
              child.material.metalnessMap = null;

              // Soft, natural anatomical eye gleam instead of sharp pinpoint glitter
              child.material.roughness = 0.40;
              child.material.metalness = 0.0;
              child.material.envMapIntensity = 0.15;
            } else if (isEyelash) {
              // Natural matte eyelashes (prevent glossy sparkling around eyelids)
              child.material.roughness = 0.90;
              child.material.metalness = 0.0;
              child.material.envMapIntensity = 0.10;
            } else {
              // Skin, hair, and clothing
              child.material.roughness = Math.min(child.material.roughness || 0.65, 0.75);
              child.material.envMapIntensity = 0.35;
            }
          }

        if (child.morphTargetDictionary && child.morphTargetInfluences) {
          meshes.push({
            mesh: child,
            dict: child.morphTargetDictionary,
          });
        }
      }
    });

    headBoneRef.current = foundHeadBone;
    morphMeshesRef.current = meshes;

    if (onLoaded) {
      onLoaded();
    }
  }, [scene, onLoaded]);

  // Idle animation state refs
  const idleState = useRef({
    nextBlinkTime: 2.0,
    blinkDuration: 0.22,
    blinkStartTime: 0,
    isBlinking: false,
    blinkVal: 0,

    nextGazeTime: 1.5,
    targetGazeX: 0,
    targetGazeY: 0,
    currentGazeX: 0,
    currentGazeY: 0,
  });

  useFrame((state, delta) => {
    const time = state.clock.getElapsedTime();
    const idle = idleState.current;

    // 1. Procedural Blinking
    if (!idle.isBlinking && time >= idle.nextBlinkTime) {
      idle.isBlinking = true;
      idle.blinkStartTime = time;
      idle.blinkDuration = 0.18 + Math.random() * 0.08;
    }
    if (idle.isBlinking) {
      const p = (time - idle.blinkStartTime) / idle.blinkDuration;
      if (p >= 1.0) {
        idle.isBlinking = false;
        idle.blinkVal = 0;
        idle.nextBlinkTime = time + (Math.random() < 0.2 ? 0.35 : 2.5 + Math.random() * 3.0);
      } else {
        idle.blinkVal = p < 0.4 ? p / 0.4 : 1.0 - (p - 0.4) / 0.6;
      }
    }

    // 2. Eye Gaze Saccades
    if (time >= idle.nextGazeTime) {
      idle.targetGazeX = (Math.random() - 0.5) * 0.12;
      idle.targetGazeY = (Math.random() - 0.5) * 0.08;
      idle.nextGazeTime = time + 2.5 + Math.random() * 3.5;
    }
    const gLerp = 1 - Math.exp(-6 * delta);
    idle.currentGazeX = THREE.MathUtils.lerp(idle.currentGazeX, idle.targetGazeX, gLerp);
    idle.currentGazeY = THREE.MathUtils.lerp(idle.currentGazeY, idle.targetGazeY, gLerp);

    // 3. Head sway and breathing
    const lt = status === 'listening' ? 0.03 : 0;
    if (avatarGroupRef.current) {
      avatarGroupRef.current.rotation.x = THREE.MathUtils.lerp(
        avatarGroupRef.current.rotation.x,
        Math.sin(time * 0.6) * 0.012 + idle.currentGazeY * 0.02,
        0.08
      );
      avatarGroupRef.current.rotation.y = THREE.MathUtils.lerp(
        avatarGroupRef.current.rotation.y,
        Math.sin(time * 0.45) * 0.02 + idle.currentGazeX * 0.03 + lt,
        0.08
      );
      avatarGroupRef.current.rotation.z = THREE.MathUtils.lerp(
        avatarGroupRef.current.rotation.z,
        Math.sin(time * 0.35) * 0.007,
        0.08
      );
      avatarGroupRef.current.position.y = THREE.MathUtils.lerp(
        avatarGroupRef.current.position.y,
        position[1] + Math.sin(time * 1.3) * 0.004,
        0.08
      );
    }

    // 4. Natural resting arm pose down beside the body (with subtle breathing sway)
    const armBreathe = Math.sin(time * 1.3) * 0.015;
    if (leftArmRef.current) {
      leftArmRef.current.rotation.x = 1.30 + armBreathe;
      leftArmRef.current.rotation.y = 0.05;
      leftArmRef.current.rotation.z = -0.08;
    }
    if (rightArmRef.current) {
      rightArmRef.current.rotation.x = 1.30 + armBreathe;
      rightArmRef.current.rotation.y = -0.05;
      rightArmRef.current.rotation.z = 0.08;
    }
    if (leftForeArmRef.current) {
      leftForeArmRef.current.rotation.x = 0.22;
      leftForeArmRef.current.rotation.z = 0.15;
    }
    if (rightForeArmRef.current) {
      rightForeArmRef.current.rotation.x = 0.22;
      rightForeArmRef.current.rotation.z = -0.15;
    }

    // 5. Morph blending across all meshes
    const meshes = morphMeshesRef.current;
    if (!Array.isArray(meshes) || meshes.length === 0) return;

    const effectiveType = avatarType === 'sky2' ? 'sky2' : 'sky';
    const eW = getEmotionPreset(emotion, effectiveType);
    const sW = getSpeechMorphs(isSpeaking, time, effectiveType);

    const iW = {
      eyeBlinkLeft:   idle.blinkVal,
      eyeBlinkRight:  idle.blinkVal,
      eyeLookOutLeft: 0.1,
      eyeLookOutRight:0.1,
    };
    const ga = 0.35;
    if (idle.currentGazeX > 0) {
      iW.eyeLookOutRight = (iW.eyeLookOutRight || 0) + idle.currentGazeX * ga;
      iW.eyeLookInLeft   = idle.currentGazeX * ga;
    } else {
      iW.eyeLookOutLeft = (iW.eyeLookOutLeft || 0) + (-idle.currentGazeX * ga);
      iW.eyeLookInRight = -idle.currentGazeX * ga;
    }
    if (idle.currentGazeY > 0) {
      iW.eyeLookUpLeft   = idle.currentGazeY * ga;
      iW.eyeLookUpRight  = idle.currentGazeY * ga;
    } else {
      iW.eyeLookDownLeft  = -idle.currentGazeY * ga;
      iW.eyeLookDownRight = -idle.currentGazeY * ga;
    }

    const lfFast = 1 - Math.exp(-26 * delta);
    const lfSlow = 1 - Math.exp(-10 * delta);

    const MOUTH_KEYS = new Set([
      'jawOpen', 'mouthFunnel', 'mouthPucker',
      'mouthUpperUpLeft', 'mouthUpperUpRight',
      'mouthLowerDownLeft', 'mouthLowerDownRight',
      'mouthStretchLeft', 'mouthStretchRight',
      'aa', 'oh', 'ou', 'E', 'ih',
    ]);

    for (const { mesh, dict } of meshes) {
      const influences = mesh.morphTargetInfluences;
      if (!influences) continue;
      for (const [name, idx] of Object.entries(dict)) {
        const target = THREE.MathUtils.clamp((eW[name] || 0) + (sW[name] || 0) + (iW[name] || 0), 0, 1);
        const lf = MOUTH_KEYS.has(name) ? lfFast : lfSlow;
        influences[idx] = THREE.MathUtils.lerp(influences[idx] || 0, target, lf);
      }
    }
  });

  return (
    <group
      position={[modelOffset?.x || 0, modelOffset?.y || 0, modelOffset?.z || 0]}
      rotation={[0, modelRotationY || 0, 0]}
      scale={modelZoom || 1.0}
    >
      <group ref={avatarGroupRef} position={position}>
        <primitive object={scene} />
      </group>
    </group>
  );
}

useGLTF.preload('/models/sky.glb');
useGLTF.preload('/models/sky2.glb');
