import { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { useLoader, useFrame } from '@react-three/fiber';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { getEmotionPreset } from './emotionPresets';
import { getSpeechMorphs } from './audioAnalyser';

/**
 * TufrillaAvatar — Real-time 3D VRM Avatar with Spring Bones & Full Facial Expressions
 * =====================================================================================
 *
 * Loads: /models/TufrillaVRM.vrm
 *
 * Features:
 *   - VRM 0.x loader with @pixiv/three-vrm
 *   - Automatic forward rotation and portrait camera framing (bust shot)
 *   - Spring-bone secondary physics (dynamic hair and clothing sway)
 *   - 67 morph targets (ARKit + VRM presets: Joy, Sorrow, Angry, Fun, Visemes)
 *   - Audio-driven lip sync and procedural natural blinking & eye saccades
 */
export default function TufrillaAvatar({
  emotion = 'neutral',
  isSpeaking = false,
  status = 'idle',
  position = [0, 0, 0],
  onLoaded,
}) {
  const avatarGroupRef = useRef();
  const morphMeshesRef = useRef([]);

  // Load VRM model with VRMLoaderPlugin
  const gltf = useLoader(GLTFLoader, '/models/TufrillaVRM.vrm', (loader) => {
    loader.register((parser) => new VRMLoaderPlugin(parser));
  });

  const vrm = gltf.userData.vrm;

  // Position, scale, and orient Tufrilla so head/bust is framed in portrait view
  const scene = useMemo(() => {
    if (!vrm) return gltf.scene;

    // Rotate VRM 0.x model to face the camera (+Z)
    VRMUtils.rotateVRM0(vrm);

    // Frame upper body: height is ~1.42m, head center is at Y ≈ 1.20m
    // Scale by 3.15 and offset Y by -3.75 to frame hair, face, and shoulders perfectly
    vrm.scene.scale.setScalar(3.15);
    vrm.scene.position.set(0, -3.75, 0);

    return vrm.scene;
  }, [vrm, gltf.scene]);

  useEffect(() => {
    if (!scene) return;

    const meshes = [];

    scene.traverse((child) => {
      if (child.isMesh) {
        child.frustumCulled = false;

        // Enhance material properties for anime/stylized aesthetic
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => { m.needsUpdate = true; });
          } else {
            child.material.needsUpdate = true;
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

    // 1. Update VRM spring bones & secondary animation
    if (vrm) {
      vrm.update(delta);

      // Natural resting arm pose (lowered from default T-pose)
      if (vrm.humanoid) {
        const leftUpperArm = vrm.humanoid.getNormalizedBoneNode('leftUpperArm');
        const rightUpperArm = vrm.humanoid.getNormalizedBoneNode('rightUpperArm');
        if (leftUpperArm) {
          leftUpperArm.rotation.z = -1.25;
          leftUpperArm.rotation.x = 0.12;
        }
        if (rightUpperArm) {
          rightUpperArm.rotation.z = 1.25;
          rightUpperArm.rotation.x = 0.12;
        }
      }
    }

    // 2. Procedural Blinking
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

    // 3. Gaze saccades
    if (time >= idle.nextGazeTime) {
      idle.targetGazeX = (Math.random() - 0.5) * 0.12;
      idle.targetGazeY = (Math.random() - 0.5) * 0.08;
      idle.nextGazeTime = time + 2.5 + Math.random() * 3.5;
    }
    const gLerp = 1 - Math.exp(-6 * delta);
    idle.currentGazeX = THREE.MathUtils.lerp(idle.currentGazeX, idle.targetGazeX, gLerp);
    idle.currentGazeY = THREE.MathUtils.lerp(idle.currentGazeY, idle.targetGazeY, gLerp);

    // 4. Subtle head sway & breathing
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

    // 5. Morph target blending on meshes
    const meshes = morphMeshesRef.current;
    const eW = getEmotionPreset(emotion, 'tufrilla');
    const sW = getSpeechMorphs(isSpeaking, time, 'tufrilla');

    const iW = {
      eyeBlinkLeft:  idle.blinkVal,
      eyeBlinkRight: idle.blinkVal,
      blink:         idle.blinkVal,
      Blink:         idle.blinkVal,
      eyeLookOutLeft: 0.08,
      eyeLookOutRight: 0.08,
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
      iW.eyeLookUpLeft  = idle.currentGazeY * ga;
      iW.eyeLookUpRight = idle.currentGazeY * ga;
      iW['Look Up']     = idle.currentGazeY * ga;
    } else {
      iW.eyeLookDownLeft  = -idle.currentGazeY * ga;
      iW.eyeLookDownRight = -idle.currentGazeY * ga;
      iW['Look Down']     = -idle.currentGazeY * ga;
    }

    const lfFast = 1 - Math.exp(-26 * delta);
    const lfSlow = 1 - Math.exp(-10 * delta);

    const MOUTH_KEYS = new Set([
      'jawOpen', 'mouthFunnel', 'mouthPucker',
      'mouthUpperUpLeft', 'mouthUpperUpRight',
      'mouthLowerDownLeft', 'mouthLowerDownRight',
      'mouthStretchLeft', 'mouthStretchRight',
      'A', 'O', 'I', 'E', 'U', 'a', 'o', 'e', 'i', 'u',
    ]);

    if (Array.isArray(meshes) && meshes.length > 0) {
      for (const { mesh, dict } of meshes) {
        const influences = mesh.morphTargetInfluences;
        if (!influences) continue;
        for (const [name, idx] of Object.entries(dict)) {
          const target = THREE.MathUtils.clamp((eW[name] || 0) + (sW[name] || 0) + (iW[name] || 0), 0, 1);
          const lf = MOUTH_KEYS.has(name) ? lfFast : lfSlow;
          influences[idx] = THREE.MathUtils.lerp(influences[idx] || 0, target, lf);
        }
      }
    }

    // 6. Also drive VRM Expression Manager if present
    if (vrm?.expressionManager) {
      const exprMgr = vrm.expressionManager;
      // Preset expression keys
      const presets = ['joy', 'angry', 'sorrow', 'fun', 'blink', 'a', 'o', 'i', 'u', 'e'];
      for (const p of presets) {
        const targetVal = THREE.MathUtils.clamp((eW[p] || 0) + (sW[p] || 0) + (iW[p] || 0), 0, 1);
        const curr = exprMgr.getValue(p) || 0;
        const lf = (p === 'a' || p === 'o') ? lfFast : lfSlow;
        exprMgr.setValue(p, THREE.MathUtils.lerp(curr, targetVal, lf));
      }
    }
  });

  return (
    <group ref={avatarGroupRef} position={position}>
      <primitive object={scene} />
    </group>
  );
}
