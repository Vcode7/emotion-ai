import { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import { getEmotionPreset } from './emotionPresets';
import { getSpeechMorphs } from './audioAnalyser';

/**
 * MetaHumanAvatar — Real-time 3D MetaHuman Head with 52 ARKit Blendshapes
 * =========================================================================
 *
 * Loads: /models/face/metaHumanHead.glb  (Clean, direct, high-performance)
 *
 * Blendshape naming: Unreal MetaHuman convention (Left/Right suffix)
 *   eyeBlinkLeft, mouthSmileLeft, browDownLeft, jawOpen …
 *
 * Features:
 *   - Instant loading from disk/cache without heavy hair or KTX2 overhead
 *   - Emotion presets (MH_EMOTION_PRESETS from emotionPresets.js)
 *   - Audio-driven lip sync (getSpeechMorphs with 'metahuman' variant)
 *   - Idle: procedural blinking, eye saccades, head sway, breathing
 */
export default function MetaHumanAvatar({
  emotion = 'neutral',
  isSpeaking = false,
  status = 'idle',
  position = [0, 0, 0],
  onLoaded,
}) {
  const headGroupRef = useRef();
  const morphMeshRef = useRef(null);

  // Load MetaHuman Head GLB directly — clean URL for instant browser caching
  const gltf = useGLTF('/models/face/metaHumanHead.glb');

  // Clone and scale head model exactly ONCE to prevent compounding transform drift
  const scene = useMemo(() => {
    if (!gltf?.scene) return null;
    const cloned = gltf.scene.clone(true);

    const box = new THREE.Box3().setFromObject(cloned);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    if (maxDim > 0) cloned.scale.setScalar(2.45 / maxDim);

    const box2 = new THREE.Box3().setFromObject(cloned);
    const center = new THREE.Vector3();
    box2.getCenter(center);
    cloned.position.sub(center);

    return cloned;
  }, [gltf.scene]);

  // High-fidelity dental & oral cavity textures (teeth, gums, tongue, palate)
  const teethTextures = useMemo(() => {
    const loader = new THREE.TextureLoader();
    const diffuse = loader.load('/models/face/textures/metaHuman_teeth_diffuse.png');
    diffuse.flipY = false;
    diffuse.colorSpace = THREE.SRGBColorSpace;
    diffuse.generateMipmaps = true;
    diffuse.minFilter = THREE.LinearMipmapLinearFilter;

    const normal = loader.load('/models/face/textures/metaHuman_teeth_normal.png');
    normal.flipY = false;

    const roughness = loader.load('/models/face/textures/metaHuman_teeth_roughness.png');
    roughness.flipY = false;

    return { diffuse, normal, roughness };
  }, []);

  useEffect(() => {
    if (!scene) return;

    const meshes = [];
    scene.traverse((child) => {
      if (child.isMesh) {
        child.frustumCulled = false;
        const nameLower = (child.name || '').toLowerCase();

        // Hair option removed: hide any embedded hair mesh for clean high-performance load
        if (nameLower.includes('hair')) {
          child.visible = false;
          return;
        }

        if (child.material) {
          child.material = child.material.clone();
          child.material.needsUpdate = true;

          if (nameLower.includes('eye')) {
            // Realistic glossy eye cornea with sharp specular catchlights
            child.material.roughness = 0.03;
            child.material.metalness = 0.0;
            child.material.envMapIntensity = 1.4;
            if ('clearcoat' in child.material) {
              child.material.clearcoat = 1.0;
              child.material.clearcoatRoughness = 0.02;
            }
          } else if (nameLower.includes('teeth') || nameLower.includes('gum')) {
            // Photorealistic moist sparkling white dental enamel + oral cavity
            child.material = new THREE.MeshPhysicalMaterial({
              color: new THREE.Color(1.0, 1.0, 1.0),
              map: teethTextures.diffuse,
              normalMap: teethTextures.normal,
              normalScale: new THREE.Vector2(0.9, 0.9),
              roughnessMap: teethTextures.roughness,
              roughness: 0.16,
              metalness: 0.0,
              clearcoat: 1.0,
              clearcoatRoughness: 0.04,
              envMapIntensity: 1.1,
              ior: 1.58,
              // Soft enamel luminescence ensures teeth look pristine white in mouth shadows
              emissive: new THREE.Color(0.22, 0.22, 0.22),
              emissiveMap: teethTextures.diffuse,
            });
            child.material.needsUpdate = true;
          } else {
            // Head skin with natural soft micro-roughness
            child.material.roughness = 0.62;
            child.material.metalness = 0.0;
            child.material.envMapIntensity = 0.65;
          }
        }

        if (child.morphTargetDictionary && child.morphTargetInfluences) {
          const count = Object.keys(child.morphTargetDictionary).length;
          console.log(`[MetaHumanAvatar] Morph mesh "${child.name}": ${count} keys`);
          meshes.push({
            mesh: child,
            dict: child.morphTargetDictionary,
          });
        }
      }
    });

    morphMeshRef.current = meshes;
    if (onLoaded) {
      onLoaded();
    }
  }, [scene, teethTextures, onLoaded]);

  // Idle animation state (persistent refs)
  const idleState = useRef({
    nextBlinkTime: 2.0, blinkDuration: 0.22, blinkStartTime: 0,
    isBlinking: false, blinkVal: 0,
    nextGazeTime: 1.5, targetGazeX: 0, targetGazeY: 0,
    currentGazeX: 0, currentGazeY: 0,
  });

  useFrame((state, delta) => {
    const time = state.clock.getElapsedTime();
    const idle = idleState.current;

    // 1. Blinking
    if (!idle.isBlinking && time >= idle.nextBlinkTime) {
      idle.isBlinking = true;
      idle.blinkStartTime = time;
      idle.blinkDuration = 0.18 + Math.random() * 0.08;
    }
    if (idle.isBlinking) {
      const p = (time - idle.blinkStartTime) / idle.blinkDuration;
      if (p >= 1.0) {
        idle.isBlinking = false; idle.blinkVal = 0;
        idle.nextBlinkTime = time + (Math.random() < 0.2 ? 0.35 : 2.5 + Math.random() * 3.0);
      } else {
        idle.blinkVal = p < 0.4 ? p / 0.4 : 1.0 - (p - 0.4) / 0.6;
      }
    }

    // 2. Gaze saccades
    if (time >= idle.nextGazeTime) {
      idle.targetGazeX = (Math.random() - 0.5) * 0.12;
      idle.targetGazeY = (Math.random() - 0.5) * 0.08;
      idle.nextGazeTime = time + 2.5 + Math.random() * 3.5;
    }
    const gLerp = 1 - Math.exp(-6 * delta);
    idle.currentGazeX = THREE.MathUtils.lerp(idle.currentGazeX, idle.targetGazeX, gLerp);
    idle.currentGazeY = THREE.MathUtils.lerp(idle.currentGazeY, idle.targetGazeY, gLerp);

    // 3. Head sway
    if (headGroupRef.current) {
      const lt = status === 'listening' ? 0.03 : 0;
      headGroupRef.current.rotation.x = THREE.MathUtils.lerp(headGroupRef.current.rotation.x, Math.sin(time * 0.6) * 0.015 + idle.currentGazeY * 0.02, 0.08);
      headGroupRef.current.rotation.y = THREE.MathUtils.lerp(headGroupRef.current.rotation.y, Math.sin(time * 0.45) * 0.025 + idle.currentGazeX * 0.03 + lt, 0.08);
      headGroupRef.current.rotation.z = THREE.MathUtils.lerp(headGroupRef.current.rotation.z, Math.sin(time * 0.35) * 0.008, 0.08);
      headGroupRef.current.position.y = THREE.MathUtils.lerp(headGroupRef.current.position.y, Math.sin(time * 1.3) * 0.004, 0.08);
    }

    // 4. Morph blending across all meshes (head, eyes, teeth)
    const meshes = morphMeshRef.current;
    if (!Array.isArray(meshes) || meshes.length === 0) return;

    const eW = getEmotionPreset(emotion, 'metahuman');
    const sW = getSpeechMorphs(isSpeaking, time, 'metahuman');

    // Idle contributions (MetaHuman Left/Right naming)
    const iW = {
      eyeBlinkLeft:   idle.blinkVal,
      eyeBlinkRight:  idle.blinkVal,
      eyeLookOutLeft: 0.12,
      eyeLookOutRight: 0.12,
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

    // Mouth/jaw morphs use a fast lerp factor for accurate lip-sync timing.
    // Emotion (brow, eye) morphs use a slower lerp for smooth expression blending.
    const lfFast = 1 - Math.exp(-26 * delta);  // ~26 Hz effective — tracks syllables
    const lfSlow = 1 - Math.exp(-10 * delta);  // ~10 Hz effective — smooth expressions

    // Morph keys that drive lip articulation (need fast tracking)
    const MOUTH_KEYS = new Set([
      'jawOpen', 'mouthFunnel', 'mouthPucker',
      'mouthUpperUpLeft', 'mouthUpperUpRight',
      'mouthLowerDownLeft', 'mouthLowerDownRight',
      'mouthStretchLeft', 'mouthStretchRight',
      'mouthUpperUp_L', 'mouthUpperUp_R',
      'mouthLowerDown_L', 'mouthLowerDown_R',
      'mouthStretch_L', 'mouthStretch_R',
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
    <group ref={headGroupRef} position={position}>
      <primitive object={scene} />
    </group>
  );
}

// Preload clean URL so it immediately caches in the browser
useGLTF.preload('/models/face/metaHumanHead.glb');