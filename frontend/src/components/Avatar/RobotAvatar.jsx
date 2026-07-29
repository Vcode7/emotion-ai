import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { RoundedBox } from '@react-three/drei';
import * as THREE from 'three';
import useAppStore from '../../store/useAppStore';

/* ─── Emotion Color Map ─── */
const EMOTION_COLORS = {
  neutral: new THREE.Color('#06b6d4'),
  happy: new THREE.Color('#f59e0b'),
  sad: new THREE.Color('#3b82f6'),
  compassion: new THREE.Color('#8b5cf6'),
  concerned: new THREE.Color('#f97316'),
  encouraging: new THREE.Color('#10b981'),
  supportive: new THREE.Color('#34d399'),
  thinking: new THREE.Color('#eab308'),
  angry: new THREE.Color('#ef4444'),
  surprised: new THREE.Color('#ec4899'),
  fearful: new THREE.Color('#a78bfa'),
};

function getEmotionColor(emotion) {
  return EMOTION_COLORS[emotion] || EMOTION_COLORS.neutral;
}

/* ─── Mouth Curve Generator ─── */
function createMouthCurve(emotion) {
  const points = [];
  const segments = 20;

  for (let i = 0; i <= segments; i++) {
    const t = (i / segments) * Math.PI;
    const x = Math.cos(t) * 0.25;
    let y = 0;

    switch (emotion) {
      case 'happy':
      case 'encouraging':
      case 'supportive':
        y = -Math.sin(t) * 0.07;
        break;
      case 'sad':
      case 'fearful':
        y = Math.sin(t) * 0.06;
        break;
      case 'thinking':
        y = Math.sin(t * 2) * 0.03;
        break;
      case 'surprised':
        y = -Math.sin(t) * 0.1;
        break;
      case 'angry':
        y = Math.sin(t) * 0.03;
        break;
      default:
        y = 0;
    }

    points.push(new THREE.Vector3(x, y, 0));
  }

  return new THREE.CatmullRomCurve3(points);
}

/* ─── Eye Component ─── */
function Eye({ position, emotion, targetColor, time }) {
  const meshRef = useRef();
  const glowRef = useRef();

  const scaleY = useMemo(() => {
    switch (emotion) {
      case 'happy':
      case 'encouraging':
        return 0.6; // squished = happy squint
      case 'sad':
        return 1.2;
      case 'surprised':
        return 1.4;
      default:
        return 1.0;
    }
  }, [emotion]);

  const positionOffset = useMemo(() => {
    switch (emotion) {
      case 'sad':
        return [0, -0.03, 0];
      case 'surprised':
        return [0, 0.05, 0];
      default:
        return [0, 0, 0];
    }
  }, [emotion]);

  useFrame(() => {
    if (meshRef.current) {
      // Smooth scale transition
      meshRef.current.scale.y = THREE.MathUtils.lerp(meshRef.current.scale.y, scaleY, 0.06);

      // Subtle random eye micro-movement for life
      const microX = Math.sin(time * 1.7 + position[0] * 10) * 0.008;
      const microY = Math.cos(time * 1.3 + position[1] * 10) * 0.005;
      meshRef.current.position.x = position[0] + positionOffset[0] + microX;
      meshRef.current.position.y = position[1] + positionOffset[1] + microY;

      // Update emissive color
      if (meshRef.current.material) {
        meshRef.current.material.emissive.lerp(targetColor, 0.05);
        meshRef.current.material.color.lerp(targetColor, 0.05);
      }
    }

    if (glowRef.current) {
      glowRef.current.material.opacity = 0.15 + Math.sin(time * 2) * 0.05;
      glowRef.current.scale.y = THREE.MathUtils.lerp(glowRef.current.scale.y, scaleY, 0.06);
    }
  });

  return (
    <group>
      {/* Glow sphere */}
      <mesh ref={glowRef} position={position}>
        <sphereGeometry args={[0.16, 16, 16]} />
        <meshStandardMaterial
          color={targetColor}
          transparent
          opacity={0.15}
          emissive={targetColor}
          emissiveIntensity={0.5}
        />
      </mesh>

      {/* Eye sphere */}
      <mesh ref={meshRef} position={position}>
        <sphereGeometry args={[0.1, 24, 24]} />
        <meshStandardMaterial
          color={targetColor}
          emissive={targetColor}
          emissiveIntensity={1.2}
          toneMapped={false}
        />
      </mesh>

      {/* Specular highlight */}
      <mesh position={[position[0] + 0.03, position[1] + 0.03, position[2] + 0.08]}>
        <sphereGeometry args={[0.025, 12, 12]} />
        <meshStandardMaterial
          color="#ffffff"
          emissive="#ffffff"
          emissiveIntensity={0.8}
          transparent
          opacity={0.8}
        />
      </mesh>
    </group>
  );
}

/* ─── Antenna Component ─── */
function Antenna({ targetColor, time }) {
  const bulbRef = useRef();
  const glowRef = useRef();

  useFrame(() => {
    if (bulbRef.current) {
      const pulse = 0.8 + Math.sin(time * 3) * 0.4;
      bulbRef.current.material.emissiveIntensity = pulse;
      bulbRef.current.material.emissive.lerp(targetColor, 0.05);
      bulbRef.current.material.color.lerp(targetColor, 0.05);
    }
    if (glowRef.current) {
      glowRef.current.material.opacity = 0.1 + Math.sin(time * 3) * 0.08;
    }
  });

  return (
    <group position={[0, 0.82, 0]}>
      {/* Antenna stick */}
      <mesh position={[0, 0.2, 0]}>
        <cylinderGeometry args={[0.02, 0.025, 0.4, 8]} />
        <meshStandardMaterial color="#475569" metalness={0.8} roughness={0.3} />
      </mesh>

      {/* Antenna bulb */}
      <mesh ref={bulbRef} position={[0, 0.45, 0]}>
        <sphereGeometry args={[0.06, 16, 16]} />
        <meshStandardMaterial
          color={targetColor}
          emissive={targetColor}
          emissiveIntensity={0.8}
          toneMapped={false}
        />
      </mesh>

      {/* Bulb glow */}
      <mesh ref={glowRef} position={[0, 0.45, 0]}>
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshStandardMaterial
          color={targetColor}
          transparent
          opacity={0.12}
          emissive={targetColor}
          emissiveIntensity={0.3}
        />
      </mesh>
    </group>
  );
}

/* ─── Mouth Component ─── */
function Mouth({ emotion, targetColor, isSpeaking }) {
  const tubeRef = useRef();

  useFrame((state) => {
    if (tubeRef.current) {
      tubeRef.current.material.emissive.lerp(targetColor, 0.05);
      tubeRef.current.material.color.lerp(targetColor, 0.05);

      if (isSpeaking) {
        const elapsed = state.clock.getElapsedTime();
        // Modulate mouth scale dynamically to simulate talking
        const talkFactor = Math.sin(elapsed * 16) * 0.45 * Math.abs(Math.sin(elapsed * 6));
        tubeRef.current.scale.y = 1.0 + talkFactor;
        tubeRef.current.scale.x = 1.0 + Math.cos(elapsed * 10) * 0.08;
      } else {
        // Return smoothly to default scale when idle
        tubeRef.current.scale.y = THREE.MathUtils.lerp(tubeRef.current.scale.y, 1.0, 0.1);
        tubeRef.current.scale.x = THREE.MathUtils.lerp(tubeRef.current.scale.x, 1.0, 0.1);
      }
    }
  });

  const curve = useMemo(() => createMouthCurve(emotion), [emotion]);

  const geometry = useMemo(() => {
    return new THREE.TubeGeometry(curve, 20, 0.018, 8, false);
  }, [curve]);

  return (
    <mesh ref={tubeRef} position={[0, -0.22, 0.38]} geometry={geometry}>
      <meshStandardMaterial
        color={targetColor}
        emissive={targetColor}
        emissiveIntensity={0.6}
        toneMapped={false}
      />
    </mesh>
  );
}

/* ─── Ear Component ─── */
function Ear({ position }) {
  return (
    <mesh position={position}>
      <boxGeometry args={[0.08, 0.2, 0.15]} />
      <meshStandardMaterial
        color="#1e293b"
        metalness={0.7}
        roughness={0.3}
      />
    </mesh>
  );
}

/* ─── Arm Component ─── */
function Arm({ position, side, time }) {
  const armRef = useRef();

  useFrame(() => {
    if (armRef.current) {
      const swing = Math.sin(time * 1.2 + (side === 'left' ? 0 : Math.PI)) * 0.08;
      armRef.current.rotation.z = swing + (side === 'left' ? 0.15 : -0.15);
    }
  });

  return (
    <group ref={armRef} position={position}>
      {/* Upper arm */}
      <mesh position={[0, -0.18, 0]}>
        <cylinderGeometry args={[0.04, 0.035, 0.35, 8]} />
        <meshStandardMaterial color="#334155" metalness={0.7} roughness={0.3} />
      </mesh>
      {/* Hand */}
      <mesh position={[0, -0.38, 0]}>
        <sphereGeometry args={[0.05, 12, 12]} />
        <meshStandardMaterial color="#475569" metalness={0.6} roughness={0.3} />
      </mesh>
    </group>
  );
}

/* ─── Main Robot Avatar ─── */
export default function RobotAvatar({ emotion = 'neutral', avatarCommands = null }) {
  const isSpeaking = useAppStore((s) => s.isSpeaking);
  const groupRef = useRef();
  const headRef = useRef();
  const timeRef = useRef(0);

  const targetColor = useMemo(() => getEmotionColor(emotion), [emotion]);

  const headTilt = useMemo(() => {
    if (avatarCommands?.head_tilt) {
      const tilt = avatarCommands.head_tilt;
      return {
        x: (tilt.pitch || 0) * 0.15,
        z: (tilt.roll || 0) * 0.1,
      };
    }
    return { x: 0, z: 0 };
  }, [avatarCommands]);

  useFrame((_, delta) => {
    timeRef.current += delta;
    const t = timeRef.current;

    // Idle floating bob
    if (groupRef.current) {
      let targetY = Math.sin(t * 0.8) * 0.06;
      let targetRotY = Math.sin(t * 0.3) * 0.05;

      if (isSpeaking) {
        // Subtle rapid bobbing while speaking
        targetY += Math.sin(t * 12) * 0.015;
      }

      groupRef.current.position.y = targetY;
      groupRef.current.rotation.y = targetRotY;
    }

    // Head tilt lerp
    if (headRef.current) {
      let targetX = headTilt.x;
      let targetZ = headTilt.z;

      if (isSpeaking) {
        // Expressive talking head gestures (nods and shakes)
        targetX += Math.sin(t * 6) * 0.04;
        targetZ += Math.cos(t * 4) * 0.025;
      }

      headRef.current.rotation.x = THREE.MathUtils.lerp(
        headRef.current.rotation.x,
        targetX,
        0.04
      );
      headRef.current.rotation.z = THREE.MathUtils.lerp(
        headRef.current.rotation.z,
        targetZ,
        0.04
      );
    }
  });

  return (
    <group ref={groupRef} position={[0, -0.2, 0]}>
      {/* Head group */}
      <group ref={headRef}>
        {/* Head */}
        <RoundedBox args={[1.0, 0.85, 0.7]} radius={0.12} smoothness={6}>
          <meshStandardMaterial
            color="#1e293b"
            metalness={0.6}
            roughness={0.35}
          />
        </RoundedBox>

        {/* Face plate - slightly raised */}
        <RoundedBox
          args={[0.8, 0.65, 0.05]}
          radius={0.08}
          smoothness={4}
          position={[0, -0.02, 0.34]}
        >
          <meshStandardMaterial
            color="#0f172a"
            metalness={0.4}
            roughness={0.5}
          />
        </RoundedBox>

        {/* Eyes */}
        <Eye
          position={[-0.2, 0.08, 0.38]}
          emotion={emotion}
          targetColor={targetColor}
          time={timeRef.current}
        />
        <Eye
          position={[0.2, 0.08, 0.38]}
          emotion={emotion}
          targetColor={targetColor}
          time={timeRef.current}
        />

        {/* Mouth */}
        <Mouth emotion={emotion} targetColor={targetColor} isSpeaking={isSpeaking} />

        {/* Antenna */}
        <Antenna targetColor={targetColor} time={timeRef.current} />

        {/* Ears */}
        <Ear position={[-0.55, 0, 0]} />
        <Ear position={[0.55, 0, 0]} />
      </group>

      {/* Neck */}
      <mesh position={[0, -0.55, 0]}>
        <cylinderGeometry args={[0.08, 0.1, 0.15, 8]} />
        <meshStandardMaterial color="#334155" metalness={0.7} roughness={0.3} />
      </mesh>

      {/* Body */}
      <RoundedBox
        args={[0.7, 0.6, 0.5]}
        radius={0.1}
        smoothness={4}
        position={[0, -0.95, 0]}
      >
        <meshStandardMaterial
          color="#1e293b"
          metalness={0.55}
          roughness={0.4}
        />
      </RoundedBox>

      {/* Chest lights */}
      <mesh position={[-0.1, -0.88, 0.26]}>
        <sphereGeometry args={[0.03, 10, 10]} />
        <meshStandardMaterial
          color="#3b82f6"
          emissive="#3b82f6"
          emissiveIntensity={0.8}
          toneMapped={false}
        />
      </mesh>
      <mesh position={[0.1, -0.88, 0.26]}>
        <sphereGeometry args={[0.03, 10, 10]} />
        <meshStandardMaterial
          color="#8b5cf6"
          emissive="#8b5cf6"
          emissiveIntensity={0.8}
          toneMapped={false}
        />
      </mesh>

      {/* Arms */}
      <Arm position={[-0.48, -0.75, 0]} side="left" time={timeRef.current} />
      <Arm position={[0.48, -0.75, 0]} side="right" time={timeRef.current} />
    </group>
  );
}
