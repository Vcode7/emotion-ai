import { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import RobotAvatar from './RobotAvatar';
import useAppStore from '../../store/useAppStore';

function LoadingFallback() {
  return (
    <mesh>
      <sphereGeometry args={[0.3, 16, 16]} />
      <meshStandardMaterial
        color="#3b82f6"
        emissive="#3b82f6"
        emissiveIntensity={0.5}
        wireframe
      />
    </mesh>
  );
}

function SceneBackground() {
  return (
    <>
      {/* Ambient fill light */}
      <ambientLight intensity={0.4} />

      {/* Key light */}
      <directionalLight
        position={[3, 4, 5]}
        intensity={0.8}
        color="#e2e8f0"
      />

      {/* Fill light from left */}
      <directionalLight
        position={[-3, 2, 3]}
        intensity={0.3}
        color="#3b82f6"
      />

      {/* Rim light */}
      <directionalLight
        position={[0, -2, -4]}
        intensity={0.2}
        color="#8b5cf6"
      />

      {/* Background plane with gradient */}
      <mesh position={[0, 0, -3]} scale={[12, 12, 1]}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial color="#000000ff" />
      </mesh>

      {/* Floor grid effect */}
      <gridHelper
        args={[10, 20, '#1e293b', '#111827']}
        position={[0, -2, 0]}
        rotation={[0, 0, 0]}
      />
    </>
  );
}

export default function AvatarScene() {
  const currentEmotion = useAppStore((s) => s.currentEmotion);
  const currentAvatarCommands = useAppStore((s) => s.currentAvatarCommands);

  return (
    <div className="relative w-full h-full overflow-hidden rounded-2xl">
      {/* Subtle gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-navy-900/0 via-transparent to-navy-900/80 z-10 pointer-events-none" />

      {/* Emotion label */}
      <div className="absolute top-4 left-4 z-20 flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-accent-cyan animate-pulse-glow" />
        <span className="text-[11px] text-white/40 font-medium uppercase tracking-wider">
          Avatar • {currentEmotion}
        </span>
      </div>

      <Canvas
        camera={{ position: [0, 0, 3.5], fov: 45 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
        style={{ background: 'transparent' }}
      >
        <SceneBackground />

        <Suspense fallback={<LoadingFallback />}>
          <RobotAvatar
            emotion={currentEmotion}
            avatarCommands={currentAvatarCommands}
          />
        </Suspense>

        <OrbitControls
          enableZoom={false}
          enablePan={false}
          minPolarAngle={Math.PI / 3}
          maxPolarAngle={Math.PI / 1.8}
          minAzimuthAngle={-Math.PI / 6}
          maxAzimuthAngle={Math.PI / 6}
          rotateSpeed={0.4}
        />
      </Canvas>
    </div>
  );
}
