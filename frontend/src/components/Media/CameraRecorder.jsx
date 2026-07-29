import { useEffect, useRef } from 'react';

/**
 * Small camera preview overlay for the chat area.
 * Shows the live camera feed and a red recording indicator.
 */
export default function CameraRecorder({ stream, isRecording }) {
  const videoRef = useRef(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
    return () => {
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    };
  }, [stream]);

  if (!stream) return null;

  return (
    <div className="absolute bottom-20 left-4 z-30 animate-scale-in">
      <div className="relative rounded-xl overflow-hidden border border-white/10 shadow-lg shadow-black/40">
        {/* Video preview */}
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-40 h-30 object-cover"
          style={{ width: 160, height: 120, transform: 'scaleX(-1)' }}
        />

        {/* Dark overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent pointer-events-none" />

        {/* Recording indicator */}
        {isRecording && (
          <div className="absolute top-2 right-2 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-red-500 recording-pulse" />
            <span className="text-[9px] font-bold text-red-400 uppercase tracking-wider">
              REC
            </span>
          </div>
        )}

        {/* Camera label */}
        <div className="absolute bottom-1.5 left-2 flex items-center gap-1">
          <svg className="w-3 h-3 text-white/50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          <span className="text-[9px] text-white/40 font-medium">Camera</span>
        </div>
      </div>
    </div>
  );
}
