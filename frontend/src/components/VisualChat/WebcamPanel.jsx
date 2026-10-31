import { useEffect, useRef } from 'react';

/**
 * WebcamPanel.jsx — Live webcam feed display for Visual Chat
 *
 * Supports both standalone card mode and floating Picture-in-Picture (PiP) mode.
 * Automatically attaches the provided MediaStream to its internal video element.
 */
export default function WebcamPanel({ stream, isActive, className = '', isPip = false }) {
  const localVideoRef = useRef(null);

  useEffect(() => {
    const video = localVideoRef.current;
    if (!video) return;

    if (stream && isActive) {
      video.srcObject = stream;
      video.play().catch((err) => {
        console.warn('[WebcamPanel] video.play() error:', err);
      });
    } else {
      video.srcObject = null;
    }
  }, [stream, isActive]);

  return (
    <div
      className={`relative w-full h-full rounded-2xl overflow-hidden bg-navy-900/70 border transition-all ${
        isPip
          ? 'border-white/20 shadow-2xl backdrop-blur-md hover:border-accent-blue/50 cursor-pointer'
          : 'border-white/[0.08]'
      } ${className}`}
    >
      {/* Live video element */}
      <video
        ref={localVideoRef}
        autoPlay
        playsInline
        muted
        className={`w-full h-full object-cover transition-opacity duration-500 ${
          isActive && stream ? 'opacity-100' : 'opacity-0'
        }`}
        style={{ transform: 'scaleX(-1)' }} /* mirror the feed */
      />

      {/* Placeholder when camera is off or connecting */}
      {(!isActive || !stream) && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-1.5 sm:gap-2 animate-fade-in p-2 text-center">
          <div className={`${isPip ? 'w-8 h-8 rounded-lg' : 'w-14 h-14 rounded-2xl'} bg-white/[0.04] border border-white/[0.08] flex items-center justify-center`}>
            <svg className={`${isPip ? 'w-4 h-4' : 'w-6 h-6'} text-white/20`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"
              />
            </svg>
          </div>
          {!isPip && (
            <span className="text-xs text-white/25 font-medium">
              {isActive ? 'Connecting camera…' : 'Camera Off'}
            </span>
          )}
        </div>
      )}

      {/* Live status badge */}
      {isActive && stream && (
        <div className={`absolute ${isPip ? 'top-1.5 left-1.5 px-1.5 py-0.5' : 'top-2.5 left-2.5 px-2 py-0.5'} rounded-full bg-black/60 backdrop-blur-sm border border-white/10 flex items-center gap-1 animate-fade-in`}>
          <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
          <span className={`${isPip ? 'text-[8px]' : 'text-[10px]'} text-white/60 font-medium uppercase tracking-wider`}>You</span>
        </div>
      )}

      {/* Subtle corner accents in standard mode */}
      {!isPip && (
        <>
          <div className="absolute top-0 left-0 w-5 h-5 border-t-2 border-l-2 border-accent-blue/30 rounded-tl-2xl pointer-events-none" />
          <div className="absolute top-0 right-0 w-5 h-5 border-t-2 border-r-2 border-accent-blue/30 rounded-tr-2xl pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-5 h-5 border-b-2 border-l-2 border-accent-blue/30 rounded-bl-2xl pointer-events-none" />
          <div className="absolute bottom-0 right-0 w-5 h-5 border-b-2 border-r-2 border-accent-blue/30 rounded-br-2xl pointer-events-none" />
        </>
      )}
    </div>
  );
}
