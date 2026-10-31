import { useState, useRef, useEffect, useCallback } from 'react';
import { detectFaceEmotion } from '../../api/client';

const EMOTION_CONFIG = {
  happy: {
    label: 'Happy',
    emoji: '😄',
    color: 'from-amber-400 to-yellow-500',
    textColor: 'text-amber-400',
    borderColor: 'border-amber-400/40',
    bgColor: 'bg-amber-400/10',
    barColor: 'from-amber-400 to-yellow-500',
    glow: 'shadow-[0_0_25px_rgba(245,158,11,0.25)]',
  },
  surprised: {
    label: 'Surprised',
    emoji: '😲',
    color: 'from-purple-400 to-pink-500',
    textColor: 'text-purple-400',
    borderColor: 'border-purple-400/40',
    bgColor: 'bg-purple-400/10',
    barColor: 'from-purple-400 to-pink-500',
    glow: 'shadow-[0_0_25px_rgba(168,85,247,0.25)]',
  },
  neutral: {
    label: 'Neutral',
    emoji: '😐',
    color: 'from-cyan-400 to-blue-500',
    textColor: 'text-cyan-400',
    borderColor: 'border-cyan-400/40',
    bgColor: 'bg-cyan-400/10',
    barColor: 'from-cyan-400 to-blue-500',
    glow: 'shadow-[0_0_25px_rgba(6,182,212,0.25)]',
  },
  sad: {
    label: 'Sad',
    emoji: '😢',
    color: 'from-blue-400 to-indigo-500',
    textColor: 'text-blue-400',
    borderColor: 'border-blue-400/40',
    bgColor: 'bg-blue-400/10',
    barColor: 'from-blue-400 to-indigo-500',
    glow: 'shadow-[0_0_25px_rgba(59,130,246,0.25)]',
  },
  angry: {
    label: 'Angry',
    emoji: '😠',
    color: 'from-red-400 to-rose-600',
    textColor: 'text-red-400',
    borderColor: 'border-red-400/40',
    bgColor: 'bg-red-400/10',
    barColor: 'from-red-400 to-rose-600',
    glow: 'shadow-[0_0_25px_rgba(239,68,68,0.25)]',
  },
  fearful: {
    label: 'Fearful',
    emoji: '😨',
    color: 'from-violet-400 to-purple-600',
    textColor: 'text-violet-400',
    borderColor: 'border-violet-400/40',
    bgColor: 'bg-violet-400/10',
    barColor: 'from-violet-400 to-purple-600',
    glow: 'shadow-[0_0_25px_rgba(139,92,246,0.25)]',
  },
};

const DEFAULT_CONFIG = {
  label: 'Unknown',
  emoji: '🤔',
  color: 'from-slate-400 to-slate-500',
  textColor: 'text-slate-400',
  borderColor: 'border-slate-400/40',
  bgColor: 'bg-slate-400/10',
  barColor: 'from-slate-400 to-slate-500',
  glow: '',
};

export default function FaceEmotionTestTab() {
  const [stream, setStream] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [scanHistory, setScanHistory] = useState([]);
  const [errorMessage, setErrorMessage] = useState(null);

  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  // ── Enumerate video devices ───────────────────────────────────────────────
  const refreshDevices = useCallback(async () => {
    try {
      const allDevices = await navigator.mediaDevices?.enumerateDevices();
      const videoDevs = allDevices?.filter((d) => d.kind === 'videoinput') || [];
      setDevices(videoDevs);
      if (videoDevs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevs[0].deviceId);
      }
    } catch (e) {
      console.warn('[FaceEmotionTest] Device enumeration error:', e);
    }
  }, [selectedDeviceId]);

  // ── Start webcam stream ───────────────────────────────────────────────────
  const startCamera = useCallback(async (deviceId) => {
    setCameraError(null);
    try {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
      const constraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      };
      const s = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(s);
      setIsCameraActive(true);
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        videoRef.current.play().catch(() => {});
      }
      refreshDevices();
    } catch (err) {
      console.error('[FaceEmotionTest] Camera access failed:', err);
      setCameraError(err.message || 'Unable to access camera. Please allow permission.');
      setIsCameraActive(false);
    }
  }, [stream, refreshDevices]);

  // ── Stop webcam stream ────────────────────────────────────────────────────
  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  }, [stream]);

  // Auto-start camera when tab mounts
  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, []);

  // Update video element when stream or ref changes
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(() => {});
    }
  }, [stream]);

  // ── Capture and scan frame from webcam ────────────────────────────────────
  const captureAndScan = async () => {
    if (!videoRef.current || !stream) {
      setErrorMessage('Please start the camera first or upload an image file.');
      return;
    }

    const video = videoRef.current;
    if (video.readyState < 2 || video.videoWidth === 0) {
      setErrorMessage('Camera feed is still loading, please wait a moment.');
      return;
    }

    setIsScanning(true);
    setErrorMessage(null);

    try {
      const width = video.videoWidth;
      const height = video.videoHeight;
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      // Mirror horizontal for realistic camera snapshot
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, width, height);

      const previewUrl = canvas.toDataURL('image/jpeg', 0.9);
      setCapturedImage(previewUrl);

      // Convert canvas to blob for API call
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9));
      if (!blob) throw new Error('Failed to create image blob from video frame');

      const data = await detectFaceEmotion(blob);
      setScanResult(data);

      const historyItem = {
        id: Date.now(),
        preview: previewUrl,
        emotion: data.emotion,
        rawEmotion: data.raw_emotion,
        confidence: data.confidence,
        time: new Date().toLocaleTimeString(),
        faceDetected: data.face_detected,
      };
      setScanHistory((prev) => [historyItem, ...prev.slice(0, 7)]);
    } catch (err) {
      console.error('[FaceEmotionTest] Scan failed:', err);
      setErrorMessage(err.response?.data?.detail || err.message || 'Detection failed.');
    } finally {
      setIsScanning(false);
    }
  };

  // ── Handle file upload scan ───────────────────────────────────────────────
  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsScanning(true);
    setErrorMessage(null);

    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const previewUrl = e.target.result;
        setCapturedImage(previewUrl);

        try {
          const data = await detectFaceEmotion(file);
          setScanResult(data);

          const historyItem = {
            id: Date.now(),
            preview: previewUrl,
            emotion: data.emotion,
            rawEmotion: data.raw_emotion,
            confidence: data.confidence,
            time: new Date().toLocaleTimeString(),
            faceDetected: data.face_detected,
          };
          setScanHistory((prev) => [historyItem, ...prev.slice(0, 7)]);
        } catch (apiErr) {
          setErrorMessage(apiErr.response?.data?.detail || apiErr.message || 'Detection failed.');
        } finally {
          setIsScanning(false);
        }
      };
      reader.readAsDataURL(file);
    } catch (err) {
      setIsScanning(false);
      setErrorMessage(err.message || 'Error reading file.');
    }
  };

  const activeConfig = scanResult?.emotion
    ? EMOTION_CONFIG[scanResult.emotion.toLowerCase()] || DEFAULT_CONFIG
    : DEFAULT_CONFIG;

  return (
    <div className="h-full w-full flex flex-col overflow-y-auto bg-navy-950 text-white p-3 sm:p-6 pb-20 sm:pb-8">
      {/* ── Top Header & Info Banner ── */}
      <div className="max-w-6xl w-full mx-auto mb-4 flex flex-col md:flex-row md:items-center justify-between gap-3 bg-white/[0.03] border border-white/[0.08] p-4 rounded-2xl backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center text-xl shadow-lg shadow-purple-500/20">
            👁️
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base sm:text-lg font-bold tracking-tight">Face Emotion Scanner</h2>
              <span className="px-2 py-0.5 text-[10px] font-semibold bg-purple-500/20 text-purple-300 border border-purple-500/30 rounded-full uppercase tracking-wider">
                Model Verification
              </span>
            </div>
            <p className="text-xs text-white/50">
              Testing <span className="font-mono text-purple-300">HardlyHumans/Facial-expression-detection</span> (ViT) & Haar Cascade.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <div className="px-3 py-1.5 rounded-xl bg-white/[0.04] border border-white/[0.08] flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-white/70">ViT Loaded & Active</span>
          </div>
          <button
            onClick={() => {
              if (isCameraActive) stopCamera();
              else startCamera(selectedDeviceId);
            }}
            className={`px-3 py-1.5 rounded-xl font-medium cursor-pointer transition-all border ${
              isCameraActive
                ? 'bg-red-500/10 border-red-500/30 text-red-300 hover:bg-red-500/20'
                : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/20'
            }`}
          >
            {isCameraActive ? 'Stop Camera' : 'Start Camera'}
          </button>
        </div>
      </div>

      {/* ── Main Layout: Camera/Capture on Left, Scan Results on Right ── */}
      <div className="max-w-6xl w-full mx-auto grid grid-cols-1 lg:grid-cols-12 gap-5 flex-1 min-h-0">
        {/* Left Column: Camera Viewfinder & Controls (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <div className="relative aspect-video w-full rounded-2xl overflow-hidden bg-black/60 border border-white/[0.1] shadow-2xl flex items-center justify-center group">
            {/* Live Video */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover transition-opacity duration-300 ${
                isCameraActive ? 'opacity-100' : 'opacity-0'
              }`}
              style={{ transform: 'scaleX(-1)' }}
            />

            {/* When Camera is Off */}
            {!isCameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-navy-900/80">
                <div className="w-16 h-16 rounded-2xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center text-3xl mb-3">
                  📷
                </div>
                <h3 className="text-sm font-semibold text-white/80 mb-1">Camera Feed Inactive</h3>
                <p className="text-xs text-white/40 max-w-sm mb-4">
                  Enable the camera to capture live expressions or choose an existing photo to scan.
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={() => startCamera(selectedDeviceId)}
                    className="px-4 py-2 rounded-xl bg-accent-blue hover:bg-accent-blue/80 text-white text-xs font-semibold shadow-md shadow-accent-blue/30 cursor-pointer transition-all"
                  >
                    Turn Camera On
                  </button>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="px-4 py-2 rounded-xl bg-white/[0.08] hover:bg-white/15 text-white/80 text-xs font-semibold cursor-pointer transition-all border border-white/10"
                  >
                    Upload Image
                  </button>
                </div>
              </div>
            )}

            {/* Face Reticle / Guide Overlay */}
            {isCameraActive && (
              <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                <div className="w-56 h-64 border-2 border-dashed border-white/30 rounded-3xl relative flex items-center justify-center">
                  <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-accent-cyan" />
                  <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-accent-cyan" />
                  <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-accent-cyan" />
                  <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-accent-cyan" />
                  <span className="text-[10px] text-white/40 font-mono tracking-wider uppercase bg-black/40 px-2 py-0.5 rounded-full">
                    Position Face Here
                  </span>
                </div>
              </div>
            )}

            {/* Scanning Laser Line Animation */}
            {isScanning && (
              <div className="absolute inset-0 pointer-events-none overflow-hidden z-20">
                <div className="w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_15px_#06b6d4] animate-bounce" />
                <div className="absolute inset-0 bg-cyan-500/10 backdrop-blur-[1px] flex items-center justify-center">
                  <div className="px-4 py-2 rounded-xl bg-black/80 border border-cyan-500/40 text-cyan-300 text-xs font-semibold flex items-center gap-2 shadow-2xl">
                    <span className="w-3 h-3 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                    Running ViT Classifier…
                  </div>
                </div>
              </div>
            )}

            {/* Live Camera Watermark */}
            {isCameraActive && (
              <div className="absolute top-3 left-3 px-2 py-1 rounded-lg bg-black/60 backdrop-blur-md border border-white/10 flex items-center gap-1.5 text-[10px] text-white/70">
                <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                <span>LIVE FEED</span>
              </div>
            )}
          </div>

          {/* Camera Action Buttons & Controls */}
          <div className="flex flex-wrap items-center justify-between gap-3 bg-white/[0.02] border border-white/[0.06] p-3 rounded-2xl">
            <div className="flex items-center gap-2">
              <button
                id="btn-take-pic-scan"
                onClick={captureAndScan}
                disabled={!isCameraActive || isScanning}
                className={`px-5 py-2.5 rounded-xl font-bold text-sm flex items-center gap-2 shadow-lg transition-all cursor-pointer ${
                  !isCameraActive || isScanning
                    ? 'opacity-40 cursor-not-allowed bg-white/10 text-white/40'
                    : 'bg-gradient-to-r from-accent-cyan via-accent-blue to-accent-purple text-white shadow-accent-cyan/25 hover:scale-[1.02] active:scale-[0.98]'
                }`}
              >
                <span>📸</span>
                <span>{isScanning ? 'Scanning…' : 'Take Pic & Scan Emotion'}</span>
              </button>

              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isScanning}
                className="px-3.5 py-2.5 rounded-xl font-medium text-xs bg-white/[0.05] hover:bg-white/10 border border-white/10 text-white/80 cursor-pointer transition-all flex items-center gap-1.5"
              >
                <span>📁</span>
                <span className="hidden sm:inline">Upload Photo</span>
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileUpload}
              />
            </div>

            {/* Device selector */}
            {devices.length > 1 && isCameraActive && (
              <select
                value={selectedDeviceId}
                onChange={(e) => {
                  setSelectedDeviceId(e.target.value);
                  startCamera(e.target.value);
                }}
                className="px-2.5 py-1.5 rounded-xl bg-navy-900 border border-white/10 text-xs text-white/70 focus:outline-none focus:border-accent-cyan"
              >
                {devices.map((d, i) => (
                  <option key={d.deviceId} value={d.deviceId}>
                    {d.label || `Camera ${i + 1}`}
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Error Banner */}
          {errorMessage && (
            <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-xs flex items-center gap-2">
              <span>⚠️</span>
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Quick expression testing tips */}
          <div className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/[0.05] text-xs text-white/50">
            <span className="font-semibold text-white/70 block mb-1">💡 Tips for testing accuracy:</span>
            <ul className="list-disc pl-4 space-y-0.5 text-[11px] text-white/40">
              <li>Smile widely with teeth or grin to test <strong>Happy</strong>.</li>
              <li>Open mouth and raise eyebrows to test <strong>Surprised</strong>.</li>
              <li>Frown or lower eyebrows firmly to test <strong>Angry</strong>.</li>
              <li>Keep a relaxed, resting facial posture to test <strong>Neutral</strong>.</li>
              <li>Ensure good front-facing lighting so the Haar Cascade accurately crops the face.</li>
            </ul>
          </div>
        </div>

        {/* Right Column: Scan Results & Emotion Breakdown (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          {scanResult ? (
            <div className={`p-5 rounded-2xl bg-white/[0.03] border ${activeConfig.borderColor} ${activeConfig.glow} flex flex-col gap-4 transition-all duration-500`}>
              {/* Snapshot thumbnail + Detected Emotion Headline */}
              <div className="flex items-center gap-4">
                {capturedImage && (
                  <div className="w-20 h-20 rounded-2xl overflow-hidden border border-white/20 shadow-md flex-shrink-0 bg-black">
                    <img src={capturedImage} alt="Captured snapshot" className="w-full h-full object-cover" />
                  </div>
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[10px] font-mono uppercase tracking-widest text-white/40">
                      Detected Emotion
                    </span>
                    <span className="text-[10px] font-mono text-white/40">
                      {scanResult.inference_ms ? `${scanResult.inference_ms}ms` : ''}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-3xl sm:text-4xl">{activeConfig.emoji}</span>
                    <div>
                      <h3 className={`text-xl sm:text-2xl font-black uppercase tracking-tight ${activeConfig.textColor}`}>
                        {scanResult.emotion}
                      </h3>
                      <p className="text-xs text-white/50">
                        Confidence: <strong className="text-white">{(scanResult.confidence * 100).toFixed(1)}%</strong>
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Status Badges */}
              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div className="p-2.5 rounded-xl bg-white/[0.03] border border-white/[0.06] flex flex-col">
                  <span className="text-white/40 text-[9px] uppercase font-mono">ViT Raw Label</span>
                  <span className="font-semibold text-white/80 capitalize truncate">
                    {scanResult.raw_emotion || '—'}
                  </span>
                </div>
                <div className="p-2.5 rounded-xl bg-white/[0.03] border border-white/[0.06] flex flex-col">
                  <span className="text-white/40 text-[9px] uppercase font-mono">Face Detection</span>
                  <span className={`font-semibold truncate ${scanResult.face_detected ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {scanResult.face_detected ? '✓ Face Cropped' : '⚠ Full Frame'}
                  </span>
                </div>
              </div>

              {/* All 8-class Probabilities Breakdown */}
              <div className="flex flex-col gap-2 pt-2 border-t border-white/[0.08]">
                <div className="flex items-center justify-between text-xs font-semibold text-white/70 mb-1">
                  <span>Class Distribution (8 classes)</span>
                  <span className="text-[10px] font-mono text-white/40">Softmax Score</span>
                </div>

                <div className="space-y-1.5">
                  {Object.entries(scanResult.probabilities || {})
                    .sort(([, a], [, b]) => b - a)
                    .map(([label, score]) => {
                      const pct = Math.round(score * 100);
                      const isTop = label.toLowerCase() === (scanResult.raw_emotion || '').toLowerCase();

                      return (
                        <div key={label} className="flex flex-col gap-0.5">
                          <div className="flex items-center justify-between text-[11px]">
                            <span className={`capitalize ${isTop ? 'font-bold text-white' : 'text-white/60'}`}>
                              {label} {isTop && '★'}{' '}
                              {label.toLowerCase() === 'contempt' && (
                                <span className="text-[9px] text-white/30 font-normal italic">(ignored)</span>
                              )}
                            </span>
                            <span className={`font-mono ${isTop ? activeConfig.textColor + ' font-bold' : 'text-white/40'}`}>
                              {(score * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full h-2 rounded-full bg-white/[0.05] overflow-hidden p-[1px]">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                isTop
                                  ? `bg-gradient-to-r ${activeConfig.barColor}`
                                  : 'bg-white/20'
                              }`}
                              style={{ width: `${Math.max(pct, 2)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>

              {/* Retake Button */}
              <button
                onClick={captureAndScan}
                disabled={!isCameraActive || isScanning}
                className="w-full mt-2 py-2 rounded-xl bg-white/[0.06] hover:bg-white/10 text-white/80 hover:text-white font-medium text-xs transition-all border border-white/10 cursor-pointer flex items-center justify-center gap-1.5"
              >
                <span>🔄</span>
                <span>Scan Another Expression</span>
              </button>
            </div>
          ) : (
            /* Standby Card when no scan has been performed yet */
            <div className="p-8 rounded-2xl bg-white/[0.02] border border-white/[0.06] flex flex-col items-center justify-center text-center gap-3 min-h-[300px]">
              <div className="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/[0.08] flex items-center justify-center text-3xl">
                🎭
              </div>
              <h4 className="text-sm font-semibold text-white/80">Awaiting Scan</h4>
              <p className="text-xs text-white/40 max-w-xs leading-relaxed">
                Click <strong>"Take Pic & Scan Emotion"</strong> on the live feed or upload an image to see the detected facial emotion and probability breakdown.
              </p>
            </div>
          )}

          {/* ── Recent Scan History Strip ── */}
          {scanHistory.length > 0 && (
            <div className="p-3.5 rounded-2xl bg-white/[0.02] border border-white/[0.05] flex flex-col gap-2">
              <span className="text-[10px] uppercase font-mono tracking-wider text-white/40">
                Session History ({scanHistory.length})
              </span>
              <div className="flex gap-2 overflow-x-auto pb-1">
                {scanHistory.map((item) => {
                  const cfg = EMOTION_CONFIG[item.emotion?.toLowerCase()] || DEFAULT_CONFIG;
                  return (
                    <div
                      key={item.id}
                      className="flex-shrink-0 w-16 p-1.5 rounded-xl bg-white/[0.03] border border-white/[0.08] flex flex-col items-center gap-1 text-center"
                    >
                      <img src={item.preview} alt="" className="w-12 h-12 rounded-lg object-cover bg-black" />
                      <span className="text-sm leading-none">{cfg.emoji}</span>
                      <span className={`text-[10px] font-bold capitalize truncate max-w-full ${cfg.textColor}`}>
                        {item.emotion}
                      </span>
                      <span className="text-[9px] font-mono text-white/30">
                        {Math.round(item.confidence * 100)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
