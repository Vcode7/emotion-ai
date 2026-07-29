import { useRef, useEffect } from 'react';
import useAppStore from '../../store/useAppStore';
import useCameraRecorder from '../../hooks/useCameraRecorder';
import MessageBubble from './MessageBubble';
import TextInput from './TextInput';
import AudioRecorder from './AudioRecorder';
import CameraRecorder from '../Media/CameraRecorder';
import LoadingIndicator from '../ui/LoadingIndicator';
import EmotionTimeline from '../ui/EmotionTimeline';

export default function ChatArea() {
  const messages = useAppStore((s) => s.messages);
  const inputMode = useAppStore((s) => s.inputMode);
  const videoEnabled = useAppStore((s) => s.videoEnabled);
  const isProcessing = useAppStore((s) => s.isProcessing);

  const messagesEndRef = useRef(null);
  const cameraRecorder = useCameraRecorder();

  // Expose camera recorder via a ref-like object for child components
  const cameraRecorderRef = useRef(cameraRecorder);
  cameraRecorderRef.current = cameraRecorder;

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isProcessing]);

  return (
    <div className="relative flex flex-col h-full">
      {/* Emotion timeline */}
      {messages.length >= 2 && <EmotionTimeline messages={messages} />}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-1">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center animate-fade-in">
            <div className="w-16 h-16 mb-4 rounded-2xl bg-gradient-to-br from-accent-blue/10 to-accent-purple/10 border border-white/5 flex items-center justify-center">
              <svg
                className="w-8 h-8 text-accent-blue/60"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z"
                />
              </svg>
            </div>
            <h3 className="text-base font-semibold text-white/60 mb-1">
              Begin a Conversation
            </h3>
            <p className="text-sm text-white/30 max-w-xs">
              I&apos;ll analyze emotions in real-time and respond with empathy. The avatar will reflect the emotional tone.
            </p>
            <div className="flex items-center gap-4 mt-6 text-[11px] text-white/20">
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-blue/40" />
                Text
              </span>
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-purple/40" />
                Audio
              </span>
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-cyan/40" />
                Video
              </span>
            </div>
          </div>
        )}

        {messages.map((msg, index) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            isLatest={index === messages.length - 1}
          />
        ))}

        {isProcessing && <LoadingIndicator />}

        <div ref={messagesEndRef} />
      </div>

      {/* Camera preview overlay */}
      {videoEnabled && (
        <CameraRecorder
          stream={cameraRecorder.stream}
          isRecording={cameraRecorder.isRecording}
        />
      )}

      {/* Input area */}
      <div className="flex-shrink-0 border-t border-white/[0.05]">
        {inputMode === 'text' ? (
          <TextInput cameraRecorderRef={cameraRecorderRef} />
        ) : (
          <AudioRecorder cameraRecorderRef={cameraRecorderRef} />
        )}
      </div>
    </div>
  );
}
