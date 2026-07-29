import { useState, useRef, useCallback, useEffect } from 'react';
import useAppStore from '../../store/useAppStore';
import { chatText, chatTextVideo } from '../../api/client';
import { v4 as uuidv4 } from 'uuid';

export default function TextInput({ cameraRecorderRef }) {
  const [text, setText] = useState('');
  const textareaRef = useRef(null);
  const hasStartedCamera = useRef(false);

  const {
    videoEnabled,
    isProcessing,
    sessionId,
    llmProvider,
    llmApiKey,
    addMessage,
    setIsProcessing,
    setCurrentEmotion,
    setCurrentAvatarCommands,
  } = useAppStore();

  // Auto-grow textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [text]);

  const handleChange = useCallback(
    (e) => {
      setText(e.target.value);

      // Start camera on first keystroke if video enabled
      if (videoEnabled && !hasStartedCamera.current && e.target.value.length === 1) {
        hasStartedCamera.current = true;
        cameraRecorderRef?.current?.startRecording?.();
      }
    },
    [videoEnabled, cameraRecorderRef]
  );

  const handleSubmit = useCallback(async () => {
    const trimmed = text.trim();
    if (!trimmed || isProcessing) return;

    // Add user message
    addMessage({
      id: uuidv4(),
      role: 'user',
      content: trimmed,
      timestamp: Date.now(),
    });

    setText('');
    hasStartedCamera.current = false;
    setIsProcessing(true);

    try {
      let response;

      if (videoEnabled && cameraRecorderRef?.current) {
        // Stop camera and get video blob
        cameraRecorderRef.current.stopRecording();

        // Wait a bit for the blob to be available
        await new Promise((resolve) => setTimeout(resolve, 500));

        const videoBlob = cameraRecorderRef.current.videoBlob;
        if (videoBlob) {
          response = await chatTextVideo(trimmed, videoBlob, sessionId, llmProvider, llmApiKey);
        } else {
          response = await chatText(trimmed, sessionId, llmProvider, llmApiKey);
        }
      } else {
        response = await chatText(trimmed, sessionId, llmProvider, llmApiKey);
      }

      // Process response
      if (response) {
        const emotion = response.emotion || response.primary_emotion || 'neutral';
        const segments = response.segments || response.response || [];
        const avatarCommands = response.avatar_commands || null;
        const voiceParams = response.voice_params || null;

        const contentText = Array.isArray(segments)
          ? segments.map((s) => s.text || s.content || '').join(' ')
          : typeof response.response === 'string'
            ? response.response
            : response.text || '';

        setCurrentEmotion(emotion);
        setCurrentAvatarCommands(avatarCommands);

        addMessage({
          id: uuidv4(),
          role: 'assistant',
          content: contentText,
          emotion,
          emotionState: response.emotion_state || null,
          segments: Array.isArray(segments) ? segments : null,
          avatarCommands,
          voiceParams,
          timestamp: Date.now(),
        });
      }
    } catch (err) {
      console.error('Chat error:', err);
      addMessage({
        id: uuidv4(),
        role: 'assistant',
        content: 'I apologize, but I encountered an error processing your message. Please check your API key and try again.',
        emotion: 'concerned',
        timestamp: Date.now(),
      });
      setCurrentEmotion('concerned');
    } finally {
      setIsProcessing(false);
    }
  }, [
    text,
    isProcessing,
    videoEnabled,
    sessionId,
    llmProvider,
    llmApiKey,
    addMessage,
    setIsProcessing,
    setCurrentEmotion,
    setCurrentAvatarCommands,
    cameraRecorderRef,
  ]);

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  const isDisabled = isProcessing || !llmApiKey;

  const placeholder = !llmApiKey
    ? 'Enter your API key in settings to begin...'
    : isProcessing
      ? 'Processing your message...'
      : 'Type your message... (Shift+Enter for new line)';

  return (
    <div className="flex items-end gap-3 px-4 py-3">
      <div className="flex-1 relative">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isDisabled}
          rows={1}
          className={`
            w-full resize-none rounded-xl px-4 py-3 text-sm
            bg-white/[0.04] border border-white/[0.08]
            text-white/90 placeholder:text-white/25
            focus:border-accent-blue/40 focus:bg-white/[0.06]
            transition-smooth
            ${isDisabled ? 'opacity-40 cursor-not-allowed' : ''}
          `}
          style={{ maxHeight: 120, minHeight: 44 }}
        />

        {/* Video indicator */}
        {videoEnabled && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1 text-accent-cyan/60">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
        )}
      </div>

      {/* Send button */}
      <button
        onClick={handleSubmit}
        disabled={isDisabled || !text.trim()}
        className={`
          flex-shrink-0 w-11 h-11 rounded-xl flex items-center justify-center
          transition-smooth btn-hover
          ${text.trim() && !isDisabled
            ? 'bg-gradient-to-br from-accent-blue to-accent-purple text-white shadow-lg shadow-accent-blue/20'
            : 'bg-white/[0.04] text-white/20 border border-white/[0.06] cursor-not-allowed'
          }
        `}
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
        </svg>
      </button>
    </div>
  );
}
