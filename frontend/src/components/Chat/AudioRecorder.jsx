import { useCallback, useEffect } from 'react';
import useAppStore from '../../store/useAppStore';
import useAudioRecorder from '../../hooks/useAudioRecorder';
import { chatAudio, chatAudioVideo } from '../../api/client';
import { v4 as uuidv4 } from 'uuid';

export default function AudioRecorder({ cameraRecorderRef }) {
  const {
    videoEnabled,
    isProcessing,
    sessionId,
    llmProvider,
    llmApiKey,
    addMessage,
    updateMessage,
    setIsProcessing,
    setIsRecording: setStoreRecording,
    setCurrentEmotion,
    setCurrentAvatarCommands,
  } = useAppStore();

  const { isRecording, startRecording, stopRecording, audioBlob, duration } =
    useAudioRecorder();

  // Sync recording state to store
  useEffect(() => {
    setStoreRecording(isRecording);
  }, [isRecording, setStoreRecording]);

  // Process audio blob when recording stops
  useEffect(() => {
    if (!audioBlob || isProcessing) return;

    const processAudio = async () => {
      setIsProcessing(true);

      const userMessageId = uuidv4();
      // Add user message placeholder
      addMessage({
        id: userMessageId,
        role: 'user',
        content: '🎤 Voice message',
        audioUrl: null,
        timestamp: Date.now(),
      });

      try {
        let response;

        if (videoEnabled && cameraRecorderRef?.current?.videoBlob) {
          const videoBlob = cameraRecorderRef.current.videoBlob;
          response = await chatAudioVideo(audioBlob, videoBlob, sessionId, llmProvider, llmApiKey);
        } else {
          response = await chatAudio(audioBlob, sessionId, llmProvider, llmApiKey);
        }

        if (response) {
          const emotion = response.emotion || response.primary_emotion || 'neutral';
          const segments = response.segments || response.response || [];
          const avatarCommands = response.avatar_commands || null;
          const voiceParams = response.voice_params || null;

          // Update the user message with transcribed text and audio url if available
          const transcribedText = response.transcript || response.transcribed_text || response.user_text;
          updateMessage(userMessageId, {
            content: transcribedText || '🎤 Voice message',
            audioUrl: response.audio_url || null,
          });

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
        console.error('Audio chat error:', err);
        updateMessage(userMessageId, {
          content: '🎤 Voice message (failed to send)',
        });
        addMessage({
          id: uuidv4(),
          role: 'assistant',
          content: 'I apologize, but I could not process your voice message. Please check your API key and try again.',
          emotion: 'concerned',
          timestamp: Date.now(),
        });
        setCurrentEmotion('concerned');
      } finally {
        setIsProcessing(false);
      }
    };

    processAudio();
  }, [audioBlob]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleToggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
      // Also stop camera if enabled
      if (videoEnabled && cameraRecorderRef?.current) {
        cameraRecorderRef.current.stopRecording();
      }
    } else {
      startRecording();
      // Also start camera if enabled
      if (videoEnabled && cameraRecorderRef?.current) {
        cameraRecorderRef.current.startRecording();
      }
    }
  }, [isRecording, startRecording, stopRecording, videoEnabled, cameraRecorderRef]);

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const isDisabled = isProcessing || !llmApiKey;

  return (
    <div className="flex flex-col items-center gap-4 py-6 px-4">
      {/* Duration display */}
      {isRecording && (
        <div className="flex items-center gap-2 animate-fade-in">
          <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          <span className="text-lg font-mono font-semibold text-white/80 tracking-wider">
            {formatDuration(duration)}
          </span>
        </div>
      )}

      {/* Mic button */}
      <button
        onClick={handleToggleRecording}
        disabled={isDisabled}
        className={`
          relative w-16 h-16 rounded-full flex items-center justify-center
          transition-smooth
          ${isDisabled ? 'opacity-30 cursor-not-allowed' : 'cursor-pointer'}
          ${isRecording
            ? 'bg-red-500/20 border-2 border-red-500/50 recording-pulse'
            : 'bg-white/[0.06] border-2 border-white/[0.1] hover:border-accent-blue/40 hover:bg-white/[0.1]'
          }
        `}
      >
        {isRecording ? (
          /* Stop icon */
          <div className="w-5 h-5 rounded-sm bg-red-400" />
        ) : (
          /* Mic icon */
          <svg
            className="w-6 h-6 text-white/70"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z"
            />
          </svg>
        )}

        {/* Pulse rings when recording */}
        {isRecording && (
          <>
            <span className="absolute inset-0 rounded-full border-2 border-red-400/30 animate-ping" />
          </>
        )}
      </button>

      {/* Label */}
      <span className="text-xs text-white/30 font-medium">
        {isDisabled
          ? 'Set API key in settings'
          : isRecording
            ? 'Tap to stop recording'
            : 'Tap to start recording'}
      </span>
    </div>
  );
}
