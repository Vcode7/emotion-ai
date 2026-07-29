import EmotionBadge from '../ui/EmotionBadge';
import VoicePlayer from '../Media/VoicePlayer';
import useAppStore from '../../store/useAppStore';

function formatTime(timestamp) {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default function MessageBubble({ message }) {
  const outputModes = useAppStore((s) => s.outputModes);
  const isUser = message.role === 'user';

  return (
    <div
      className={`flex w-full mb-4 ${isUser ? 'justify-end' : 'justify-start'}`}
      style={{
        animation: isUser
          ? 'slideInRight 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards'
          : 'slideInLeft 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards',
      }}
    >
      <div
        className={`
          max-w-[85%] md:max-w-[75%] rounded-2xl px-4 py-3
          ${isUser
            ? 'bg-gradient-to-br from-accent-blue/80 to-accent-purple/80 text-white rounded-br-md'
            : 'glass-panel-dense text-white/90 rounded-bl-md'
          }
        `}
      >
        {/* ── User message ── */}
        {isUser && (
          <div>
            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {message.content}
            </p>
            {message.audioUrl && (
              <div className="mt-2.5">
                <audio
                  src={message.audioUrl}
                  controls
                  className="w-full max-w-[240px] h-8 rounded-lg outline-none"
                />
              </div>
            )}
          </div>
        )}

        {/* ── Assistant message ── */}
        {!isUser && (
          <div>
            {/* If we have segments, render them */}
            {message.segments && message.segments.length > 0 ? (
              <div className="space-y-3">
                {message.segments.map((segment, idx) => (
                  <div key={idx}>
                    {/* Emotion badge for each segment */}
                    {segment.emotion && (
                      <div className="mb-1.5">
                        <EmotionBadge emotion={segment.emotion} />
                      </div>
                    )}
                    <p className="text-sm leading-relaxed whitespace-pre-wrap text-white/85">
                      {segment.text || segment.content || ''}
                    </p>
                    {idx < message.segments.length - 1 && (
                      <div className="border-b border-white/5 mt-3" />
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div>
                {/* Single emotion badge */}
                {message.emotion && message.emotion !== 'neutral' && (
                  <div className="mb-2">
                    <EmotionBadge emotion={message.emotion} />
                  </div>
                )}
                <p className="text-sm leading-relaxed whitespace-pre-wrap text-white/85">
                  {message.content}
                </p>
              </div>
            )}

            {/* Voice player */}
            {outputModes.voice && message.segments && (
              <VoicePlayer
                segments={message.segments}
                voiceParams={message.voiceParams}
              />
            )}
          </div>
        )}

        {/* Timestamp */}
        <div
          className={`mt-2 text-[10px] ${isUser ? 'text-white/40 text-right' : 'text-white/30'}`}
        >
          {formatTime(message.timestamp)}
        </div>
      </div>
    </div>
  );
}
