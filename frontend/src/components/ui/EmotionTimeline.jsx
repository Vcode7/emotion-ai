import { useMemo, useState } from 'react';
import { getEmotionConfig } from './EmotionBadge';

export default function EmotionTimeline({ messages }) {
  const [hoveredIndex, setHoveredIndex] = useState(null);

  const emotionPoints = useMemo(() => {
    return messages
      .filter((msg) => msg.role === 'assistant' && msg.emotion)
      .map((msg, idx) => ({
        emotion: msg.emotion,
        config: getEmotionConfig(msg.emotion),
        content: msg.content?.substring(0, 60) || '',
        timestamp: msg.timestamp,
        index: idx,
      }));
  }, [messages]);

  if (emotionPoints.length < 2) return null;

  return (
    <div className="w-full px-4 py-3">
      <div className="flex items-center gap-1 text-[10px] text-white/30 font-medium mb-2 uppercase tracking-widest">
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
        Emotion Flow
      </div>

      <div className="relative flex items-center w-full h-8">
        {/* Connecting line */}
        <div className="absolute top-1/2 left-0 right-0 h-[2px] -translate-y-1/2 rounded-full overflow-hidden">
          <div
            className="w-full h-full"
            style={{
              background: emotionPoints.length >= 2
                ? `linear-gradient(to right, ${emotionPoints.map((p, i) => `${p.config.color} ${(i / (emotionPoints.length - 1)) * 100}%`).join(', ')})`
                : emotionPoints[0]?.config.color || '#475569',
              opacity: 0.4,
            }}
          />
        </div>

        {/* Dots */}
        {emotionPoints.map((point, idx) => {
          const left = emotionPoints.length === 1
            ? 50
            : (idx / (emotionPoints.length - 1)) * 100;

          return (
            <div
              key={idx}
              className="absolute -translate-x-1/2 -translate-y-1/2 top-1/2"
              style={{ left: `${left}%` }}
              onMouseEnter={() => setHoveredIndex(idx)}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              {/* Outer glow ring */}
              <div
                className="absolute inset-0 rounded-full transition-smooth"
                style={{
                  width: hoveredIndex === idx ? 24 : 16,
                  height: hoveredIndex === idx ? 24 : 16,
                  margin: hoveredIndex === idx ? -4 : 0,
                  background: point.config.color,
                  opacity: hoveredIndex === idx ? 0.2 : 0.1,
                  filter: 'blur(4px)',
                }}
              />

              {/* Dot */}
              <div
                className="relative w-4 h-4 rounded-full cursor-pointer transition-smooth border-2"
                style={{
                  backgroundColor: point.config.color,
                  borderColor: hoveredIndex === idx ? '#fff' : `${point.config.color}80`,
                  transform: hoveredIndex === idx ? 'scale(1.4)' : 'scale(1)',
                  boxShadow: `0 0 8px ${point.config.color}40`,
                }}
              />

              {/* Tooltip */}
              {hoveredIndex === idx && (
                <div
                  className="absolute bottom-full left-1/2 -translate-x-1/2 mb-3 animate-scale-in z-50 pointer-events-none"
                  style={{ minWidth: 120 }}
                >
                  <div className="glass-panel-dense px-3 py-2 text-center">
                    <div className="flex items-center justify-center gap-1.5 mb-1">
                      <span className="text-sm">{point.config.icon}</span>
                      <span
                        className="text-xs font-semibold"
                        style={{ color: point.config.color }}
                      >
                        {point.config.label}
                      </span>
                    </div>
                    <div className="text-[10px] text-white/40 truncate max-w-[140px]">
                      {point.content}...
                    </div>
                  </div>
                  {/* Arrow */}
                  <div
                    className="absolute left-1/2 -translate-x-1/2 -bottom-1 w-2 h-2 rotate-45"
                    style={{ background: 'rgba(255,255,255,0.06)', borderRight: '1px solid rgba(255,255,255,0.1)', borderBottom: '1px solid rgba(255,255,255,0.1)' }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
