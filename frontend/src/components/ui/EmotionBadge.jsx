const EMOTION_CONFIG = {
  neutral: {
    color: '#94a3b8',
    bgClass: 'bg-slate-500/15',
    textClass: 'text-slate-300',
    borderClass: 'border-slate-500/30',
    icon: '😐',
    label: 'Neutral',
  },
  happy: {
    color: '#f59e0b',
    bgClass: 'bg-amber-500/15',
    textClass: 'text-amber-300',
    borderClass: 'border-amber-500/30',
    icon: '😊',
    label: 'Happy',
  },
  sad: {
    color: '#3b82f6',
    bgClass: 'bg-blue-500/15',
    textClass: 'text-blue-300',
    borderClass: 'border-blue-500/30',
    icon: '😢',
    label: 'Sad',
  },
  compassion: {
    color: '#8b5cf6',
    bgClass: 'bg-purple-500/15',
    textClass: 'text-purple-300',
    borderClass: 'border-purple-500/30',
    icon: '💜',
    label: 'Compassion',
  },
  concerned: {
    color: '#f97316',
    bgClass: 'bg-orange-500/15',
    textClass: 'text-orange-300',
    borderClass: 'border-orange-500/30',
    icon: '😟',
    label: 'Concerned',
  },
  supportive: {
    color: '#10b981',
    bgClass: 'bg-emerald-500/15',
    textClass: 'text-emerald-300',
    borderClass: 'border-emerald-500/30',
    icon: '🤗',
    label: 'Supportive',
  },
  encouraging: {
    color: '#34d399',
    bgClass: 'bg-emerald-400/15',
    textClass: 'text-emerald-300',
    borderClass: 'border-emerald-400/30',
    icon: '💪',
    label: 'Encouraging',
  },
  thinking: {
    color: '#eab308',
    bgClass: 'bg-yellow-500/15',
    textClass: 'text-yellow-300',
    borderClass: 'border-yellow-500/30',
    icon: '🤔',
    label: 'Thinking',
  },
  angry: {
    color: '#ef4444',
    bgClass: 'bg-red-500/15',
    textClass: 'text-red-300',
    borderClass: 'border-red-500/30',
    icon: '😠',
    label: 'Angry',
  },
  surprised: {
    color: '#ec4899',
    bgClass: 'bg-pink-500/15',
    textClass: 'text-pink-300',
    borderClass: 'border-pink-500/30',
    icon: '😮',
    label: 'Surprised',
  },
  fearful: {
    color: '#a78bfa',
    bgClass: 'bg-violet-400/15',
    textClass: 'text-violet-300',
    borderClass: 'border-violet-400/30',
    icon: '😨',
    label: 'Fearful',
  },
};

function getEmotionConfig(emotion) {
  const key = (emotion || 'neutral').toLowerCase();
  return EMOTION_CONFIG[key] || EMOTION_CONFIG.neutral;
}

export default function EmotionBadge({ emotion }) {
  const config = getEmotionConfig(emotion);

  return (
    <span
      className={`
        inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium
        ${config.bgClass} ${config.textClass} border ${config.borderClass}
        transition-smooth
      `}
      style={{
        boxShadow: `0 0 12px ${config.color}15`,
      }}
    >
      <span className="text-xs leading-none">{config.icon}</span>
      <span>{config.label}</span>
    </span>
  );
}

export { getEmotionConfig, EMOTION_CONFIG };
