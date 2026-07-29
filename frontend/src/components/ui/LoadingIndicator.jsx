export default function LoadingIndicator() {
  return (
    <div className="flex items-center justify-start px-4 py-3 animate-fade-in">
      <div className="glass-panel-dense px-5 py-3 flex items-center gap-3">
        {/* Bouncing dots */}
        <div className="flex items-center gap-1.5">
          <span
            className="w-2 h-2 rounded-full bg-accent-blue animate-bounce-dot"
            style={{ animationDelay: '0s' }}
          />
          <span
            className="w-2 h-2 rounded-full bg-accent-purple animate-bounce-dot"
            style={{ animationDelay: '0.15s' }}
          />
          <span
            className="w-2 h-2 rounded-full bg-accent-cyan animate-bounce-dot"
            style={{ animationDelay: '0.3s' }}
          />
        </div>
        <span className="text-xs text-white/50 font-medium">
          Analyzing emotions...
        </span>
      </div>
    </div>
  );
}
