export default function TypingIndicator() {
  return (
    <div className="flex items-end gap-2 animate-fade-in">
      <div className="w-7 h-7 rounded-full bg-saffron-500 flex items-center justify-center flex-shrink-0">
        <span className="text-white font-display font-bold text-xs">FS</span>
      </div>
      <div className="bg-[var(--agent-bubble)] border border-saffron-100 rounded-2xl rounded-bl-sm px-4 py-3">
        <div className="flex items-center gap-1.5">
          <span className="typing-dot w-2 h-2 rounded-full bg-saffron-400 inline-block" />
          <span className="typing-dot w-2 h-2 rounded-full bg-saffron-400 inline-block" />
          <span className="typing-dot w-2 h-2 rounded-full bg-saffron-400 inline-block" />
        </div>
      </div>
    </div>
  );
}
