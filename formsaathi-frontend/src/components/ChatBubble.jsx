export default function ChatBubble({ message }) {
  const isAgent = message.role === 'agent';

  return (
    <div className={`flex items-end gap-2 animate-slide-up ${isAgent ? 'justify-start' : 'justify-end'}`}>
      {isAgent && (
        <div className="w-7 h-7 rounded-full bg-saffron-500 flex items-center justify-center flex-shrink-0 mb-1">
          <span className="text-white font-display font-bold text-xs">FS</span>
        </div>
      )}
      <div
        className={`max-w-[82%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap
          ${isAgent
            ? 'bg-[var(--agent-bubble)] border border-saffron-100 text-[var(--text-primary)] rounded-bl-sm'
            : 'bg-[var(--user-bubble)] text-white rounded-br-sm'}`}
      >
        {message.text}
        {message.timestamp && (
          <p className={`text-[10px] mt-1.5 ${isAgent ? 'text-[var(--text-secondary)]' : 'text-indigo-200'}`}>
            {message.timestamp}
          </p>
        )}
      </div>
    </div>
  );
}
