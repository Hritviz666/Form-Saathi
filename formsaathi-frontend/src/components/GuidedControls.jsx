export default function GuidedControls({ onCommand, disabled }) {
  const commands = [
    { label: 'Next →', cmd: 'next', style: 'bg-saffron-500 text-white hover:bg-saffron-600' },
    { label: 'Skip', cmd: 'skip', style: 'bg-[var(--bg)] text-[var(--text-secondary)] hover:bg-saffron-50 border border-[var(--border)]' },
    { label: '← Back', cmd: 'back', style: 'bg-[var(--bg)] text-[var(--text-secondary)] hover:bg-saffron-50 border border-[var(--border)]' },
  ];

  return (
    <div className="flex gap-2 px-4 pb-2">
      {commands.map(({ label, cmd, style }) => (
        <button
          key={cmd}
          onClick={() => onCommand(cmd)}
          disabled={disabled}
          className={`flex-1 py-2 rounded-xl text-xs font-display font-semibold transition-all duration-150 
            active:scale-95 disabled:opacity-40 ${style}`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
