export default function ProgressBar({ current, total, fieldName }) {
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;

  return (
    <div className="bg-white border-b border-[var(--border)] px-4 py-3 space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="font-display font-semibold text-[var(--text-primary)]">
          {fieldName
            ? <span>Field: <span className="text-saffron-600">{fieldName}</span></span>
            : 'Starting…'}
        </span>
        <span className="text-[var(--text-secondary)]">{current} / {total} fields</span>
      </div>
      <div className="h-1.5 bg-[var(--bg)] rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-saffron-400 to-saffron-500 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
