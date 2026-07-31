export default function SessionHeader({ formType, mode, onReset }) {
  return (
    <div className="bg-white border-b border-[var(--border)] px-4 py-3 flex items-center justify-between">
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-saffron-500 flex items-center justify-center flex-shrink-0">
          <span className="text-white font-display font-bold text-xs">FS</span>
        </div>
        <div>
          <p className="font-display font-semibold text-sm text-[var(--text-primary)] leading-tight">
            {formType || 'FormSaathi'}
          </p>
          <p className="text-[10px] text-[var(--text-secondary)] leading-tight capitalize">
            {mode === 'guided' ? '📋 Guided Mode' : '💬 Free Query'}
          </p>
        </div>
      </div>

      <button
        onClick={onReset}
        className="text-xs text-[var(--text-secondary)] hover:text-saffron-600 font-medium px-2 py-1 
          rounded-lg hover:bg-saffron-50 transition-colors"
      >
        New form
      </button>
    </div>
  );
}
