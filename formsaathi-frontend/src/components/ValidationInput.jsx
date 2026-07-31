import { useState, useCallback, useRef } from 'react';
import { validateField } from '../api';

export default function ValidationInput({ onSend, disabled, currentField }) {
  const [value, setValue] = useState('');
  const [validation, setValidation] = useState(null); // { valid, error, suggestion }
  const [validating, setValidating] = useState(false);
  const debounceRef = useRef(null);

  const runValidation = useCallback(async (text) => {
    if (!currentField || !text.trim()) { setValidation(null); return; }
    setValidating(true);
    try {
      const result = await validateField(currentField, text);
      if (result) setValidation(result);
    } catch { /* silent */ } finally {
      setValidating(false);
    }
  }, [currentField]);

  const handleChange = (e) => {
    const v = e.target.value;
    setValue(v);
    setValidation(null);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => runValidation(v), 700);
  };

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
    setValidation(null);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const borderColor = validation
    ? validation.valid ? 'border-forest-400 focus:border-forest-500' : 'border-red-400 focus:border-red-500'
    : 'border-[var(--border)] focus:border-saffron-400';

  return (
    <div className="border-t border-[var(--border)] bg-white px-4 pt-3 pb-4 space-y-2">
      {/* Validation feedback */}
      {validation && (
        <div className={`text-xs px-3 py-2 rounded-xl animate-slide-up flex items-start gap-2
          ${validation.valid ? 'bg-forest-50 text-forest-700' : 'bg-red-50 text-red-700'}`}>
          <span className="mt-0.5 flex-shrink-0">
            {validation.valid ? '✓' : '✗'}
          </span>
          <span>
            {validation.valid ? 'Looks good!' : validation.error}
            {validation.suggestion && (
              <span className="ml-1 text-[var(--text-secondary)]">— {validation.suggestion}</span>
            )}
          </span>
        </div>
      )}

      <div className="flex items-end gap-2">
        <div className="flex-1 relative">
          <textarea
            rows={1}
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder={currentField ? `Enter value for ${currentField}…` : 'Type your message…'}
            className={`w-full resize-none rounded-xl border px-3.5 py-2.5 text-sm bg-[var(--bg)]
              text-[var(--text-primary)] placeholder:text-[var(--text-secondary)]
              focus:outline-none focus:bg-white transition-colors duration-150
              disabled:opacity-50 max-h-32 ${borderColor}`}
            style={{ height: 'auto', minHeight: '42px' }}
            onInput={(e) => {
              e.target.style.height = 'auto';
              e.target.style.height = Math.min(e.target.scrollHeight, 128) + 'px';
            }}
          />
          {validating && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <svg className="animate-spin w-3.5 h-3.5 text-saffron-400" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.2"/>
                <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/>
              </svg>
            </div>
          )}
        </div>

        <button
          onClick={handleSend}
          disabled={!value.trim() || disabled}
          className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-all duration-150
            ${value.trim() && !disabled
              ? 'bg-saffron-500 text-white hover:bg-saffron-600 active:scale-95 shadow-sm shadow-saffron-200'
              : 'bg-[var(--bg)] text-[var(--text-secondary)]'}`}
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
          </svg>
        </button>
      </div>
    </div>
  );
}
