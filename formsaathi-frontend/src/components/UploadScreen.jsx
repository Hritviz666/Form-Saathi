import { useState, useRef, useCallback } from 'react';
import { uploadForm } from '../api';

const FORM_ICON = (
  <svg viewBox="0 0 48 48" fill="none" className="w-12 h-12" xmlns="http://www.w3.org/2000/svg">
    <rect x="8" y="4" width="32" height="40" rx="3" fill="#fff0d9" stroke="#f07c00" strokeWidth="2"/>
    <line x1="14" y1="14" x2="34" y2="14" stroke="#f07c00" strokeWidth="2" strokeLinecap="round"/>
    <line x1="14" y1="20" x2="34" y2="20" stroke="#d4c9bc" strokeWidth="2" strokeLinecap="round"/>
    <line x1="14" y1="26" x2="26" y2="26" stroke="#d4c9bc" strokeWidth="2" strokeLinecap="round"/>
    <circle cx="34" cy="34" r="8" fill="#f07c00"/>
    <line x1="34" y1="30" x2="34" y2="38" stroke="white" strokeWidth="2" strokeLinecap="round"/>
    <line x1="30" y1="34" x2="38" y2="34" stroke="white" strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

export default function UploadScreen({ onSessionStart }) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState('guided');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef();

  const handleFile = useCallback((f) => {
    if (!f) return;
    const ok = f.type === 'application/pdf' || f.type.startsWith('image/');
    if (!ok) { setError('Only images (JPG, PNG) or PDF files are supported.'); return; }
    setError('');
    setFile(f);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const onSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    try {
      const data = await uploadForm(file, mode);
      onSessionStart({ ...data, mode });
    } catch (e) {
      setError('Could not connect to backend. Make sure the server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-dvh flex flex-col items-center justify-center px-4 py-10 animate-fade-in">
      {/* Header */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 mb-3">
          <div className="w-8 h-8 rounded-lg bg-saffron-500 flex items-center justify-center">
            <span className="text-white font-display font-bold text-sm">FS</span>
          </div>
          <h1 className="font-display font-bold text-2xl text-[var(--text-primary)]">FormSaathi</h1>
        </div>
        <p className="text-[var(--text-secondary)] text-sm max-w-xs">
          Upload any government or bank form. We'll help you fill it, step by step.
        </p>
      </div>

      <div className="w-full max-w-md space-y-4">
        {/* Drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current.click()}
          className={`relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 
            ${dragging ? 'border-saffron-400 bg-saffron-50 scale-[1.01]' : 
              file ? 'border-forest-400 bg-forest-50' : 
              'border-[var(--border)] bg-white hover:border-saffron-300 hover:bg-saffron-50'}`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*,.pdf"
            className="hidden"
            onChange={(e) => handleFile(e.target.files[0])}
          />
          
          {file ? (
            <div className="space-y-2">
              <div className="inline-flex items-center gap-2 bg-forest-100 text-forest-700 px-3 py-1.5 rounded-full text-sm font-medium">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {file.name}
              </div>
              <p className="text-xs text-[var(--text-secondary)]">Click to change file</p>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex justify-center">{FORM_ICON}</div>
              <div>
                <p className="font-display font-semibold text-[var(--text-primary)]">Drop your form here</p>
                <p className="text-sm text-[var(--text-secondary)] mt-1">or click to browse</p>
              </div>
              <p className="text-xs text-[var(--text-secondary)] bg-[var(--bg)] rounded-lg px-3 py-1.5 inline-block">
                JPG · PNG · PDF
              </p>
            </div>
          )}
        </div>

        {/* Mode selector */}
        <div className="bg-white rounded-2xl border border-[var(--border)] p-1 flex gap-1">
          {[
            { id: 'guided', label: 'Guided Mode', desc: 'Field by field' },
            { id: 'free', label: 'Free Query', desc: 'Ask anything' },
          ].map((m) => (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className={`flex-1 rounded-xl py-2.5 px-3 text-left transition-all duration-150
                ${mode === m.id ? 'bg-saffron-500 text-white shadow-sm' : 'text-[var(--text-secondary)] hover:bg-[var(--bg)]'}`}
            >
              <p className={`text-sm font-display font-semibold ${mode === m.id ? 'text-white' : ''}`}>{m.label}</p>
              <p className={`text-xs ${mode === m.id ? 'text-saffron-100' : 'text-[var(--text-secondary)]'}`}>{m.desc}</p>
            </button>
          ))}
        </div>

        {/* Mode description */}
        <p className="text-xs text-[var(--text-secondary)] px-1">
          {mode === 'guided'
            ? '📋 The agent will walk through each field one by one, explain what to fill, and validate your input.'
            : '💬 Upload the form and ask any question about it freely — like "what is AO code?" or "which fields are mandatory?"'}
        </p>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 text-sm rounded-xl px-4 py-3">
            {error}
          </div>
        )}

        <button
          onClick={onSubmit}
          disabled={!file || loading}
          className={`w-full py-4 rounded-2xl font-display font-bold text-base transition-all duration-200
            ${file && !loading
              ? 'bg-saffron-500 text-white hover:bg-saffron-600 active:scale-[0.98] shadow-lg shadow-saffron-200'
              : 'bg-[var(--border)] text-[var(--text-secondary)] cursor-not-allowed'}`}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.2"/>
                <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/>
              </svg>
              Analysing form…
            </span>
          ) : 'Start Filling →'}
        </button>
      </div>

      <p className="mt-10 text-xs text-[var(--text-secondary)] text-center max-w-xs">
        Supports PAN, Aadhaar, passport, bank KYC, and 8 more form types. Sensitive fields processed on-device.
      </p>
    </div>
  );
}
