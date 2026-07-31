import { useState, useEffect, useRef, useCallback } from 'react';
import { sendChat } from '../api';
import ChatBubble from './ChatBubble';
import TypingIndicator from './TypingIndicator';
import ValidationInput from './ValidationInput';
import ProgressBar from './ProgressBar';
import GuidedControls from './GuidedControls';
import SessionHeader from './SessionHeader';

function timestamp() {
  return new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
}

export default function ChatScreen({ session, onReset }) {
  const { session_id, form_type, mode, message: firstMessage, field_count, current_field } = session;

  const [messages, setMessages] = useState([
    { role: 'agent', text: firstMessage, timestamp: timestamp() }
  ]);
  const [typing, setTyping] = useState(false);
  const [currentField, setCurrentField] = useState(current_field || null);
  const [totalFields, setTotalFields] = useState(field_count || 0);
  const [filledCount, setFilledCount] = useState(0);
  const [complete, setComplete] = useState(false);
  const bottomRef = useRef();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typing]);

  const sendMessage = useCallback(async (text) => {
    setMessages(prev => [...prev, { role: 'user', text, timestamp: timestamp() }]);
    setTyping(true);
    try {
      const data = await sendChat(session_id, text, mode);
      setMessages(prev => [...prev, { role: 'agent', text: data.response, timestamp: timestamp() }]);
      if (data.current_field) setCurrentField(data.current_field);
      if (data.total_fields) setTotalFields(data.total_fields);
      if (data.current_field && data.total_fields) {
        // derive filled count from field index if available
        const idx = parseInt(data.current_field_index ?? filledCount);
        if (!isNaN(idx)) setFilledCount(idx);
      }
      if (data.is_complete) setComplete(true);
    } catch {
      setMessages(prev => [...prev, {
        role: 'agent',
        text: 'Sorry, something went wrong. Please try again.',
        timestamp: timestamp()
      }]);
    } finally {
      setTyping(false);
    }
  }, [session_id, mode, filledCount]);

  const handleCommand = (cmd) => sendMessage(cmd);

  return (
    <div className="flex flex-col h-dvh">
      <SessionHeader formType={form_type} mode={mode} onReset={onReset} />

      {mode === 'guided' && (
        <ProgressBar
          current={filledCount}
          total={totalFields}
          fieldName={currentField}
        />
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {messages.map((m, i) => <ChatBubble key={i} message={m} />)}
        {typing && <TypingIndicator />}

        {complete && (
          <div className="animate-slide-up text-center py-4">
            <div className="inline-block bg-forest-50 border border-forest-200 rounded-2xl px-5 py-4">
              <p className="text-2xl mb-1">🎉</p>
              <p className="font-display font-bold text-forest-700 text-sm">Form Complete!</p>
              <p className="text-xs text-forest-600 mt-0.5">All fields have been filled.</p>
              <button
                onClick={onReset}
                className="mt-3 bg-forest-500 text-white text-xs font-semibold px-4 py-2 rounded-xl hover:bg-forest-600 transition-colors"
              >
                Upload another form
              </button>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      {!complete && (
        <div className="flex-shrink-0">
          {mode === 'guided' && (
            <GuidedControls onCommand={handleCommand} disabled={typing} />
          )}
          <ValidationInput
            onSend={sendMessage}
            disabled={typing}
            currentField={mode === 'guided' ? currentField : null}
          />
        </div>
      )}
    </div>
  );
}
