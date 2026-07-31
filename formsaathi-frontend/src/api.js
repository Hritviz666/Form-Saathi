const BASE = 'http://localhost:8000';

export async function uploadForm(file, mode) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('mode', mode);
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

export async function sendChat(sessionId, message, mode) {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message, mode }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
  return res.json();
}

export async function validateField(fieldName, value) {
  const res = await fetch(`${BASE}/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ field_name: fieldName, value }),
  });
  if (!res.ok) return null;
  return res.json();
}

export async function getSession(sessionId) {
  const res = await fetch(`${BASE}/session/${sessionId}`);
  if (!res.ok) return null;
  return res.json();
}

export async function deleteSession(sessionId) {
  await fetch(`${BASE}/session/${sessionId}`, { method: 'DELETE' });
}

export async function checkHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) return null;
  return res.json();
}
