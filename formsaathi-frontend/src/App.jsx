import { useState } from 'react';
import UploadScreen from './components/UploadScreen';
import ChatScreen from './components/ChatScreen';

export default function App() {
  const [session, setSession] = useState(null);

  const handleSessionStart = (data) => setSession(data);
  const handleReset = () => setSession(null);

  return session
    ? <ChatScreen session={session} onReset={handleReset} />
    : <UploadScreen onSessionStart={handleSessionStart} />;
}
