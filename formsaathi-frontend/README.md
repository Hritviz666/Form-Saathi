# FormSaathi — React PWA Frontend

## Setup

```bash
npm install
npm run dev        # starts on http://localhost:5173
```

Make sure the FastAPI backend is running on port 8000 before opening the app:
```bash
d:\FormSaathi\Saathi\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Structure

```
src/
├── api.js                  ← All fetch() calls to the backend
├── App.jsx                 ← Screen router (Upload ↔ Chat)
└── components/
    ├── UploadScreen.jsx    ← Drag-drop upload + mode selector
    ├── ChatScreen.jsx      ← Main chat view (both modes)
    ├── ChatBubble.jsx      ← Individual message bubble
    ├── TypingIndicator.jsx ← Animated dots while agent responds
    ├── ValidationInput.jsx ← Textarea with live POST /validate
    ├── ProgressBar.jsx     ← Guided mode field progress
    ├── GuidedControls.jsx  ← Next / Skip / Back buttons
    └── SessionHeader.jsx   ← Top bar with form name + mode
```

## PWA
- `public/manifest.json` — app metadata and icons
- `public/sw.js` — service worker (caches app shell, skips API calls)

## Build for production
```bash
npm run build     # output in dist/
```
Copy `dist/` to wherever you serve the frontend from, or serve with:
```bash
npx serve dist
```
