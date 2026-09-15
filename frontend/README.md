# AI Data Scientist frontend

Milestone 10A provides the local React interface for dataset upload, preview,
target recommendations, and target validation.

## Requirements

- Node.js 22.13 or newer
- FastAPI backend running at `http://127.0.0.1:8000`

## Run locally

```bash
cp .env.example .env
npm install
npm run dev
```

Open `http://127.0.0.1:5173` in a Windows browser.

To use another backend URL, change `VITE_API_BASE_URL` in `.env`. The backend
origin must also be listed in `AI_DATA_SCIENTIST_CORS_ORIGINS`.

## Production build

```bash
npm run build
```

The static application is written to `dist/`.
