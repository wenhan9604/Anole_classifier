# Florida Anole Classifier - Frontend

A React web application for identifying and classifying Florida anole species from uploaded images. The app provides a user-friendly interface for species detection with confidence scoring and citizen science integration.

## What This Does

- **Species Classification**: Identifies 5 Florida anole species (Green Anole, Brown Anole, Crested Anole, Knight Anole, Bark Anole)
- **Multi-Detection**: Can detect multiple lizards in a single image
- **Confidence Scoring**: Shows confidence levels for each species prediction with visual indicators
- **Mobile Support**: Responsive design optimized for mobile devices
- **Citizen Science**: iNaturalist OAuth runs on the API (`VITE_API_BASE_URL`); the browser stores only an HTTP-only session cookie. Upload posts the image to `/api/observations` with credentials.

## Current Status

Set **`VITE_API_BASE_URL`** to your FastAPI origin (same scheme/host you use for `/api/predict`). The prediction page uses that API for inference and for **Connect iNaturalist** / **Upload observation**. After OAuth, configure the API to redirect back to e.g. `https://your-site/predict?inat=connected` via `INAT_FRONTEND_SUCCESS_URL`.

## How to Launch

### Prerequisites
- Node.js 18 or higher
- npm or yarn

### Installation and Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The application will be available at `http://localhost:5173` (or the next available port).

### Inference location (`?gpu=`)

On the prediction page, you can choose where inference runs:

| Query | Behavior |
|--------|----------|
| *(default)* | Backend PyTorch CPU (`/api/predict`) |
| `?gpu=server` | Backend PyTorch CPU (same as default; kept for links) |
| `?gpu=client-side` or `?gpu=client` | Client ONNX: **WebGPU first** (`onnxruntime-web/webgpu`), falls back to WASM if WebGPU is unavailable or init fails |
| `?gpu=client-wasm` | Client ONNX **WASM only** |
| `?gpu=client-gpu` | Client ONNX **WebGPU required** (fails if `navigator.gpu` is missing) |

On the prediction page you can also use the **Inference location** dropdown (it updates the URL `gpu` query for sharing).

Client runs need `yolo_best.onnx` and `swin_model.onnx` served from `/models/` (see `public/` / deploy config). After a run, expand **Inference timings (client ONNX)** for stage-level milliseconds (YOLO preprocess / inference / postprocess, per-crop Swin).

**Manual check (no automation in CI here):** With the same image, run Predict under default (backend), then `?gpu=client-wasm`, then `?gpu=client-side` in a WebGPU-capable browser; compare total time in the timings JSON and the badge `EP=` label (`wasm` vs `webgpu+wasm`).

### Building for Production
```bash
# Build the application
npm run build

# Preview the production build
npm run preview
```

## Project Structure

- `src/pages/LandingPage.tsx` - Welcome screen with features overview
- `src/pages/PredictionPage.tsx` - Main classification interface
- `src/services/iNaturalistService.ts` - iNaturalist API integration service
- `src/App.css` - Styling and mobile responsiveness

## Next Steps

- Tune iNaturalist observation fields (date, taxon_id vs species_guess) as needed
- Optional: persist OAuth tokens in Redis/DB instead of in-memory store for multi-instance APIs
