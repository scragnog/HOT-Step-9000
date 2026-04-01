# ACE-Step UI: Architecture & Data Flow Documentation

This document provides a technical overview of the ACE-Step UI architecture, specifically focusing on the flow of data from the user interface, through the intermediate Node.js API, and down to the underlying Python execution environment. It is designed to help agents and developers understand where to find, edit, or add code when modifying the application.

## 1. High-Level Architecture

The system is composed of three primary layers:

1. **Frontend (React/Vite):** A single-page application built with React, TypeScript, and TailwindCSS. It manages user state, provides the UI controls, and communicates with the Node.js middleware.
2. **Middleware (Node.js/Express):** An intermediate server that acts as a proxy, database manager (SQLite), and file storage handler. It translates UI requests into payloads suitable for the Python backend.
3. **Backend (Python/FastAPI):** The core execution engine (`acestep`). It manages the GPU, loads the models (DiT, LLM, Vocoders), handles the job queue, and executes the actual audio generation and manipulation tasks.

## 2. The Generation Data Flow (UI → Node API → Python API)

The biggest pain point for new developers is understanding how a user's click on "Generate" turns into audio. Here is the step-by-step trace of that pipeline.

### Step 1: The UI Layer (`ace-step-ui/components/CreatePanel.tsx`)
The `CreatePanel` is the monolithic control center of the frontend. It holds the state for dozens of generation parameters (prompts, BPM, solvers, guidance modes, LoRA settings).
- When the user clicks "Generate", the UI aggregates all these state variables into a `GenerationParams` object.
- It then calls `generateApi.startGeneration(params, token)` located in `ace-step-ui/services/api.ts`.

### Step 2: The API Client (`ace-step-ui/services/api.ts`)
This file defines the TypeScript interfaces for the API contract.
- The `startGeneration` function sends a `POST` request to the Node.js middleware at `/api/generate`.
- It also handles polling the middleware for job status via `generateApi.getStatus`.

### Step 3: The Node.js Middleware (`ace-step-ui/server/src/routes/generate.ts`)
This is the crucial translation layer. When the `POST /api/generate` request arrives:
1. **Database Tracking:** The Node server immediately creates a local job record in the SQLite database (`generation_jobs` table) with a status of `queued`.
2. **Payload Translation:** It maps the frontend's `GenerateBody` interface into the specific JSON structure expected by the Python backend's `/release_task` endpoint. 
3. **File Path Resolution:** Crucially, it resolves frontend URLs (e.g., `/audio/my-reference.mp3`) into absolute local filesystem paths that the Python backend can read directly.
4. **Proxying:** It sends the translated payload to the Python API running on port 8001 (usually via `http://localhost:8001/release_task`).
5. **ID Mapping:** It receives an `acestep_task_id` from Python, updates the local SQLite database, and returns a local `jobId` to the frontend.

*Note on Polling:* The frontend polls the Node server's `/api/generate/status/:jobId` endpoint. The Node server, in turn, polls the Python backend's `/query_result` endpoint, parses the progress text (e.g., `14%|###| 27/200`), formats the results, and passes them back to the UI.

### Step 4: The Python API Router (`acestep/api/route_setup.py` & `release_task_route.py`)
The Python backend receives the `/release_task` request.
- `route_setup.py` acts as the hub, registering all FastAPI routes.
- The `release_task_route.py` module parses the incoming JSON, validates it against Pydantic models, and enqueues the job into an in-memory thread-safe queue (`app.state.job_queue`).

### Step 5: The Execution Runtime (`acestep/api_server.py` & `job_execution_runtime.py`)
This is where the actual work happens.
- `api_server.py` defines the application lifespan and spins up background worker tasks.
- The worker pulls the job from the queue and calls `run_one_job_runtime()` (in `job_execution_runtime.py`).
- This runtime ensures the correct models are loaded in VRAM (handling hot-swapping if necessary).
- It executes the blocking generation function, which interfaces with the core `AceStepHandler` and `LLMHandler` to produce the audio tensors.
- Finally, it saves the output to disk and updates the in-memory job store with the success status and file paths.

## 3. Where to Edit or Add Code

Depending on the feature you are adding, you will need to touch different parts of the stack.

### Adding a New Generation Parameter (e.g., a new slider)
If you are adding a new setting that affects the audio model:
1. **Frontend UI:** Add the state and UI control in `ace-step-ui/components/CreatePanel.tsx` (or one of its sub-components).
2. **Frontend API:** Add the parameter to the `GenerationParams` interface in `ace-step-ui/services/api.ts`.
3. **Node Middleware:** Update the `GenerateBody` interface and the payload mapping logic in `ace-step-ui/server/src/routes/generate.ts` to forward the new parameter to Python.
4. **Python API:** Add the parameter to the Pydantic request model in `acestep/api/http/release_task_models.py` and map it in `acestep/api/http/release_task_param_parser.py`.
5. **Python Core:** Thread the parameter through `acestep/inference.py` (`GenerationParams` dataclass) and into the core generation logic (`acestep/core/generation/handler/generate_music.py`).

### Managing Models and Environment
If you are working on model loading, hot-switching, or `.env` configuration:
- Look at `ace-step-ui/server/src/routes/models.ts` for the Node-side orchestration and `.env` writing.
- Look at `acestep/api/http/model_switch_routes.py` and `acestep/api_server.py` for the Python-side VRAM management and loading logic.

### Handling User Uploads (Reference Tracks)
If you are modifying how audio files are uploaded or processed before generation:
- Look at `ace-step-ui/server/src/routes/referenceTrack.ts`. This handles the multer upload, MIME validation, storage, and SQLite database tracking for user-provided audio assets.

## 4. Lyric Studio (Lireek) Architecture

The Lyric Studio operates as an independent subsystem with its own database and API flow, distinct from the main audio generation pipeline.

### Direct-to-Python Proxying
Unlike the main audio generation which routes through the Node.js middleware for database tracking and payload translation, the Lyric Studio frontend (`ace-step-ui/services/lyricStudioApi.ts`) communicates with the Python backend directly. 
- In `vite.config.ts`, requests to `/api/lireek` are proxied directly to the Python API (typically port 8001).
- This means the Node.js middleware is entirely bypassed for Lyric Studio operations.

### Python Backend & Database
The Python backend manages the Lyric Studio logic and its separate persistence layer:
- **`acestep/api/http/lireek_routes.py`**: Defines the API endpoints for the full pipeline (fetching from Genius, profiling, generating, and refining).
- **`acestep/api/lireek/lireek_db.py`**: Manages a separate SQLite database schema for Lyric Studio entities (`artists`, `lyrics_sets`, `profiles`, `generations`, `album_presets`).
- **`acestep/api/lireek/generation_service.py`**: Contains the core logic for generating and refining lyrics, including prompt design, duration estimation, and strict formatting rules (e.g., enforcing verse/chorus line counts and section headers).

### Frontend Streaming State
Because lyric generation and profile building can be long-running tasks, the frontend uses a module-level singleton state manager (`ace-step-ui/stores/streamingStore.ts`). This allows the streaming text output and the bulk operation queue to survive UI navigation, ensuring users don't lose progress if they switch panels or navigate away from the Lyric Studio view.

## 5. Summary

To successfully modify the ACE-Step UI, you must respect the dual-API architecture. The frontend never talks directly to Python; it always speaks to the Node.js middleware. The middleware is responsible for database persistence, file path resolution, and translating UI-friendly concepts into the strict schemas required by the Python execution engine.
