import { Router, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth.js';

const router = Router();

// Resolve project root (ace-step-ui/server/src/routes -> ../../../../)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..', '..', '..', '..');

const ACESTEP_API_URL = process.env.ACESTEP_API_URL || 'http://127.0.0.1:8001';
const ACESTEP_API_KEY = process.env.ACESTEP_API_KEY || '';

async function proxyToAceStep(endpoint: string, method: string, data?: any) {
    const headers: Record<string, string> = {
        'Content-Type': 'application/json',
    };

    if (ACESTEP_API_KEY) {
        headers['x-api-key'] = ACESTEP_API_KEY;
        headers['Authorization'] = `Bearer ${ACESTEP_API_KEY}`;
    }

    const options: RequestInit = {
        method,
        headers,
    };

    if (data && (method === 'POST' || method === 'PUT')) {
        options.body = JSON.stringify(data);
    }

    const response = await fetch(`${ACESTEP_API_URL}${endpoint}`, options);

    if (!response.ok) {
        const errorData: any = await response.json().catch(() => ({ error: 'Request failed' }));
        const detail = errorData?.detail;
        const detailMsg = typeof detail === 'string'
            ? detail
            : Array.isArray(detail)
                ? detail.map((d: any) => d?.msg || JSON.stringify(d)).join('; ')
                : undefined;
        throw new Error(errorData?.error || errorData?.message || detailMsg || 'Request failed');
    }

    const result = await response.json();

    if (result && typeof result === 'object') {
        if ('code' in result && result.code && result.code !== 200) {
            throw new Error(result.error || result.message || 'Request failed');
        }
        if ('data' in result) {
            return result.data;
        }
    }
    return result;
}

// GET /api/models - List available models with active/preloaded status
router.get('/', async (_req, res: Response) => {
    try {
        const result = await proxyToAceStep('/v1/models/list', 'GET');
        res.json(result);
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

// GET /api/models/vocoders - List available vocoders
router.get('/vocoders', async (_req, res: Response) => {
    try {
        const result = await proxyToAceStep('/api/vocoders', 'GET');
        res.json(result);
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

// GET /api/models/status - Lightweight status check (active model only)
router.get('/status', async (_req, res: Response) => {
    try {
        const result = await proxyToAceStep('/v1/models/status', 'GET');
        res.json(result);
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

// POST /api/models/switch - Switch the active DiT model
router.post('/switch', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
    try {
        const result = await proxyToAceStep('/v1/models/switch', 'POST', req.body);
        res.json(result);
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

// POST /api/models/lm/backend - Hot-switch LM backend (pt ↔ vllm)
router.post('/lm/backend', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
    try {
        const result = await proxyToAceStep('/v1/models/lm/backend', 'POST', req.body);
        // Also update .env so the choice persists across restarts
        const envPath = path.join(PROJECT_ROOT, '.env');
        if (req.body.backend && fs.existsSync(envPath)) {
            let envContent = fs.readFileSync(envPath, 'utf-8');
            if (/^ACESTEP_LM_BACKEND=.*/m.test(envContent)) {
                envContent = envContent.replace(
                    /^ACESTEP_LM_BACKEND=.*/m,
                    `ACESTEP_LM_BACKEND=${req.body.backend}`
                );
            } else {
                envContent = envContent.trimEnd() + `\nACESTEP_LM_BACKEND=${req.body.backend}\n`;
            }
            fs.writeFileSync(envPath, envContent, 'utf-8');
        }
        res.json(result);
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

// POST /api/models/update-env - Update .env model selections (used by loading screen)
// No auth required — only called during startup before app is fully loaded
router.post('/update-env', async (req: any, res: Response) => {
    try {
        const { ACESTEP_CONFIG_PATH, ACESTEP_LM_MODEL_PATH, ACESTEP_LM_BACKEND, ACESTEP_REDMOND_MODE, ACESTEP_REDMOND_SCALE, ACESTEP_NO_INIT, ACESTEP_QUANTIZATION, ACESTEP_LM_GPU_LAYERS, ACESTEP_GGUF_QUANT, ACESTEP_ADAPTER_MERGE_MODE } = req.body;
        if (!ACESTEP_CONFIG_PATH && !ACESTEP_LM_MODEL_PATH && !ACESTEP_LM_BACKEND && ACESTEP_REDMOND_MODE === undefined && !ACESTEP_REDMOND_SCALE && ACESTEP_NO_INIT === undefined && ACESTEP_QUANTIZATION === undefined && ACESTEP_LM_GPU_LAYERS === undefined && ACESTEP_GGUF_QUANT === undefined && ACESTEP_ADAPTER_MERGE_MODE === undefined) {
            res.status(400).json({ error: 'At least one setting required' });
            return;
        }

        const envPath = path.join(PROJECT_ROOT, '.env');
        if (!fs.existsSync(envPath)) {
            // Auto-create .env from .env.example (fresh installs won't have it yet)
            const examplePath = path.join(PROJECT_ROOT, '.env.example');
            if (fs.existsSync(examplePath)) {
                fs.copyFileSync(examplePath, envPath);
                console.log('[Models] Created .env from .env.example (first run)');
            } else {
                // Minimal fallback if .env.example is also missing
                const minimal = [
                    'ACESTEP_CONFIG_PATH=acestep-v15-turbo',
                    'ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B',
                    'ACESTEP_LM_BACKEND=vllm',
                    'ACESTEP_DEVICE=auto',
                    'ACESTEP_INIT_LLM=auto',
                    'ACESTEP_QUANTIZATION=auto',
                    'VITE_PORT=3000',
                    '',
                ].join('\n');
                fs.writeFileSync(envPath, minimal, 'utf-8');
                console.log('[Models] Created minimal .env (no .env.example found)');
            }
        }

        let envContent = fs.readFileSync(envPath, 'utf-8');

        if (ACESTEP_CONFIG_PATH) {
            envContent = envContent.replace(
                /^ACESTEP_CONFIG_PATH=.*/m,
                `ACESTEP_CONFIG_PATH=${ACESTEP_CONFIG_PATH}`
            );
        }
        if (ACESTEP_LM_MODEL_PATH) {
            envContent = envContent.replace(
                /^ACESTEP_LM_MODEL_PATH=.*/m,
                `ACESTEP_LM_MODEL_PATH=${ACESTEP_LM_MODEL_PATH}`
            );
        }
        if (ACESTEP_LM_BACKEND) {
            if (/^ACESTEP_LM_BACKEND=.*/m.test(envContent)) {
                envContent = envContent.replace(
                    /^ACESTEP_LM_BACKEND=.*/m,
                    `ACESTEP_LM_BACKEND=${ACESTEP_LM_BACKEND}`
                );
            } else {
                // Append if not present
                envContent = envContent.trimEnd() + `\nACESTEP_LM_BACKEND=${ACESTEP_LM_BACKEND}\n`;
            }
        }

        // Redmond Mode
        if (ACESTEP_REDMOND_MODE !== undefined) {
            if (/^ACESTEP_REDMOND_MODE=.*/m.test(envContent)) {
                envContent = envContent.replace(
                    /^ACESTEP_REDMOND_MODE=.*/m,
                    `ACESTEP_REDMOND_MODE=${ACESTEP_REDMOND_MODE}`
                );
            } else {
                envContent = envContent.trimEnd() + `\nACESTEP_REDMOND_MODE=${ACESTEP_REDMOND_MODE}\n`;
            }
        }
        if (ACESTEP_REDMOND_SCALE) {
            if (/^ACESTEP_REDMOND_SCALE=.*/m.test(envContent)) {
                envContent = envContent.replace(
                    /^ACESTEP_REDMOND_SCALE=.*/m,
                    `ACESTEP_REDMOND_SCALE=${ACESTEP_REDMOND_SCALE}`
                );
            } else {
                envContent = envContent.trimEnd() + `\nACESTEP_REDMOND_SCALE=${ACESTEP_REDMOND_SCALE}\n`;
            }
        }
        
        // Lazy Load Models toggle
        if (ACESTEP_NO_INIT !== undefined) {
            if (/^ACESTEP_NO_INIT=.*/m.test(envContent)) {
                envContent = envContent.replace(
                    /^ACESTEP_NO_INIT=.*/m,
                    `ACESTEP_NO_INIT=${ACESTEP_NO_INIT}`
                );
            } else {
                envContent = envContent.trimEnd() + `\nACESTEP_NO_INIT=${ACESTEP_NO_INIT}\n`;
            }
        }

        // Quantization setting
        if (ACESTEP_QUANTIZATION !== undefined) {
            if (/^ACESTEP_QUANTIZATION=.*/m.test(envContent)) {
                envContent = envContent.replace(
                    /^ACESTEP_QUANTIZATION=.*/m,
                    `ACESTEP_QUANTIZATION=${ACESTEP_QUANTIZATION}`
                );
            } else {
                envContent = envContent.trimEnd() + `\nACESTEP_QUANTIZATION=${ACESTEP_QUANTIZATION}\n`;
            }
        }

        // LM GPU Layers setting (llama-cpp backend)
        if (ACESTEP_LM_GPU_LAYERS !== undefined) {
            if (/^ACESTEP_LM_GPU_LAYERS=.*/m.test(envContent)) {
                envContent = envContent.replace(
                    /^ACESTEP_LM_GPU_LAYERS=.*/m,
                    `ACESTEP_LM_GPU_LAYERS=${ACESTEP_LM_GPU_LAYERS}`
                );
            } else {
                envContent = envContent.trimEnd() + `\nACESTEP_LM_GPU_LAYERS=${ACESTEP_LM_GPU_LAYERS}\n`;
            }
        }

        // GGUF quantization preference (llama-cpp backend)
        if (ACESTEP_GGUF_QUANT !== undefined) {
            if (/^ACESTEP_GGUF_QUANT=.*/m.test(envContent)) {
                envContent = envContent.replace(
                    /^ACESTEP_GGUF_QUANT=.*/m,
                    `ACESTEP_GGUF_QUANT=${ACESTEP_GGUF_QUANT}`
                );
            } else {
                envContent = envContent.trimEnd() + `\nACESTEP_GGUF_QUANT=${ACESTEP_GGUF_QUANT}\n`;
            }
        }

        // VRAM-optimized adapter merge mode
        if (ACESTEP_ADAPTER_MERGE_MODE !== undefined) {
            if (/^ACESTEP_ADAPTER_MERGE_MODE=.*/m.test(envContent)) {
                envContent = envContent.replace(
                    /^ACESTEP_ADAPTER_MERGE_MODE=.*/m,
                    `ACESTEP_ADAPTER_MERGE_MODE=${ACESTEP_ADAPTER_MERGE_MODE}`
                );
            } else {
                envContent = envContent.trimEnd() + `\nACESTEP_ADAPTER_MERGE_MODE=${ACESTEP_ADAPTER_MERGE_MODE}\n`;
            }
        }

        fs.writeFileSync(envPath, envContent, 'utf-8');
        console.log(`[Models] .env updated: CONFIG_PATH=${ACESTEP_CONFIG_PATH || '(unchanged)'}, LM_MODEL_PATH=${ACESTEP_LM_MODEL_PATH || '(unchanged)'}, LM_BACKEND=${ACESTEP_LM_BACKEND || '(unchanged)'}, REDMOND_MODE=${ACESTEP_REDMOND_MODE ?? '(unchanged)'}, NO_INIT=${ACESTEP_NO_INIT ?? '(unchanged)'}, QUANTIZATION=${ACESTEP_QUANTIZATION ?? '(unchanged)'}, LM_GPU_LAYERS=${ACESTEP_LM_GPU_LAYERS ?? '(unchanged)'}, GGUF_QUANT=${ACESTEP_GGUF_QUANT ?? '(unchanged)'}, ADAPTER_MERGE_MODE=${ACESTEP_ADAPTER_MERGE_MODE ?? '(unchanged)'}`);
        res.json({ success: true });
    } catch (error: any) {
        console.error('[Models] Failed to update .env:', error);
        res.status(500).json({ error: error.message });
    }
});

// GET /api/models/env-config - Return current .env config for frontend initialization
// No auth — called at app startup before auth context is available
router.get('/env-config', async (_req: any, res: Response) => {
    try {
        const envPath = path.join(PROJECT_ROOT, '.env');
        let lmBackend = 'vllm';
        let redmondMode = 'false';
        let redmondScale = '0.7';
        if (fs.existsSync(envPath)) {
            const content = fs.readFileSync(envPath, 'utf-8');
            const matchLm = content.match(/^ACESTEP_LM_BACKEND=(.*)$/m);
            if (matchLm) lmBackend = matchLm[1].trim().toLowerCase();
            const matchRedmond = content.match(/^ACESTEP_REDMOND_MODE=(.*)$/m);
            if (matchRedmond) redmondMode = matchRedmond[1].trim().toLowerCase();
            const matchScale = content.match(/^ACESTEP_REDMOND_SCALE=(.*)$/m);
            if (matchScale) redmondScale = matchScale[1].trim();
        }
        // Check if Redmond adapter is available on disk
        const redmondAdapterPath = path.join(PROJECT_ROOT, 'checkpoints', 'redmond-refine', 'standard', 'adapter_config.json');
        const redmondAvailable = fs.existsSync(redmondAdapterPath);
        // Read quantization setting
        let quantization = 'auto';
        if (fs.existsSync(envPath)) {
            const qContent = fs.readFileSync(envPath, 'utf-8');
            const matchQuant = qContent.match(/^ACESTEP_QUANTIZATION=(.*)$/m);
            if (matchQuant) quantization = matchQuant[1].trim().toLowerCase();
        }
        // Read adapter merge mode setting
        let adapterMergeMode = 'false';
        if (fs.existsSync(envPath)) {
            const amContent = fs.readFileSync(envPath, 'utf-8');
            const matchAm = amContent.match(/^ACESTEP_ADAPTER_MERGE_MODE=(.*)$/m);
            if (matchAm) adapterMergeMode = matchAm[1].trim().toLowerCase();
        }
        res.json({ lmBackend, redmondMode, redmondScale, redmondAvailable, quantization, adapterMergeMode });
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

// GET /api/models/gguf-status/:model - Return GGUF conversion status for a model
// No auth — called during loading screen
router.get('/gguf-status/:model', async (req: any, res: Response) => {
    try {
        const modelName = req.params.model;
        const modelDir = path.join(PROJECT_ROOT, 'checkpoints', modelName);
        const allQuants = ['Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0', 'BF16'];

        // Check safetensors
        let safetensorsPresent = false;
        if (fs.existsSync(modelDir)) {
            const files = fs.readdirSync(modelDir);
            safetensorsPresent = files.some(f => f.endsWith('.safetensors'));
        }

        // Check which GGUFs exist
        const availableGguf: Record<string, boolean> = {};
        for (const quant of allQuants) {
            const ggufPath = path.join(modelDir, `${modelName}-${quant}.gguf`);
            availableGguf[quant] = fs.existsSync(ggufPath);
        }

        res.json({ safetensorsPresent, availableGguf, modelDir });
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

// GET /api/models/convert-gguf - Convert model to GGUF with streaming progress (SSE)
// Uses GET so EventSource can be used from the loading screen (file:// origin)
// No auth — called during loading screen
router.get('/convert-gguf', async (req: any, res: Response) => {
    const model = req.query.model as string;
    const quant = req.query.quant as string;
    const convertAll = req.query.convertAll === 'true';
    console.log(`[GGUF Convert] Request: model=${model}, quant=${quant}, convertAll=${convertAll}`);
    if (!model || !quant) {
        res.status(400).json({ error: 'model and quant query params required' });
        return;
    }

    // Set SSE headers + CORS for EventSource from file:// origins
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.flushHeaders();

    const sendEvent = (data: string, event?: string) => {
        if (event) res.write(`event: ${event}\n`);
        res.write(`data: ${data}\n\n`);
    };

    // Find Python executable
    const venvPython = path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe');
    const pythonExe = fs.existsSync(venvPython) ? venvPython : 'python';

    const converterModule = path.join(PROJECT_ROOT, 'acestep', 'tools', 'gguf_converter.py');
    if (!fs.existsSync(converterModule)) {
        sendEvent('❌ gguf_converter.py not found', 'error');
        res.end();
        return;
    }

    // Build command args
    const args = [converterModule];
    if (convertAll) {
        args.push('--all');
    } else {
        args.push(model);
    }
    args.push('--quant', quant);
    args.push('--checkpoints', path.join(PROJECT_ROOT, 'checkpoints'));

    sendEvent(`Starting conversion: ${convertAll ? 'all models' : model} → ${quant}`);

    console.log('[GGUF Convert] Spawning:', pythonExe, args.join(' '));
    const proc = spawn(pythonExe, args, {
        cwd: PROJECT_ROOT,
        env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8', ACESTEP_PROJECT_ROOT: PROJECT_ROOT },
    });
    console.log('[GGUF Convert] Process spawned, PID:', proc.pid);

    proc.stdout.on('data', (chunk: Buffer) => {
        const lines = chunk.toString('utf-8').split('\n');
        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed) sendEvent(trimmed);
        }
    });

    proc.stderr.on('data', (chunk: Buffer) => {
        const lines = chunk.toString('utf-8').split('\n');
        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed) sendEvent(trimmed);
        }
    });

    proc.on('close', (code: number | null) => {
        console.log(`[GGUF Convert] Process closed with code: ${code}`);
        if (code === 0) {
            sendEvent('✅ All conversions completed successfully', 'done');
        } else {
            sendEvent(`❌ Conversion failed (exit code ${code})`, 'error');
        }
        res.end();
    });

    proc.on('error', (err: Error) => {
        console.log(`[GGUF Convert] Process error: ${err.message}`);
        sendEvent(`❌ Failed to start conversion: ${err.message}`, 'error');
        res.end();
    });

    // Handle client disconnect
    req.on('close', () => {
        console.log(`[GGUF Convert] Client disconnected, proc.killed=${proc.killed}`);
        if (!proc.killed) {
            proc.kill('SIGTERM');
        }
    });
});

export default router;
