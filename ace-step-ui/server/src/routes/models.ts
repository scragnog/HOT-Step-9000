import { Router, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
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
        const { ACESTEP_CONFIG_PATH, ACESTEP_LM_MODEL_PATH, ACESTEP_LM_BACKEND, ACESTEP_REDMOND_MODE, ACESTEP_REDMOND_SCALE, ACESTEP_NO_INIT } = req.body;
        if (!ACESTEP_CONFIG_PATH && !ACESTEP_LM_MODEL_PATH && !ACESTEP_LM_BACKEND && ACESTEP_REDMOND_MODE === undefined && !ACESTEP_REDMOND_SCALE && ACESTEP_NO_INIT === undefined) {
            res.status(400).json({ error: 'At least one setting required' });
            return;
        }

        const envPath = path.join(PROJECT_ROOT, '.env');
        if (!fs.existsSync(envPath)) {
            res.status(404).json({ error: '.env file not found' });
            return;
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

        fs.writeFileSync(envPath, envContent, 'utf-8');
        console.log(`[Models] .env updated: CONFIG_PATH=${ACESTEP_CONFIG_PATH || '(unchanged)'}, LM_MODEL_PATH=${ACESTEP_LM_MODEL_PATH || '(unchanged)'}, LM_BACKEND=${ACESTEP_LM_BACKEND || '(unchanged)'}, REDMOND_MODE=${ACESTEP_REDMOND_MODE ?? '(unchanged)'}, NO_INIT=${ACESTEP_NO_INIT ?? '(unchanged)'}`);
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
        res.json({ lmBackend, redmondMode, redmondScale, redmondAvailable });
    } catch (error: any) {
        res.status(500).json({ error: error.message });
    }
});

export default router;
