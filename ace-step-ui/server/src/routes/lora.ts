import { Router, Response } from 'express';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth.js';
import { config } from '../config/index.js';

const router = Router();

const ACESTEP_API_URL = config.acestep.apiUrl;
const ACESTEP_API_KEY = process.env.ACESTEP_API_KEY || '';

async function proxyToAceStep(endpoint: string, method: string, data?: any) {
  try {
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
  } catch (error: any) {
    throw new Error(error.message || 'Request failed');
  }
}

router.post('/load', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/load', 'POST', req.body);
    res.json(result || { message: 'LoRA loaded' });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/unload', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/unload', 'POST');
    res.json(result || { message: 'LoRA unloaded' });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/toggle', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/toggle', 'POST', req.body);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/scale', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/scale', 'POST', req.body);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

router.get('/status', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/status', 'GET');
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// ── File browser ───────────────────────────────────────────────────
router.get('/list-files', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const folder = req.query.folder as string || '';
    const result = await proxyToAceStep(`/v1/lora/list-files?folder=${encodeURIComponent(folder)}`, 'GET');
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// ── Advanced adapter endpoints ─────────────────────────────────────
router.post('/group-scales', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/group-scales', 'POST', req.body);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/slot-group-scales', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/slot-group-scales', 'POST', req.body);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/slot-layer-scales', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/slot-layer-scales', 'POST', req.body);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/slot-layer-scale', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/slot-layer-scale', 'POST', req.body);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/temporal-schedule', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/temporal-schedule', 'POST', req.body);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/audio-diff', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/audio/diff', 'POST', req.body);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// ── LM LoRA (PEFT adapter on the 5Hz LLM) ─────────────────────────
router.post('/lm-load', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/lm-load', 'POST', req.body);
    res.json(result || { message: 'LM LoRA loaded' });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/lm-unload', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/lm-unload', 'POST', req.body);
    res.json(result || { message: 'LM LoRA unloaded' });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

router.post('/lm-scale', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/lm-scale', 'POST', req.body);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

router.get('/lm-status', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/lm-status', 'GET');
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

export default router;

