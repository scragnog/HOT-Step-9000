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

// Health check (no auth required for connectivity check)
router.get('/health', async (_req, res: Response) => {
  try {
    const result = await proxyToAceStep('/health', 'GET');
    res.json(result);
  } catch (error: any) {
    const isConnRefused = error?.cause?.code === 'ECONNREFUSED' ||
      error?.code === 'ECONNREFUSED' ||
      error?.message?.includes('ECONNREFUSED');
    if (isConnRefused) {
      res.status(503).json({
        status: 'unavailable',
        models_initialized: false,
        llm_initialized: false,
        loaded_model: null,
        loaded_lm_model: null,
      });
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// List models
router.get('/models', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/model_inventory', 'GET');
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Initialize / switch model
router.post('/init', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/init', 'POST', req.body);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Hot-switch LM model
router.post('/lm/switch', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/models/lm/switch', 'POST', req.body);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Reinitialize after training
router.post('/reinitialize', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/reinitialize', 'POST');
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Server stats
router.get('/stats', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/stats', 'GET');
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

export default router;
