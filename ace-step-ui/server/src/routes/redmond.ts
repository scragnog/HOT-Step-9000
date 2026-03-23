import { Router, Response } from 'express';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth.js';
import { config } from '../config/index.js';

const router = Router();

const ACESTEP_API_URL = config.acestep.apiUrl;
const ACESTEP_API_KEY = process.env.ACESTEP_API_KEY || '';

async function proxyToAceStep(endpoint: string, method: string, data?: any) {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  if (ACESTEP_API_KEY) {
    headers['x-api-key'] = ACESTEP_API_KEY;
    headers['Authorization'] = `Bearer ${ACESTEP_API_KEY}`;
  }

  const options: RequestInit = { method, headers };

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

// GET /api/redmond/status — current Redmond Mode state
router.get('/status', authMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/redmond/status', 'GET');
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// POST /api/redmond/toggle — enable/disable Redmond Mode
router.post('/toggle', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/redmond/toggle', 'POST', req.body);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// POST /api/redmond/scale — set Redmond Mode scale
router.post('/scale', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/redmond/scale', 'POST', req.body);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

export default router;
