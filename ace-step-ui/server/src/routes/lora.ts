import { Router, Response } from 'express';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth.js';
import { config } from '../config/index.js';
import { readdirSync, statSync, existsSync, readFileSync, openSync, readSync, closeSync } from 'fs';
import { join, dirname, resolve } from 'path';
import { homedir } from 'os';

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
    const result = await proxyToAceStep('/v1/lora/unload', 'POST', req.body);
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

// ── In-browser file/folder browser (Node-native, no PowerShell) ───────

/**
 * Detect adapter type from file metadata.
 * - Reads safetensors binary header for "lokr_config" key → LoKR
 * - Checks for adapter_config.json in parent/same dir → PEFT LoRA
 * - Falls back to 'unknown'
 */
function detectAdapterType(filePath: string): { type: 'lokr' | 'lora' | 'unknown'; name: string } {
  const fileName = filePath.replace(/\\/g, '/').split('/').pop() || '';
  const dir = dirname(filePath);
  const parentDir = dirname(dir);

  // Check safetensors header for lokr_config metadata
  if (fileName.toLowerCase().endsWith('.safetensors') && existsSync(filePath)) {
    try {
      // Safetensors format: first 8 bytes = little-endian uint64 header size
      // Then header_size bytes of JSON metadata
      const fd = openSync(filePath, 'r');
      const sizeBuf = Buffer.alloc(8);
      readSync(fd, sizeBuf, 0, 8, 0);
      const headerSize = Number(sizeBuf.readBigUInt64LE(0));
      // Cap at 1MB to avoid reading huge headers
      const safeSize = Math.min(headerSize, 1024 * 1024);
      const headerBuf = Buffer.alloc(safeSize);
      readSync(fd, headerBuf, 0, safeSize, 8);
      closeSync(fd);
      const headerStr = headerBuf.toString('utf8');
      // Check for lokr_config in the metadata (top-level __metadata__ key)
      if (headerStr.includes('"lokr_config"')) {
        return { type: 'lokr', name: fileName };
      }
    } catch {
      // Failed to read header — fall through
    }
  }

  // Check for adapter_config.json (PEFT LoRA)
  // Could be in same dir as file, or if file is adapter_model.safetensors, in its dir
  const configPaths = [
    join(dir, 'adapter_config.json'),
    join(parentDir, 'adapter_config.json'),
  ];
  for (const cp of configPaths) {
    if (existsSync(cp)) {
      return { type: 'lora', name: fileName };
    }
  }

  // Check filename convention
  if (fileName.toLowerCase() === 'lokr_weights.safetensors') {
    return { type: 'lokr', name: fileName };
  }
  if (fileName.toLowerCase() === 'adapter_model.safetensors') {
    return { type: 'lora', name: fileName };
  }

  return { type: 'unknown', name: fileName };
}

/** Detect adapter type for a given path */
router.get('/detect-type', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const filePath = (req.query.path as string || '').trim();
    if (!filePath) {
      return res.status(400).json({ error: 'path parameter required' });
    }
    const result = detectAdapterType(filePath);
    res.json(result);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});



/**
 * List directories and .safetensors files at a given path.
 * Used by the in-browser file browser modal.
 */
router.get('/browse-dir', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    let dir = (req.query.path as string || '').trim();

    // Default to user's home directory
    if (!dir) {
      dir = homedir();
    }

    dir = resolve(dir);

    // On Windows, list drive letters if path is empty or root-like
    const entries: Array<{ name: string; path: string; type: 'dir' | 'file'; size?: number }> = [];

    // Add parent directory entry (unless we're at a root)
    const parent = dirname(dir);
    if (parent !== dir) {
      entries.push({ name: '..', path: parent, type: 'dir' });
    }

    try {
      const items = readdirSync(dir, { withFileTypes: true });
      for (const item of items) {
        // Skip hidden files/folders
        if (item.name.startsWith('.')) continue;

        const fullPath = join(dir, item.name);

        if (item.isDirectory()) {
          entries.push({ name: item.name, path: fullPath, type: 'dir' });
        } else if (item.name.toLowerCase().endsWith('.safetensors')) {
          try {
            const stat = statSync(fullPath);
            entries.push({ name: item.name, path: fullPath, type: 'file', size: stat.size });
          } catch {
            entries.push({ name: item.name, path: fullPath, type: 'file' });
          }
        }
      }
    } catch (fsErr: any) {
      return res.status(400).json({ error: `Cannot read directory: ${fsErr.message}`, current: dir });
    }

    // Sort: directories first (alphabetical), then files (alphabetical)
    entries.sort((a, b) => {
      if (a.name === '..') return -1;
      if (b.name === '..') return 1;
      if (a.type !== b.type) return a.type === 'dir' ? -1 : 1;
      return a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
    });

    res.json({ current: dir, entries });
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

router.post('/slot-trigger-word', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await proxyToAceStep('/v1/lora/slot-trigger-word', 'POST', req.body);
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

