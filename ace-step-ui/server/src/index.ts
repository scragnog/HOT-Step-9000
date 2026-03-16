import dotenv from 'dotenv';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import path from 'path';
import { fileURLToPath } from 'url';

// Load .env from project root (parent of server directory)
const __filename_init = fileURLToPath(import.meta.url);
const __dirname_init = path.dirname(__filename_init);
dotenv.config({ path: path.join(__dirname_init, '../../.env') });
import cron from 'node-cron';
import { config } from './config/index.js';
import { runCleanupJob, cleanupDeletedSongs } from './services/cleanup.js';

// Setup mirroring of console output to session log file if configured by LAUNCH.bat
if (process.env.ACESTEP_LOG_DIR) {
  const fs = await import('fs');
  const logFile = path.join(process.env.ACESTEP_LOG_DIR, 'node_console.log');

  const originalStdoutWrite = process.stdout.write.bind(process.stdout);
  const originalStderrWrite = process.stderr.write.bind(process.stderr);

  (process.stdout as any).write = (chunk: any, encoding?: any, callback?: any) => {
    try { fs.appendFileSync(logFile, chunk); } catch (e) { /* ignore */ }
    return originalStdoutWrite(chunk, encoding, callback);
  };

  (process.stderr as any).write = (chunk: any, encoding?: any, callback?: any) => {
    try { fs.appendFileSync(logFile, chunk); } catch (e) { /* ignore */ }
    return originalStderrWrite(chunk, encoding, callback);
  };

  console.log(`[Logger] Node.js console output is being mirrored transparently to ${logFile}`);
}


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import authRoutes from './routes/auth.js';
import songsRoutes from './routes/songs.js';
import generateRoutes from './routes/generate.js';
import usersRoutes from './routes/users.js';
import playlistsRoutes from './routes/playlists.js';
import contactRoutes from './routes/contact.js';
import referenceTrackRoutes from './routes/referenceTrack.js';
import loraRoutes from './routes/lora.js';
import steeringRoutes from './routes/steering.js';
import modelsRoutes from './routes/models.js';
import systemRoutes from './routes/system.js';
import analyzeRoutes from './routes/analyze.js';
import modelRoutes from './routes/model.js';
import { pool } from './db/pool.js';
import './db/migrate.js';

const app = express();

// Security headers
app.use(helmet({
  crossOriginResourcePolicy: { policy: 'cross-origin' },
  crossOriginEmbedderPolicy: false,
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      baseUri: ["'self'"],
      fontSrc: ["'self'", 'https:', 'data:'],
      formAction: ["'self'"],
      frameAncestors: ["'self'"],
      imgSrc: ["'self'", 'data:', 'https:'],
      objectSrc: ["'none'"],
      scriptSrc: ["'self'"],
      scriptSrcAttr: ["'none'"],
      styleSrc: ["'self'", 'https:', "'unsafe-inline'"],
      upgradeInsecureRequests: [],
    },
  },
}));

// Middleware
app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (mobile apps, curl, etc.)
    // Also allow 'null' origin from file:// pages (loading screen)
    if (!origin || origin === 'null') return callback(null, true);
    // Allow localhost and 127.0.0.1 on any port in development
    if (config.nodeEnv === 'development') {
      if (origin.includes('localhost') || origin.includes('127.0.0.1')) {
        return callback(null, true);
      }
      // Allow LAN IPs (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
      const lanPattern = /^https?:\/\/(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.)/;
      if (lanPattern.test(origin)) {
        return callback(null, true);
      }
    }
    // Allow configured frontend URL
    if (origin === config.frontendUrl) {
      return callback(null, true);
    }
    callback(new Error('Not allowed by CORS'));
  },
  credentials: true,
}));

app.use(express.json());

// Serve static audio files
app.use('/audio', express.static(path.join(__dirname, '../public/audio')));

// Serve LRC (synced lyrics) files — resolves from audio URL
app.get('/api/lrc', async (req, res) => {
  const audioUrl = req.query.audioUrl as string;
  if (!audioUrl) {
    res.status(400).json({ error: 'audioUrl required' });
    return;
  }

  // Resolve audio URL to local filesystem path
  let audioPath: string | null = null;

  if (audioUrl.startsWith('/audio/')) {
    // Static audio path: /audio/jobId/uuid.flac
    audioPath = path.join(__dirname, '../public', audioUrl);
  } else {
    // Python API URL: http://localhost:8001/v1/audio?path=C:\...\uuid.flac
    try {
      const parsed = new URL(audioUrl, 'http://localhost');
      const pathParam = parsed.searchParams.get('path');
      if (pathParam) audioPath = pathParam;
    } catch { /* ignore */ }
  }

  if (!audioPath) {
    res.status(404).json({ error: 'Cannot resolve audio path' });
    return;
  }

  // Swap extension to .lrc
  const lrcPath = audioPath.replace(/\.\w+$/, '.lrc');

  try {
    const fs = await import('fs/promises');
    const content = await fs.readFile(lrcPath, 'utf-8');
    res.type('text/plain').send(content);
  } catch {
    res.status(404).json({ error: 'LRC file not found' });
  }
});

// Serve training dataset audio files (securely)
app.get('/api/audio/file', async (req, res) => {
  const filePath = req.query.path as string;
  if (!filePath) {
    res.status(400).json({ error: 'Path required' });
    return;
  }

  console.log('[Audio File] Requested path:', filePath);

  // Security: only allow access to common audio directories
  const allowedDirs = ['datasets', 'dataset', 'audio', 'training', 'lora', 'samples'];

  // Normalize path - handle both Unix and Windows paths
  let normalizedPath = filePath.replace(/\\/g, '/');
  normalizedPath = path.normalize(normalizedPath);

  console.log('[Audio File] Normalized path:', normalizedPath);

  // Check if path contains any allowed directory
  const isAllowed = allowedDirs.some(dir =>
    normalizedPath.toLowerCase().includes('/' + dir.toLowerCase() + '/') ||
    normalizedPath.toLowerCase().includes('\\' + dir.toLowerCase() + '\\') ||
    normalizedPath.toLowerCase().startsWith(dir.toLowerCase() + '/') ||
    normalizedPath.toLowerCase().startsWith(dir.toLowerCase() + '\\') ||
    normalizedPath.toLowerCase().startsWith('./' + dir.toLowerCase())
  );

  if (!isAllowed) {
    console.log('[Audio File] Access denied - not in allowed directories:', normalizedPath);
    res.status(403).json({ error: 'Access denied', path: normalizedPath });
    return;
  }

  // Prevent directory traversal
  if (normalizedPath.includes('..')) {
    res.status(403).json({ error: 'Invalid path - directory traversal detected' });
    return;
  }

  // Resolve to absolute path
  let absolutePath: string;
  const fs = await import('fs');

  // Try multiple possible base paths
  const possiblePaths: string[] = [];

  // Handle URL encoding and special characters
  let decodedPath = normalizedPath;
  try {
    // Try to decode if it's URL encoded
    decodedPath = decodeURIComponent(normalizedPath);
  } catch (e) {
    // Not URL encoded or already decoded, use as is
  }

  // Also try double decoding in case of double encoding
  let doubleDecodedPath = decodedPath;
  try {
    doubleDecodedPath = decodeURIComponent(decodedPath);
  } catch (e) {
    // Not double encoded
  }

  console.log('[Audio File] Original path:', filePath);
  console.log('[Audio File] Normalized path:', normalizedPath);
  console.log('[Audio File] Decoded path:', decodedPath);
  console.log('[Audio File] Double decoded path:', doubleDecodedPath);

  if (path.isAbsolute(filePath)) {
    // Already absolute path - use as-is
    possiblePaths.push(filePath);
    possiblePaths.push(decodedPath);
    possiblePaths.push(doubleDecodedPath);
  } else {
    // Try different base directories
    // __dirname is server/src, so:
    // .. = server, ../.. = ace-step-ui, ../../.. = ace-step-ui parent (where datasets usually is)
    const projectRoot = path.join(__dirname, '../..');        // ace-step-ui
    const workspaceRoot = path.join(__dirname, '../../..');    // ace-step-ui parent (datasets location)
    const serverRoot = path.join(__dirname, '..');            // server
    const cwd = process.cwd();

    // Remove leading ./ if present
    const cleanPath = normalizedPath.startsWith('./') ? normalizedPath.slice(2) : normalizedPath;
    const cleanDecodedPath = decodedPath.startsWith('./') ? decodedPath.slice(2) : decodedPath;
    const cleanDoubleDecodedPath = doubleDecodedPath.startsWith('./') ? doubleDecodedPath.slice(2) : doubleDecodedPath;

    // Try various combinations
    const bases = [workspaceRoot, projectRoot, serverRoot, cwd];
    const paths = [normalizedPath, decodedPath, doubleDecodedPath, cleanPath, cleanDecodedPath, cleanDoubleDecodedPath];

    for (const base of bases) {
      for (const p of paths) {
        if (p && !possiblePaths.includes(path.join(base, p))) {
          possiblePaths.push(path.join(base, p));
        }
      }
    }

    // Also try without 'datasets' prefix if it's already in the path
    for (const cleanP of [cleanPath, cleanDecodedPath, cleanDoubleDecodedPath]) {
      if (cleanP.includes('datasets/') || cleanP.includes('datasets\\')) {
        const withoutDatasetsPrefix = cleanP.replace(/^datasets[/\\]/, '');
        for (const base of bases) {
          const p1 = path.join(base, withoutDatasetsPrefix);
          const p2 = path.join(base, 'datasets', withoutDatasetsPrefix);
          if (!possiblePaths.includes(p1)) possiblePaths.push(p1);
          if (!possiblePaths.includes(p2)) possiblePaths.push(p2);
        }
      }
    }
  }

  console.log('[Audio File] Trying paths:', possiblePaths);

  // Find first existing path
  absolutePath = possiblePaths.find(p => fs.existsSync(p)) || '';

  if (!absolutePath) {
    console.log('[Audio File] File not found. Tried:', possiblePaths);
    res.status(404).json({ error: 'File not found', tried: possiblePaths, original: filePath });
    return;
  }

  console.log('[Audio File] Found file at:', absolutePath);

  // Set proper content type for audio files
  const ext = path.extname(absolutePath).toLowerCase();
  const contentType: Record<string, string> = {
    '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav',
    '.flac': 'audio/flac',
    '.ogg': 'audio/ogg',
    '.m4a': 'audio/mp4',
    '.aac': 'audio/aac',
  };
  if (contentType[ext]) {
    res.setHeader('Content-Type', contentType[ext]);
  }

  // Set CORS headers for audio streaming
  res.setHeader('Accept-Ranges', 'bytes');

  // Send file
  res.sendFile(absolutePath, (err) => {
    if (err) {
      console.error('[Audio File] Failed to send:', err);
      if (!res.headersSent) {
        res.status(500).json({ error: 'Failed to send file' });
      }
    } else {
      console.log('[Audio File] Successfully served:', absolutePath);
    }
  });
});

// Debug endpoint to check file paths
app.get('/api/audio/debug', async (req, res) => {
  const fs = await import('fs');
  const path = await import('path');

  const projectRoot = path.join(__dirname, '../..');
  const workspaceRoot = path.join(__dirname, '../../..');
  const serverRoot = path.join(__dirname, '..');
  const cwd = process.cwd();

  // Check if datasets directory exists in various locations
  const checks = [
    { name: 'workspaceRoot (ace-step-ui parent)', path: workspaceRoot },
    { name: 'projectRoot (ace-step-ui)', path: projectRoot },
    { name: 'serverRoot', path: serverRoot },
    { name: 'cwd', path: cwd },
    { name: 'datasets in workspaceRoot', path: path.join(workspaceRoot, 'datasets') },
    { name: 'datasets in projectRoot', path: path.join(projectRoot, 'datasets') },
    { name: 'datasets in serverRoot', path: path.join(serverRoot, 'datasets') },
    { name: 'datasets in cwd', path: path.join(cwd, 'datasets') },
  ];

  const results: any = {
    workingDirectory: cwd,
    __dirname: __dirname,
    checks: {} as Record<string, { exists: boolean; isDirectory: boolean; contents?: string[] }>
  };

  for (const check of checks) {
    try {
      const exists = fs.existsSync(check.path);
      if (exists) {
        const stat = fs.statSync(check.path);
        const isDir = stat.isDirectory();
        const contents = isDir ? fs.readdirSync(check.path).slice(0, 20) : undefined;
        results.checks[check.name] = { exists, isDirectory: isDir, contents };
      } else {
        results.checks[check.name] = { exists: false, isDirectory: false };
      }
    } catch (e: any) {
      results.checks[check.name] = { exists: false, isDirectory: false, error: e.message };
    }
  }

  // If a test path is provided, check it directly
  const testPath = req.query.path as string;
  if (testPath) {
    results.testPath = {
      original: testPath,
      normalized: path.normalize(testPath),
      decoded: decodeURIComponent(testPath),
      exists: fs.existsSync(testPath),
      existsNormalized: fs.existsSync(path.normalize(testPath)),
      existsDecoded: fs.existsSync(decodeURIComponent(testPath)),
    };
  }

  res.json(results);
});

// Audio Editor (AudioMass) - needs relaxed CSP for inline scripts and external images
app.use('/editor', (req, res, next) => {
  res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; media-src 'self' blob: data: http://localhost:* https:; connect-src 'self' http://localhost:* https:; worker-src 'self' blob:");
  next();
}, express.static(path.join(__dirname, '../audio-editor')));

// Demucs Web (Stem Extraction) - requires COOP/COEP headers for SharedArrayBuffer and relaxed CSP for ONNX runtime
app.use('/demucs-web', (req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Content-Security-Policy', [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: https://cdn.jsdelivr.net",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data: https:",
    "media-src 'self' blob: data: http://localhost:* https:",
    "connect-src 'self' blob: http://localhost:* https://cdn.jsdelivr.net https://huggingface.co https://*.huggingface.co https://*.hf.co",
    "worker-src 'self' blob:",
    "child-src 'self' blob:"
  ].join('; '));
  next();
}, express.static(path.join(__dirname, '../public/demucs-web')));

// Health check
app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'ACE-Step UI API' });
});

// Shutdown endpoint - kills all ACE-Step processes
app.post('/api/shutdown', async (req, res) => {
  const preserveRoot = req.query.preserve_root === '1';
  console.log(`[Shutdown] Shutdown requested (preserve_root=${preserveRoot}) — terminating ACE-Step processes...`);
  res.json({ success: true, message: 'Shutting down...' });

  // Give the response a moment to flush
  setTimeout(async () => {
    try {
      const { execSync } = await import('child_process');

      // Helper: run a command and return stdout, or empty string on failure
      const run = (cmd: string): string => {
        try {
          return execSync(cmd, { encoding: 'utf-8', timeout: 5000 }).trim();
        } catch {
          return '';
        }
      };

      // ── 1. Snapshot the entire process table in ONE call (~200ms) ──
      const psCmd = `powershell -NoProfile -Command "Get-CimInstance Win32_Process | ForEach-Object { $_.ProcessId.ToString() + '|' + $_.ParentProcessId.ToString() + '|' + $_.Name + '|' + $_.CommandLine } "`;
      const psOut = run(psCmd);
      const procMap = new Map<string, { parentPid: string; name: string; cmdLine: string }>();
      for (const line of psOut.split('\n')) {
        const parts = line.trim().split('|');
        if (parts.length >= 3) {
          procMap.set(parts[0], { parentPid: parts[1], name: parts[2].toLowerCase(), cmdLine: parts.slice(3).join('|') });
        }
      }
      console.log(`[Shutdown] Loaded ${procMap.size} processes from snapshot`);

      // ── 2. Walk UP the tree from a PID, collecting shell ancestors ──
      // We only target cmd.exe and conhost.exe.
      // We MUST NOT target windowsterminal.exe, powershell.exe, or pwsh.exe,
      // as they may be hosting the parent `launch.bat` script which needs to survive to run the update loop.
      const shellNames = new Set(['cmd.exe', 'conhost.exe']);
      const collectAncestors = (startPid: string): string[] => {
        const ancestors: string[] = [];
        let current = startPid;
        const visited = new Set<string>();
        for (let depth = 0; depth < 10; depth++) {
          const info = procMap.get(current);
          if (!info) break;
          const pp = info.parentPid;
          if (!pp || pp === '0' || pp === '4' || visited.has(pp)) break;
          visited.add(pp);
          const parentInfo = procMap.get(pp);
          if (parentInfo && shellNames.has(parentInfo.name)) {
            // Safety check: Don't kill the main launch.bat cmd.exe during updates
            if (preserveRoot && parentInfo.name === 'cmd.exe') {
               const cmdLine = (parentInfo.cmdLine || '').toLowerCase();
               // We only want to kill cmd.exe wrappers created by npm, or our `cmd /k` windows.
               // launch.bat's cmd.exe lacks these signatures.
               if (!cmdLine.includes('/k') && !cmdLine.includes('npm') && !cmdLine.includes('node') && !cmdLine.includes('python')) {
                   console.log(`[Shutdown] Refusing to kill root cmd.exe ${pp}: ${cmdLine}`);
                   break;
               }
            }
            ancestors.push(pp);
            current = pp;
          } else {
            break;
          }
        }
        return ancestors;
      };

      // ── 3. Find PIDs on target ports (netstat is fast, <100ms) ──
      const findPidsOnPort = (port: number): string[] => {
        const out = run(`netstat -aon | findstr :${port} | findstr LISTENING`);
        const pids: string[] = [];
        for (const line of out.split('\n')) {
          const pid = line.trim().split(/\s+/).pop();
          if (pid && /^\d+$/.test(pid) && pid !== '0') pids.push(pid);
        }
        return [...new Set(pids)];
      };

      // ── 4. Collect all PIDs to kill ──
      const pidsToKill = new Set<string>();

      for (const pid of findPidsOnPort(8001)) {
        console.log(`[Shutdown] Python API PID ${pid}`);
        pidsToKill.add(pid);
        collectAncestors(pid).forEach(a => { console.log(`[Shutdown]   ancestor ${a}`); pidsToKill.add(a); });
      }

      const vitePort = parseInt(process.env.VITE_PORT || '3000', 10);
      for (const pid of findPidsOnPort(vitePort)) {
        console.log(`[Shutdown] Vite PID ${pid} (port ${vitePort})`);
        pidsToKill.add(pid);
        collectAncestors(pid).forEach(a => { console.log(`[Shutdown]   ancestor ${a}`); pidsToKill.add(a); });
      }

      const expressPid = String(process.pid);
      console.log(`[Shutdown] Express PID ${expressPid}`);
      collectAncestors(expressPid).forEach(a => { console.log(`[Shutdown]   ancestor ${a}`); pidsToKill.add(a); });


      // ── 5. Kill everything ──
      console.log(`[Shutdown] Killing ${pidsToKill.size} processes...`);
      
      // We must avoid committing suicide before killing other windows.
      // Easiest and safest way: pass ALL PIDs to a single detached taskkill command that outlives Node.
      if (pidsToKill.size > 0) {
        const pidArgs = Array.from(pidsToKill).map(pid => `/PID ${pid}`).join(' ');
        
        const { spawn } = await import('child_process');
        
        // Spawn detached taskkill so Node's death doesn't kill the taskkill process itself
        const child = spawn('cmd', ['/c', `taskkill /F /T ${pidArgs}`], {
          detached: true,
          stdio: 'ignore',
          windowsHide: true
        });
        child.unref();
      }

    } catch (e) {
      console.error('[Shutdown] Error during shutdown', e);
    }
    // Exit this Express process last (if it isn't killed already)
    setTimeout(() => process.exit(0), 500);
  }, 300);
});

// oEmbed endpoint for rich embeds
app.get('/api/oembed', async (req, res) => {
  const url = req.query.url as string;
  if (!url) {
    res.status(400).json({ error: 'URL required' });
    return;
  }

  const match = url.match(/\/song\/([a-zA-Z0-9-]+)/);
  if (!match) {
    res.status(404).json({ error: 'Song not found' });
    return;
  }

  try {
    const result = await pool.query(
      `SELECT s.id, s.title, s.style, s.cover_url, s.duration,
              COALESCE(u.username, 'Anonymous') as creator
       FROM songs s
       LEFT JOIN users u ON s.user_id = u.id
       WHERE s.id = ? AND s.is_public = 1`,
      [match[1]]
    );

    if (result.rows.length === 0) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }

    const song = result.rows[0];
    res.json({
      version: '1.0',
      type: 'rich',
      provider_name: 'ACE-Step UI',
      provider_url: config.frontendUrl,
      title: song.title,
      author_name: song.creator,
      thumbnail_url: song.cover_url,
      thumbnail_width: 400,
      thumbnail_height: 400,
      html: `<iframe src="${config.frontendUrl}/embed/${song.id}" width="100%" height="152" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>`,
      width: 400,
      height: 152
    });
  } catch (error) {
    console.error('oEmbed error:', error);
    res.status(500).json({ error: 'Internal error' });
  }
});

// Song share page handler
app.get('/song/:id', async (req, res) => {
  const songId = req.params.id;
  const userAgent = req.get('User-Agent') || '';

  // Check if request is from a social media bot
  const isSocialBot = /twitterbot|facebookexternalhit|linkedinbot|slackbot|redditbot|discordbot|telegrambot|whatsapp|pinterestbot|tumblr|embedly|quora|outbrain|vkshare|w3c_validator|baiduspider|bingbot/i.test(userAgent);

  if (!isSocialBot) {
    res.redirect(`${config.frontendUrl}?song=${songId}`);
    return;
  }

  try {
    const result = await pool.query(
      `SELECT s.id, s.title, s.style, s.cover_url, s.audio_url, s.duration, s.like_count, s.view_count,
              COALESCE(u.username, 'Anonymous') as creator
       FROM songs s
       LEFT JOIN users u ON s.user_id = u.id
       WHERE s.id = ? AND s.is_public = 1`,
      [songId]
    );

    if (result.rows.length === 0) {
      res.redirect(config.frontendUrl);
      return;
    }

    const song = result.rows[0];
    const coverUrl = song.cover_url || `https://picsum.photos/seed/${song.id}/1200/630`;
    const title = `${song.title} by ${song.creator}`;
    const description = `🎵 ${song.style} • Create your own AI music free on ACE-Step UI`;
    const pageUrl = `${config.frontendUrl}/song/${song.id}`;

    res.send(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title} | ACE-Step UI</title>
  <meta name="title" content="${title}">
  <meta name="description" content="${description}">
  <meta property="og:type" content="music.song">
  <meta property="og:url" content="${pageUrl}">
  <meta property="og:title" content="${title}">
  <meta property="og:description" content="${description}">
  <meta property="og:image" content="${coverUrl}">
  <meta property="og:site_name" content="ACE-Step UI">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="${title}">
  <meta name="twitter:description" content="${description}">
  <meta name="twitter:image" content="${coverUrl}">
  <meta http-equiv="refresh" content="0;url=${config.frontendUrl}?song=${song.id}">
</head>
<body>
  <p>Redirecting to <a href="${config.frontendUrl}?song=${song.id}">ACE-Step UI</a>...</p>
</body>
</html>`);
  } catch (error) {
    console.error('Error serving song share page:', error);
    res.redirect(config.frontendUrl);
  }
});

// Image proxy for CORS
app.get('/api/proxy/image', async (req, res) => {
  const url = req.query.url as string;
  if (!url) {
    res.status(400).json({ error: 'URL required' });
    return;
  }

  try {
    const response = await fetch(url);
    if (!response.ok) {
      res.status(response.status).json({ error: 'Failed to fetch image' });
      return;
    }

    const contentType = response.headers.get('content-type') || 'image/jpeg';
    const buffer = await response.arrayBuffer();

    res.setHeader('Content-Type', contentType);
    res.setHeader('Cache-Control', 'public, max-age=86400');
    res.send(Buffer.from(buffer));
  } catch (error) {
    console.error('Image proxy error:', error);
    res.status(500).json({ error: 'Failed to proxy image' });
  }
});

// Pexels API proxy - accepts API key from header or uses server config
app.get('/api/pexels/photos', async (req, res) => {
  const query = req.query.query as string;
  if (!query) {
    res.status(400).json({ error: 'Query required' });
    return;
  }

  // Accept API key from header (user-provided) or fall back to server config
  const apiKey = req.headers['x-pexels-api-key'] as string || config.pexels.apiKey;

  if (!apiKey) {
    res.status(400).json({ error: 'Pexels API key not configured. Please set your API key in the Video Generator settings.' });
    return;
  }

  try {
    const response = await fetch(
      `https://api.pexels.com/v1/search?query=${encodeURIComponent(query)}&per_page=20&orientation=landscape`,
      { headers: { Authorization: apiKey } }
    );

    if (!response.ok) {
      if (response.status === 401) {
        res.status(401).json({ error: 'Invalid Pexels API key' });
        return;
      }
      res.status(response.status).json({ error: 'Pexels API error' });
      return;
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Pexels photos error:', error);
    res.status(500).json({ error: 'Failed to fetch from Pexels' });
  }
});

app.get('/api/pexels/videos', async (req, res) => {
  const query = req.query.query as string;
  if (!query) {
    res.status(400).json({ error: 'Query required' });
    return;
  }

  // Accept API key from header (user-provided) or fall back to server config
  const apiKey = req.headers['x-pexels-api-key'] as string || config.pexels.apiKey;

  if (!apiKey) {
    res.status(400).json({ error: 'Pexels API key not configured. Please set your API key in the Video Generator settings.' });
    return;
  }

  try {
    const response = await fetch(
      `https://api.pexels.com/videos/search?query=${encodeURIComponent(query)}&per_page=15&orientation=landscape`,
      { headers: { Authorization: apiKey } }
    );

    if (!response.ok) {
      if (response.status === 401) {
        res.status(401).json({ error: 'Invalid Pexels API key' });
        return;
      }
      res.status(response.status).json({ error: 'Pexels API error' });
      return;
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Pexels videos error:', error);
    res.status(500).json({ error: 'Failed to fetch from Pexels' });
  }
});

// Search endpoint
app.get('/api/search', async (req, res) => {
  const query = (req.query.q as string)?.trim();
  const type = req.query.type as string;

  if (!query) {
    res.status(400).json({ error: 'Search query required' });
    return;
  }

  try {
    const searchPattern = `%${query}%`;
    const results: { songs: unknown[]; creators: unknown[]; playlists: unknown[] } = {
      songs: [],
      creators: [],
      playlists: [],
    };

    if (!type || type === 'all' || type === 'songs') {
      const songsResult = await pool.query(
        `SELECT s.id, s.title, s.lyrics, s.style, s.caption, s.cover_url, s.audio_url,
                s.duration, s.tags, s.like_count, s.view_count, s.is_public, s.created_at,
                u.username as creator, u.avatar_url as creator_avatar
         FROM songs s
         LEFT JOIN users u ON s.user_id = u.id
         WHERE s.is_public = 1
           AND (s.title LIKE ? COLLATE NOCASE OR s.style LIKE ? COLLATE NOCASE)
         ORDER BY s.like_count DESC
         LIMIT 20`,
        [searchPattern, searchPattern]
      );
      results.songs = songsResult.rows;
    }

    if (!type || type === 'all' || type === 'creators') {
      const creatorsResult = await pool.query(
        `SELECT u.id, u.username, u.bio, u.avatar_url, u.created_at,
                (SELECT COUNT(*) FROM followers WHERE following_id = u.id) as follower_count
         FROM users u
         WHERE u.username LIKE ? COLLATE NOCASE
         ORDER BY (SELECT COUNT(*) FROM followers WHERE following_id = u.id) DESC
         LIMIT 20`,
        [searchPattern]
      );
      results.creators = creatorsResult.rows;
    }

    if (!type || type === 'all' || type === 'playlists') {
      const playlistsResult = await pool.query(
        `SELECT p.id, p.name, p.description, p.cover_url, p.created_at,
                u.username as creator, u.avatar_url as creator_avatar,
                (SELECT COUNT(*) FROM playlist_songs ps WHERE ps.playlist_id = p.id) as song_count
         FROM playlists p
         JOIN users u ON p.user_id = u.id
         WHERE p.is_public = 1 AND p.name LIKE ? COLLATE NOCASE
         ORDER BY (SELECT COUNT(*) FROM playlist_songs ps WHERE ps.playlist_id = p.id) DESC
         LIMIT 20`,
        [searchPattern]
      );
      results.playlists = playlistsResult.rows;
    }

    res.json(results);
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).json({ error: 'Search failed' });
  }
});

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/songs', songsRoutes);
app.use('/api/generate', generateRoutes);
app.use('/api/users', usersRoutes);
app.use('/api/playlists', playlistsRoutes);
app.use('/api/contact', contactRoutes);
app.use('/api/reference-tracks', referenceTrackRoutes);
app.use('/api/lora', loraRoutes);
app.use('/api/steering', steeringRoutes);
app.use('/api/models', modelsRoutes);
app.use('/api/system', systemRoutes);
app.use('/api/analyze', analyzeRoutes);
app.use('/api/model', modelRoutes);

// Error handler
app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Schedule cleanup job to run daily at 3 AM
cron.schedule('0 3 * * *', async () => {
  console.log('Running scheduled cleanup job...');
  try {
    await runCleanupJob();
    await cleanupDeletedSongs();
  } catch (error) {
    console.error('Cleanup job failed:', error);
  }
});

// Start server on configured host (default 0.0.0.0 for LAN access)
app.listen(config.port, config.host, () => {
  console.log(`ACE-Step UI Server running on http://localhost:${config.port}`);
  console.log(`Environment: ${config.nodeEnv}`);
  console.log(`ACE-Step API: ${config.acestep.apiUrl}`);

  // Show LAN access info
  import('os').then(os => {
    const nets = os.networkInterfaces();
    for (const name of Object.keys(nets)) {
      for (const net of nets[name] || []) {
        if (net.family === 'IPv4' && !net.internal) {
          console.log(`LAN access: http://${net.address}:${config.port}`);
        }
      }
    }
  });
});
