import { Router, Response } from 'express';
import { Readable } from 'node:stream';
import { v4 as uuidv4 } from 'uuid';
import { pool } from '../db/pool.js';
import { authMiddleware, optionalAuthMiddleware, AuthenticatedRequest } from '../middleware/auth.js';
import { getStorageProvider } from '../services/storage/factory.js';
import { config } from '../config/index.js';
import { spawn } from 'child_process';
import path from 'path';

const router = Router();

// Helper: resolve audio URL (generates signed URL for S3)
async function resolveAudioUrl(audioUrl: string | null): Promise<string | null> {
  if (!audioUrl) return null;

  if (audioUrl.startsWith('s3://')) {
    const storageKey = audioUrl.replace('s3://', '');
    const storage = getStorageProvider();
    return storage.getUrl(storageKey, 3600); // 1 hour expiry
  }

  return audioUrl;
}

// Helper: resolve audio URL for direct playback
async function resolveAccessibleAudioUrl(audioUrl: string | null, isPublic: boolean): Promise<string | null> {
  if (!audioUrl) return null;
  if (audioUrl.startsWith('s3://')) {
    const storageKey = audioUrl.replace('s3://', '');
    const storage = getStorageProvider();
    return isPublic ? storage.getPublicUrl(storageKey) : storage.getUrl(storageKey, 3600);
  }
  return audioUrl;
}

// Get audio - proxies from S3 to avoid CORS issues
router.get('/:id/audio', optionalAuthMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await pool.query(
      `SELECT s.audio_url, s.is_public, s.user_id FROM songs s WHERE s.id = $1`,
      [req.params.id]
    );

    if (result.rows.length === 0) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }

    const song = result.rows[0];

    if (!song.is_public && (!req.user || req.user.id !== song.user_id)) {
      res.status(403).json({ error: 'Access denied' });
      return;
    }

    const audioUrl = await resolveAudioUrl(song.audio_url);
    if (!audioUrl) {
      res.status(404).json({ error: 'Audio not available' });
      return;
    }

    // Local files - redirect
    if (audioUrl.startsWith('/')) {
      res.redirect(audioUrl);
      return;
    }

    // S3/remote - proxy to avoid CORS
    const range = req.headers.range;
    const audioRes = await fetch(audioUrl, {
      headers: range ? { Range: range } : undefined,
    });
    if (!audioRes.ok && audioRes.status !== 206) {
      res.status(502).json({ error: 'Failed to fetch audio' });
      return;
    }

    const contentType = audioRes.headers.get('content-type') || 'audio/mpeg';
    res.setHeader('Content-Type', contentType);
    res.setHeader('Accept-Ranges', 'bytes');

    const contentLength = audioRes.headers.get('content-length');
    if (contentLength) {
      res.setHeader('Content-Length', contentLength);
    }

    const contentRange = audioRes.headers.get('content-range');
    if (contentRange) {
      res.status(206);
      res.setHeader('Content-Range', contentRange);
    }

    if (audioRes.body) {
      Readable.fromWeb(audioRes.body as any).pipe(res);
      return;
    }

    const arrayBuffer = await audioRes.arrayBuffer();
    res.send(Buffer.from(arrayBuffer));
  } catch (error) {
    console.error('Get audio error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Look up DB songs by audio URLs (batch) — used by RecordingsTab to get cover art, originalAudioUrl, etc.
router.post('/by-urls', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { audioUrls } = req.body as { audioUrls: string[] };
    if (!Array.isArray(audioUrls) || audioUrls.length === 0) {
      res.json({ songs: [] });
      return;
    }
    // Build parameterised query for the URL list
    const placeholders = audioUrls.map((_, i) => `$${i + 2}`).join(', ');
    const result = await pool.query(
      `SELECT s.id, s.title, s.lyrics, s.style, s.caption, s.audio_url, s.cover_url,
              s.duration, s.bpm, s.key_scale, s.time_signature, s.tags, s.generation_params,
              s.source, s.created_at
       FROM songs s
       WHERE s.user_id = $1 AND s.audio_url IN (${placeholders})`,
      [req.user!.id, ...audioUrls]
    );
    const songs = result.rows.map((r: any) => ({
      id: r.id,
      title: r.title,
      lyrics: r.lyrics || '',
      style: r.style || '',
      audioUrl: r.audio_url,
      coverUrl: r.cover_url || '',
      duration: r.duration?.toString() || '0',
      createdAt: r.created_at,
      tags: typeof r.tags === 'string' ? JSON.parse(r.tags || '[]') : (r.tags || []),
      generationParams: typeof r.generation_params === 'string'
        ? JSON.parse(r.generation_params || '{}')
        : (r.generation_params || {}),
    }));
    res.json({ songs });
  } catch (error) {
    console.error('Fetch songs by URLs error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Helper: build ffmpeg metadata args from song data
function buildMetadataArgs(song: Record<string, unknown>, creator: string, format: string): string[] {
  const args: string[] = [];
  const add = (key: string, value: unknown) => {
    if (value !== null && value !== undefined && String(value).trim() !== '') {
      args.push('-metadata', `${key}=${String(value)}`);
    }
  };

  const songTitle = song.title as string || '';
  const style = song.style as string || '';
  const caption = song.caption as string || '';
  const lyrics = song.lyrics as string || '';
  const bpm = song.bpm as number | null;
  const keyScale = song.key_scale as string || '';
  const createdAt = song.created_at as string || '';
  const model = song.model as string || '';
  const genParams = song.generation_params;

  // Build comment field with custom info (model, seed, etc.)
  let comment = caption;
  const extras: string[] = [];
  if (model) extras.push(`Model: ${model}`);
  if (genParams) {
    try {
      const params = typeof genParams === 'string' ? JSON.parse(genParams) : genParams;
      if (params?.seed !== undefined) extras.push(`Seed: ${params.seed}`);
    } catch { /* ignore parse errors */ }
  }
  if (extras.length > 0) {
    comment = comment
      ? `${comment}\n\nGenerated by ACE-Step | ${extras.join(' | ')}`
      : `Generated by ACE-Step | ${extras.join(' | ')}`;
  }

  const year = createdAt ? new Date(createdAt).getFullYear().toString() : '';

  if (format === 'mp3') {
    // ID3v2 tags
    add('title', songTitle);
    add('artist', creator);
    add('genre', style);
    add('comment', comment);
    add('TBPM', bpm);
    add('TKEY', keyScale);
    add('date', year);
    if (lyrics) add('lyrics-eng', lyrics);
  } else if (format === 'flac' || format === 'opus') {
    // Vorbis comments — supports arbitrary keys
    add('TITLE', songTitle);
    add('ARTIST', creator);
    add('GENRE', style);
    add('DESCRIPTION', caption);
    add('COMMENT', extras.length > 0 ? `Generated by ACE-Step | ${extras.join(' | ')}` : '');
    add('BPM', bpm);
    add('KEY', keyScale);
    add('DATE', createdAt ? createdAt.split('T')[0] : '');
    if (lyrics) add('LYRICS', lyrics);
  } else if (format === 'wav') {
    // RIFF INFO — limited set
    add('title', songTitle);
    add('artist', creator);
    add('genre', style);
    add('comment', comment);
    add('date', year);
  }

  return args;
}

// Download audio with format conversion and metadata tagging
router.get('/download', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const audioUrl = req.query.audioUrl as string;
    const format = req.query.format as string;
    const title = (req.query.title as string) || 'download';
    const songId = req.query.songId as string | undefined;
    const mp3Bitrate = req.query.mp3Bitrate as string | undefined;
    const opusBitrate = req.query.opusBitrate as string | undefined;

    if (!audioUrl || !format) {
      res.status(400).json({ error: 'audioUrl and format are required' });
      return;
    }

    const validFormats = ['mp3', 'flac', 'wav', 'opus'];
    if (!validFormats.includes(format)) {
      res.status(400).json({ error: 'Invalid format requested' });
      return;
    }

    // Determine the actual file path. 
    // In this app, audioUrl might be a relative path like /audio/xxxxx.xxx.
    const urlObj = new URL(audioUrl, `${req.protocol}://${req.get('host')}`);
    let sourcePath = '';

    // We only support local files via /audio/ mounts or /api/audio/file for conversion currently
    if (urlObj.pathname.startsWith('/audio/')) {
      const audioDir = config.storage.audioDir;
      const filename = decodeURIComponent(urlObj.pathname.replace('/audio/', ''));
      sourcePath = path.join(audioDir, filename);
    } else if (urlObj.pathname === '/api/audio/file') {
      const filePath = urlObj.searchParams.get('path');
      if (!filePath) {
        res.status(400).json({ error: 'Missing path parameter in audioUrl' });
        return;
      }

      // Handle absolute or relative paths
      if (path.isAbsolute(filePath)) {
        sourcePath = filePath;
      } else {
        // Assume it's relative to the datasets or audio dir. The `/api/audio/file` route checks 
        // multiple places, but for generated songs they usually live in audioDir or datasets.
        sourcePath = path.join(config.storage.audioDir, filePath);

        // Fallback: If it's not in audioDir, let's assume it might be a raw dataset path
        const fs = await import('fs');
        if (!fs.existsSync(sourcePath)) {
          // Same base resolution logic as /api/audio/file in index.ts
          const serverRoot = path.join(process.cwd());
          const projectRoot = path.join(serverRoot, '..');
          const workspaceRoot = path.join(projectRoot, '..');

          const possible = [
            path.join(workspaceRoot, filePath),
            path.join(projectRoot, filePath),
            path.join(serverRoot, filePath)
          ];

          sourcePath = possible.find(p => fs.existsSync(p)) || sourcePath;
        }
      }
    } else if (urlObj.pathname === '/v1/audio') {
      const filePath = urlObj.searchParams.get('path');
      if (!filePath) {
        res.status(400).json({ error: 'Missing path parameter in audioUrl' });
        return;
      }
      // This is typically an absolute path coming from the fastAPI backend
      sourcePath = filePath;
    } else {
      res.status(400).json({ error: 'Unsupported audioUrl format for conversion' });
      return;
    }

    const fs = await import('fs');
    if (!fs.existsSync(sourcePath)) {
      console.error('[Download Route] Source file not found on disk:', sourcePath, 'Original URL:', audioUrl);
      res.status(404).json({ error: 'Source file not found on disk: ' + sourcePath });
      return;
    }

    console.log('[Download Route] Successfully resolved source path:', sourcePath);

    // Fetch song metadata from DB if songId provided
    let metadataArgs: string[] = [];
    if (songId && !songId.startsWith('temp_') && !songId.startsWith('job_')) {
      try {
        const songResult = await pool.query(
          `SELECT s.title, s.lyrics, s.style, s.caption, s.bpm, s.key_scale,
                  s.time_signature, s.model, s.generation_params, s.created_at,
                  COALESCE(u.username, 'Anonymous') as creator
           FROM songs s
           LEFT JOIN users u ON s.user_id = u.id
           WHERE s.id = $1`,
          [songId]
        );
        if (songResult.rows.length > 0) {
          metadataArgs = buildMetadataArgs(songResult.rows[0], songResult.rows[0].creator, format);
          console.log(`[Download Route] Embedding ${metadataArgs.length / 2} metadata tags for song ${songId}`);
        }
      } catch (err) {
        console.error('[Download Route] Failed to fetch song metadata, continuing without tags:', err);
      }
    }

    // Determine output file extension
    const outputExt = format === 'opus' ? 'ogg' : format;

    // Clean up title for filename header to prevent quote breakage
    const safeTitle = title.replace(/"/g, "'").replace(/[\\/]/g, "");

    // Set headers for download
    res.setHeader('Content-Disposition', `attachment; filename="${safeTitle}.${outputExt}"`);
    res.setHeader('Content-Type', `audio/${format === 'mp3' ? 'mpeg' : (format === 'opus' ? 'ogg' : format)}`);

    const currentExt = path.extname(sourcePath).toLowerCase().replace('.', '');
    const hasMetadata = metadataArgs.length > 0;

    if (currentExt === format && !hasMetadata) {
      // No conversion and no metadata to embed — just stream the file directly
      console.log('[Download Route] Starting native stream for:', sourcePath);

      const readStream = fs.createReadStream(sourcePath);
      readStream.on('error', (err) => {
        console.error('[Download Route] Native read stream error:', err);
        if (!res.headersSent) {
          res.status(500).json({ error: 'Failed to read source file' });
        }
      });
      readStream.pipe(res);
    } else {
      console.log('[Download Route] Starting FFMPEG conversion to', format, hasMetadata ? 'with metadata' : '');
      // We need to convert (or re-encode to embed metadata) using ffmpeg
      const crypto = await import('crypto');
      const os = await import('os');

      const tempFilename = path.join(os.tmpdir(), `ace-dl-${crypto.randomUUID()}.${outputExt}`);
      const ffmpegArgs = ['-i', sourcePath];

      if (format === 'mp3') {
        ffmpegArgs.push('-c:a', 'libmp3lame');
        // Bitrate: VBR V0 by default, or CBR if specified
        if (mp3Bitrate && mp3Bitrate !== 'V0') {
          ffmpegArgs.push('-b:a', `${mp3Bitrate}k`);
        } else {
          ffmpegArgs.push('-q:a', '0');
        }
        ffmpegArgs.push('-f', 'mp3');
      } else if (format === 'flac') {
        ffmpegArgs.push('-c:a', 'flac', '-sample_fmt', 's16', '-ar', '44100', '-f', 'flac');
      } else if (format === 'wav') {
        ffmpegArgs.push('-c:a', 'pcm_s16le', '-f', 'wav');
      } else if (format === 'opus') {
        const bitrate = opusBitrate || '128';
        ffmpegArgs.push('-c:a', 'libopus', '-b:a', `${bitrate}k`, '-vbr', 'on', '-compression_level', '10', '-f', 'ogg');
      }

      // Add metadata tags
      ffmpegArgs.push(...metadataArgs);

      ffmpegArgs.push('-y', tempFilename); // Write to temp file instead of pipe

      const ffmpeg = spawn('ffmpeg', ffmpegArgs);

      let ffmpegErr = '';
      ffmpeg.stderr.on('data', (data) => {
        ffmpegErr += data.toString();
      });

      ffmpeg.on('close', (code) => {
        if (code !== 0) {
          console.error(`ffmpeg process exited with code ${code}. err: ${ffmpegErr}`);
          if (!res.headersSent) {
            res.status(500).json({ error: 'Audio conversion failed' });
          }
          // Cleanup
          if (fs.existsSync(tempFilename)) fs.unlinkSync(tempFilename);
        } else {
          // Send the converted file
          res.sendFile(tempFilename, (err) => {
            if (err) console.error('Error sending converted file:', err);
            // Cleanup temp file after sending finishes or fails
            if (fs.existsSync(tempFilename)) fs.unlinkSync(tempFilename);
          });
        }
      });

      // Ensure we kill the process and clean up if the client aborts the request
      req.on('close', () => {
        ffmpeg.kill();
        setTimeout(() => {
          if (fs.existsSync(tempFilename)) fs.unlinkSync(tempFilename);
        }, 1000); // Give ffmpeg time to die before deleting its output file
      });
    }

  } catch (error) {
    console.error('Download audio error:', error);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Internal server error' });
    }
  }
});

// Get user's songs (optional ?source= filter, default: create + NULL)
router.get('/', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const sourceFilter = (req.query.source as string) || null;
    // Build LIKE pattern for generation_params fallback (catches pre-fix covers)
    const sourceJsonPattern = sourceFilter ? `%"source":"${sourceFilter}"%` : null;
    const result = await pool.query(
      `SELECT s.id, s.title, s.lyrics, s.style, s.caption, s.cover_url, s.audio_url,
              s.duration, s.bpm, s.key_scale, s.time_signature, s.tags, s.is_public,
              s.like_count, s.view_count, s.user_id, s.model as dit_model, s.created_at,
              s.generation_params,
              COALESCE(u.username, 'Anonymous') as creator
       FROM songs s
       LEFT JOIN users u ON s.user_id = u.id
       WHERE s.user_id = $1 AND (
         CASE WHEN $2 IS NOT NULL THEN (
           s.source = $2
           OR s.generation_params LIKE $3
         )
         ELSE (s.source IS NULL OR s.source = 'create')
         END
       )
       ORDER BY s.created_at DESC`,
      [req.user!.id, sourceFilter, sourceJsonPattern]
    );

    const songs = await Promise.all(
      result.rows.map(async (row) => ({
        ...row,
        ditModel: row.dit_model,
        audio_url: await resolveAccessibleAudioUrl(row.audio_url, row.is_public),
      }))
    );

    res.json({ songs });
  } catch (error) {
    console.error('Get songs error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get featured songs (random songs for discover page)
router.get('/public/featured', optionalAuthMiddleware, async (_req: AuthenticatedRequest, res: Response) => {
  try {
    // Return random songs - for local app, show all songs randomly
    const result = await pool.query(
      `SELECT s.id, s.title, s.lyrics, s.style, s.caption, s.cover_url, s.audio_url,
              s.duration, s.bpm, s.key_scale, s.time_signature, s.tags, s.like_count, s.view_count, s.created_at, s.user_id,
              COALESCE(u.username, 'Anonymous') as creator, u.avatar_url as creator_avatar, s.generation_params
       FROM songs s
       LEFT JOIN users u ON s.user_id = u.id
       ORDER BY RANDOM()
       LIMIT 20`
    );

    const songs = await Promise.all(
      result.rows.map(async (row) => ({
        id: row.id,
        title: row.title,
        lyrics: row.lyrics,
        style: row.style,
        caption: row.caption,
        cover_url: row.cover_url,
        audio_url: await resolveAccessibleAudioUrl(row.audio_url, true),
        duration: row.duration,
        bpm: row.bpm,
        key_scale: row.key_scale,
        time_signature: row.time_signature,
        generation_params: row.generation_params,
        tags: row.tags || [],
        like_count: row.like_count || 0,
        view_count: row.view_count || 0,
        ditModel: row.dit_model,
        created_at: row.created_at,
        creator: row.creator,
        creator_avatar: row.creator_avatar,
        user_id: row.user_id,
        is_public: true
      }))
    );

    res.json({ songs });
  } catch (error) {
    console.error('Get featured/random songs error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get public songs (for explore/home)
router.get('/public', optionalAuthMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const limit = Math.min(parseInt(req.query.limit as string) || 20, 100);
    const offset = parseInt(req.query.offset as string) || 0;

    const result = await pool.query(
      `SELECT s.id, s.title, s.lyrics, s.style, s.caption, s.cover_url, s.audio_url,
              s.duration, s.bpm, s.key_scale, s.time_signature, s.tags, s.like_count, s.created_at,
              COALESCE(u.username, 'Anonymous') as creator, s.generation_params
       FROM songs s
       LEFT JOIN users u ON s.user_id = u.id
       WHERE s.is_public = true
       ORDER BY s.created_at DESC
       LIMIT $1 OFFSET $2`,
      [limit, offset]
    );

    const songs = await Promise.all(
      result.rows.map(async (row) => ({
        ...row,
        ditModel: row.dit_model,
        audio_url: await resolveAccessibleAudioUrl(row.audio_url, true),
      }))
    );

    res.json({ songs });
  } catch (error) {
    console.error('Get public songs error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get single song
router.get('/:id', optionalAuthMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await pool.query(
      `SELECT s.id, s.user_id, s.title, s.lyrics, s.style, s.caption, s.cover_url, s.audio_url,
              s.duration, s.bpm, s.key_scale, s.time_signature, s.tags, s.is_public, s.like_count, s.view_count, s.created_at,
              COALESCE(u.username, 'Anonymous') as creator, u.avatar_url as creator_avatar, s.generation_params
       FROM songs s
       LEFT JOIN users u ON s.user_id = u.id
       WHERE s.id = $1`,
      [req.params.id]
    );

    if (result.rows.length === 0) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }

    const song = result.rows[0];

    // Check access
    if (!song.is_public && (!req.user || req.user.id !== song.user_id)) {
      res.status(403).json({ error: 'Access denied' });
      return;
    }

    const resolvedSong = {
      ...song,
      ditModel: song.dit_model,
      audio_url: await resolveAccessibleAudioUrl(song.audio_url, song.is_public),
    };

    res.json({ song: resolvedSong });
  } catch (error) {
    console.error('Get song error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get full song details (including comments)
router.get('/:id/full', optionalAuthMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const [songResult, commentsResult] = await Promise.all([
      pool.query(
        `SELECT s.id, s.user_id, s.title, s.lyrics, s.style, s.caption, s.cover_url, s.audio_url,
                s.duration, s.bpm, s.key_scale, s.time_signature, s.tags, s.is_public,
                s.like_count, s.view_count, s.created_at, s.generation_params,
                COALESCE(u.username, 'Anonymous') as creator, u.avatar_url as creator_avatar
         FROM songs s
         LEFT JOIN users u ON s.user_id = u.id
         WHERE s.id = $1`,
        [req.params.id]
      ),
      pool.query(
        `SELECT c.id, c.content, c.created_at, c.updated_at,
                u.id as user_id, u.username, u.avatar_url
         FROM comments c
         JOIN users u ON c.user_id = u.id
         WHERE c.song_id = $1
         ORDER BY c.created_at DESC`,
        [req.params.id]
      )
    ]);

    if (songResult.rows.length === 0) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }

    const song = songResult.rows[0];

    // Check access
    if (!song.is_public && (!req.user || req.user.id !== song.user_id)) {
      res.status(403).json({ error: 'Access denied' });
      return;
    }

    // Increment view count
    await pool.query('UPDATE songs SET view_count = view_count + 1 WHERE id = $1', [req.params.id]);

    const resolvedSong = {
      ...song,
      ditModel: song.dit_model,
      audio_url: await resolveAccessibleAudioUrl(song.audio_url, song.is_public),
    };

    res.json({
      song: resolvedSong,
      comments: commentsResult.rows
    });
  } catch (error) {
    console.error('Get full song error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Create song (manual, not from generation)
router.post('/', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const {
      title,
      lyrics,
      style,
      caption,
      coverUrl,
      audioUrl,
      duration,
      bpm,
      keyScale,
      timeSignature,
      tags,
      isPublic,
    } = req.body;

    const result = await pool.query(
      `INSERT INTO songs (user_id, title, lyrics, style, caption, cover_url, audio_url,
                          duration, bpm, key_scale, time_signature, tags, is_public)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
       RETURNING *`,
      [
        req.user!.id,
        title,
        lyrics,
        style,
        caption,
        coverUrl,
        audioUrl,
        duration,
        bpm,
        keyScale,
        timeSignature,
        tags || [],
        isPublic || false,
      ]
    );

    res.status(201).json({ song: result.rows[0] });
  } catch (error) {
    console.error('Create song error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Update song
router.patch('/:id', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    // Verify ownership
    const check = await pool.query('SELECT user_id FROM songs WHERE id = $1', [req.params.id]);
    if (check.rows.length === 0) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }
    if (check.rows[0].user_id !== req.user!.id) {
      res.status(403).json({ error: 'Access denied' });
      return;
    }

    const updates: string[] = [];
    const values: unknown[] = [];
    let paramCount = 1;

    const allowedFields = ['title', 'lyrics', 'style', 'caption', 'cover_url', 'is_public', 'tags'];
    for (const field of allowedFields) {
      if (req.body[field] !== undefined) {
        updates.push(`${field} = $${paramCount}`);
        values.push(req.body[field]);
        paramCount++;
      }
    }

    if (updates.length === 0) {
      res.status(400).json({ error: 'No fields to update' });
      return;
    }

    updates.push(`updated_at = CURRENT_TIMESTAMP`);
    values.push(req.params.id);

    await pool.query(
      `UPDATE songs
       SET ${updates.join(', ')}
       WHERE id = $${paramCount}`,
      values
    );

    // Fetch complete song data with creator info
    const result = await pool.query(
      `SELECT s.id, s.title, s.lyrics, s.style, s.caption, s.cover_url, s.audio_url,
              s.duration, s.bpm, s.key_scale, s.time_signature, s.tags, s.is_public,
              s.like_count, s.view_count, s.user_id, s.model as dit_model, s.created_at,
              s.generation_params,
              COALESCE(u.username, 'Anonymous') as creator
       FROM songs s
       LEFT JOIN users u ON s.user_id = u.id
       WHERE s.id = $1`,
      [req.params.id]
    );

    // Map database fields to frontend format
    const song = result.rows[0];
    const mappedSong = {
      ...song,
      ditModel: song.dit_model,
      audio_url: await resolveAccessibleAudioUrl(song.audio_url, song.is_public),
    };

    res.json({ song: mappedSong });
  } catch (error) {
    console.error('Update song error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Delete ALL songs for the authenticated user
router.delete('/all', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    // Get all songs for this user (to delete files)
    const result = await pool.query(
      'SELECT id, audio_url, cover_url FROM songs WHERE user_id = $1',
      [req.user!.id]
    );

    if (result.rows.length === 0) {
      res.json({ success: true, deletedCount: 0 });
      return;
    }

    const storage = getStorageProvider();

    // Delete all associated files
    for (const song of result.rows) {
      if (song.audio_url) {
        try {
          const storageKey = song.audio_url.startsWith('/audio/')
            ? song.audio_url.replace('/audio/', '')
            : song.audio_url.replace('s3://', '');
          await storage.delete(storageKey);
        } catch (err) {
          console.error(`Failed to delete audio file ${song.audio_url}:`, err);
        }
      }
      if (song.cover_url && song.cover_url.startsWith('/audio/')) {
        try {
          const coverKey = song.cover_url.replace('/audio/', '');
          await storage.delete(coverKey);
        } catch (err) {
          console.error(`Failed to delete cover ${song.cover_url}:`, err);
        }
      }
    }

    // Bulk delete all songs from DB
    await pool.query('DELETE FROM songs WHERE user_id = $1', [req.user!.id]);

    res.json({ success: true, deletedCount: result.rows.length });
  } catch (error) {
    console.error('Delete all songs error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Delete song
router.delete('/:id', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const check = await pool.query('SELECT user_id, audio_url, cover_url FROM songs WHERE id = $1', [req.params.id]);
    if (check.rows.length === 0) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }
    if (check.rows[0].user_id !== req.user!.id) {
      res.status(403).json({ error: 'Access denied' });
      return;
    }

    const song = check.rows[0];
    const storage = getStorageProvider();

    // Delete audio file from storage
    if (song.audio_url) {
      try {
        // Handle local storage paths (/audio/filename.mp3 -> filename.mp3)
        const storageKey = song.audio_url.startsWith('/audio/')
          ? song.audio_url.replace('/audio/', '')
          : song.audio_url.replace('s3://', '');
        await storage.delete(storageKey);
      } catch (err) {
        console.error(`Failed to delete audio file ${song.audio_url}:`, err);
      }
    }

    // Delete cover image if it's stored locally
    if (song.cover_url && song.cover_url.startsWith('/audio/')) {
      try {
        const coverKey = song.cover_url.replace('/audio/', '');
        await storage.delete(coverKey);
      } catch (err) {
        console.error(`Failed to delete cover ${song.cover_url}:`, err);
      }
    }

    await pool.query('DELETE FROM songs WHERE id = $1', [req.params.id]);
    res.json({ success: true });
  } catch (error) {
    console.error('Delete song error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Like/unlike song
router.post('/:id/like', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  const client = await pool.connect();
  try {
    await client.query('BEGIN');

    // Check if already liked
    const existing = await client.query(
      'SELECT 1 FROM liked_songs WHERE user_id = $1 AND song_id = $2',
      [req.user!.id, req.params.id]
    );

    if (existing.rows.length > 0) {
      // Unlike
      await client.query('DELETE FROM liked_songs WHERE user_id = $1 AND song_id = $2', [
        req.user!.id,
        req.params.id,
      ]);
      // Decrement like_count
      await client.query(
        'UPDATE songs SET like_count = GREATEST(like_count - 1, 0) WHERE id = $1',
        [req.params.id]
      );
      await client.query('COMMIT');
      res.json({ liked: false });
    } else {
      // Like
      await client.query('INSERT INTO liked_songs (user_id, song_id) VALUES ($1, $2)', [
        req.user!.id,
        req.params.id,
      ]);
      // Increment like_count
      await client.query(
        'UPDATE songs SET like_count = like_count + 1 WHERE id = $1',
        [req.params.id]
      );
      await client.query('COMMIT');
      res.json({ liked: true });
    }
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Like song error:', error);
    res.status(500).json({ error: 'Internal server error' });
  } finally {
    client.release();
  }
});

// Get liked songs
router.get('/liked/list', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await pool.query(
      `SELECT s.id, s.title, s.lyrics, s.style, s.cover_url, s.audio_url,
              s.duration, s.tags, s.like_count, s.created_at, s.is_public,
              COALESCE(u.username, 'Anonymous') as creator, s.generation_params
       FROM liked_songs ls
       JOIN songs s ON ls.song_id = s.id
       LEFT JOIN users u ON s.user_id = u.id
       WHERE ls.user_id = $1
       ORDER BY ls.liked_at DESC`,
      [req.user!.id]
    );

    const songs = await Promise.all(
      result.rows.map(async (row) => ({
        ...row,
        ditModel: row.dit_model,
        audio_url: await resolveAccessibleAudioUrl(row.audio_url, row.is_public),
      }))
    );

    res.json({ songs });
  } catch (error) {
    console.error('Get liked songs error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Toggle song privacy (paid users only can make songs private)
router.patch('/:id/privacy', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    // Get user's account tier
    const userResult = await pool.query('SELECT account_tier FROM users WHERE id = $1', [req.user!.id]);
    const accountTier = userResult.rows[0]?.account_tier || 'free';

    const check = await pool.query('SELECT user_id, is_public FROM songs WHERE id = $1', [req.params.id]);
    if (check.rows.length === 0) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }
    if (check.rows[0].user_id !== req.user!.id) {
      res.status(403).json({ error: 'Access denied' });
      return;
    }

    const newPublicState = !check.rows[0].is_public;

    // Free users cannot make songs private
    if (accountTier === 'free' && !newPublicState) {
      res.status(403).json({ error: 'Upgrade to Pro or Unlimited to make songs private' });
      return;
    }

    await pool.query('UPDATE songs SET is_public = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2', [
      newPublicState,
      req.params.id,
    ]);

    res.json({ isPublic: newPublicState });
  } catch (error) {
    console.error('Toggle privacy error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Track song play
router.post('/:id/play', optionalAuthMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await pool.query(
      `UPDATE songs
       SET view_count = COALESCE(view_count, 0) + 1, updated_at = CURRENT_TIMESTAMP
       WHERE id = $1
       RETURNING view_count`,
      [req.params.id]
    );

    if (result.rows.length === 0) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }

    res.json({ viewCount: result.rows[0].view_count });
  } catch (error) {
    console.error('Track play error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Get comments for a song
router.get('/:id/comments', optionalAuthMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const result = await pool.query(
      `SELECT c.id, c.content, c.created_at, u.username, u.id as user_id, u.avatar_url
       FROM comments c
       JOIN users u ON c.user_id = u.id
       WHERE c.song_id = $1
       ORDER BY c.created_at DESC`,
      [req.params.id]
    );

    res.json({ comments: result.rows });
  } catch (error) {
    console.error('Get comments error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Add comment to a song
router.post('/:id/comments', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { content } = req.body;

    if (!content || content.trim().length === 0) {
      res.status(400).json({ error: 'Comment content is required' });
      return;
    }

    // Check if song exists and is public
    const songCheck = await pool.query('SELECT is_public FROM songs WHERE id = $1', [req.params.id]);
    if (songCheck.rows.length === 0) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }
    if (!songCheck.rows[0].is_public) {
      res.status(403).json({ error: 'Cannot comment on private songs' });
      return;
    }

    const result = await pool.query(
      `INSERT INTO comments (song_id, user_id, content)
       VALUES ($1, $2, $3)
       RETURNING id, content, created_at`,
      [req.params.id, req.user!.id, content.trim()]
    );

    const comment = {
      ...result.rows[0],
      username: req.user!.username,
      user_id: req.user!.id,
    };

    res.status(201).json({ comment });
  } catch (error) {
    console.error('Add comment error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Delete comment
router.delete('/comments/:commentId', authMiddleware, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const check = await pool.query('SELECT user_id FROM comments WHERE id = $1', [req.params.commentId]);
    if (check.rows.length === 0) {
      res.status(404).json({ error: 'Comment not found' });
      return;
    }
    if (check.rows[0].user_id !== req.user!.id) {
      res.status(403).json({ error: 'Access denied' });
      return;
    }

    await pool.query('DELETE FROM comments WHERE id = $1', [req.params.commentId]);
    res.json({ success: true });
  } catch (error) {
    console.error('Delete comment error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

export default router;
