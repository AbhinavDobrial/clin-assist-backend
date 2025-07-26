import 'dotenv/config';
import express from 'express';
import expressWs from 'express-ws';
import multer from 'multer';
import { OpenAI } from 'openai';
import { v4 as uuid } from 'uuid';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const app = express();
expressWs(app); // adds .ws()

app.use(express.json({ limit: '20mb' }));

// Simple health check
app.get('/health', (_, res) => res.send('OK'));

/**
 * 1) FAST PATH: WebSocket for low-latency updates
 * - Client sends small base64 audio chunks {type:"audioChunk", data:"..."}
 * - Client sends {type:"end"} when done.
 * - Server batches 3-5s chunks -> calls Whisper -> pushes partial transcript
 * - Then calls GPT for summary at the end (or incremental if you want)
 */
const sessions = new Map(); // {sessionId: {chunks:[], transcript:''}}

app.ws('/stream', (ws, req) => {
  const sessionId = uuid();
  sessions.set(sessionId, { chunks: [], transcript: '' });

  ws.send(JSON.stringify({ type: 'session', sessionId }));

  ws.on('message', async (msg) => {
    try {
      const payload = JSON.parse(msg);

      if (payload.type === 'audioChunk') {
        // Save chunk (base64 string)
        sessions.get(sessionId).chunks.push(payload.data);
        // OPTIONAL: you can accumulate 3-5 secs then call STT here for "live"
      }

      if (payload.type === 'chunkEnd') {
        // When a chunk ends, transcribe that chunk now
        const combinedBase64 = sessions.get(sessionId).chunks.join('');
        sessions.get(sessionId).chunks = []; // reset chunk buffer

        const audioBuffer = Buffer.from(combinedBase64, 'base64');

        const transcription = await openai.audio.transcriptions.create({
          file: new File([audioBuffer], "chunk.wav", { type: "audio/wav" }),
          model: "whisper-1"
        });

        // Append transcript and send partial back
        sessions.get(sessionId).transcript += ' ' + transcription.text;
        ws.send(JSON.stringify({ type: 'partialTranscript', text: transcription.text }));
      }

      if (payload.type === 'end') {
        // Finalize: call GPT for SOAP summary
        const fullText = sessions.get(sessionId).transcript.trim();

        const prompt = `
You are a clinical note helper. Convert the given doctor-patient transcript into concise SOAP sections and redFlags.
Return ONLY JSON:
{"subjective":[],"objective":[],"assessment":[],"plan":[],"redFlags":[]}

Transcript:
${fullText}
        `;

        const completion = await openai.responses.create({
          model: "gpt-4o-mini",
          input: prompt,
          temperature: 0.2
        });

        let summary;
        try {
          summary = JSON.parse(completion.output[0].content[0].text);
        } catch (e) {
          summary = { error: "Invalid JSON from model", raw: completion.output[0].content[0].text };
        }

        ws.send(JSON.stringify({ type: 'final', transcript: fullText, summary }));
        ws.close();
        sessions.delete(sessionId);
      }

    } catch (err) {
      console.error('WS error:', err);
      ws.send(JSON.stringify({ type: 'error', message: err.message }));
    }
  });
});

/**
 * 2) SIMPLE HTTP fallback (if WS is hard in FlutterFlow)
 * - Upload entire audio file -> get transcript & summary once
 */
const upload = multer({ limits: { fileSize: 40 * 1024 * 1024 } }); // 40MB
app.post('/one-shot', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No audio' });

    const transcription = await openai.audio.transcriptions.create({
      file: new File([req.file.buffer], "audio.wav", { type: "audio/wav" }),
      model: "whisper-1"
    });

    const prompt = `
You are a clinical note helper. Convert the given doctor-patient transcript into concise SOAP sections and redFlags.
Return ONLY JSON:
{"subjective":[],"objective":[],"assessment":[],"plan":[],"redFlags":[]}

Transcript:
${transcription.text}
    `;

    const completion = await openai.responses.create({
      model: "gpt-4o-mini",
      input: prompt,
      temperature: 0.2
    });

    let summary;
    try {
      summary = JSON.parse(completion.output[0].content[0].text);
    } catch {
      summary = { error: "Invalid JSON", raw: completion.output[0].content[0].text };
    }

    res.json({
      transcript: transcription.text,
      summary
    });

  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

const port = process.env.PORT || 10000;
app.listen(port, () => console.log('Server live on ' + port));
