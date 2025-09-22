require('dotenv').config();
const fs = require("fs");
const path = require("path");
const express = require("express");
const axios = require('axios'); // To send data to the transcription service
const SileroVADStream = require('./silero_stream'); // Require from same directory

const app = express();

// --- Configuration ---
// Use environment variables with defaults - ADJUSTED FOR BETTER ARABIC RECOGNITION
const MIN_SPEECH_DURATION = parseInt(process.env.VAD_MIN_SPEECH_DURATION_MS || '800', 10); // Reduced minimum duration
const STT_URL = process.env.STT_URL || 'http://localhost:6022/transcribe'; // Updated default port

// Audio configuration (Input from client)
const INPUT_AUDIO_CONFIG = {
  sampleRate: 8000,
  channels: 1,
  bitsPerSample: 16,
};

// Path to the ONNX model (relative to this server.js file)
const MODEL_PATH = path.join(__dirname, 'silero_vad.onnx');
if (!fs.existsSync(MODEL_PATH)) {
    console.error(`\n!!! FATAL ERROR: ONNX model not found at ${MODEL_PATH}`);
    console.error(`Ensure 'silero_vad.onnx' is inside the 'vad_service' directory.`);
    process.exit(1);
}

/**
 * Simple upsampling function from 8kHz to 16kHz
 * @param {Buffer} inputBuffer - Input audio buffer at 8kHz
 * @returns {Buffer} Output audio buffer at 16kHz
 */
function upsampleTo16kHz(inputBuffer) {
  const inputSamples = [];
  
  // Convert buffer to samples
  for (let i = 0; i < inputBuffer.length; i += 2) {
    const sample = inputBuffer.readInt16LE(i);
    inputSamples.push(sample);
  }
  
  // Simple linear interpolation upsampling (2x)
  const outputSamples = [];
  
  for (let i = 0; i < inputSamples.length - 1; i++) {
    // Keep original sample
    outputSamples.push(inputSamples[i]);
    
    // Interpolate between current and next sample
    const interpolated = Math.round((inputSamples[i] + inputSamples[i + 1]) / 2);
    outputSamples.push(interpolated);
  }
  
  // Add last sample
  if (inputSamples.length > 0) {
    outputSamples.push(inputSamples[inputSamples.length - 1]);
    outputSamples.push(inputSamples[inputSamples.length - 1]); // Duplicate for even count
  }
  
  // Convert back to buffer
  const outputBuffer = Buffer.allocUnsafe(outputSamples.length * 2);
  for (let i = 0; i < outputSamples.length; i++) {
    outputBuffer.writeInt16LE(outputSamples[i], i * 2);
  }
  
  return outputBuffer;
}

// --- VAD Stream Handler ---
const handleAudioStream = async (req, res) => {
  let speechStartTime = null;
  let vadStream = null;
  let accumulatedSpeechBuffer = Buffer.alloc(0); // Track complete speech segments

  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  console.log(`\n[${new Date().toISOString()}] VAD Service: New connection`);

  try {
      // Optimized VAD parameters for Arabic speech
      const vadThreshold = parseFloat(process.env.VAD_THRESHOLD || '0.3'); // Lower threshold for better sensitivity
      const vadMinSilenceMs = parseInt(process.env.VAD_MIN_SILENCE_MS || '600', 10); // Longer silence detection
      const vadSpeechPadMs = parseInt(process.env.VAD_SPEECH_PAD_MS || '400', 10); // More padding
      const onnxProvider = process.env.ONNX_PROVIDER || 'cpu';

      vadStream = new SileroVADStream({
        inputSampleRate: INPUT_AUDIO_CONFIG.sampleRate, // 8kHz input
        modelPath: MODEL_PATH,
        threshold: vadThreshold,
        minSilenceDurationMs: vadMinSilenceMs,
        speechPadMs: vadSpeechPadMs,
        provider: onnxProvider,
        // Additional parameters for better Arabic detection
        frameSize: 512, // Standard frame size
        sampleRate: 16000 // VAD internal processing rate
      });

      const outputSampleRate = vadStream.options.sampleRate; // Should be 16000Hz
      console.log(`[VAD Service] VAD initialized. Input: ${INPUT_AUDIO_CONFIG.sampleRate}Hz, VAD Processing: ${outputSampleRate}Hz`);
      console.log(` - Threshold: ${vadThreshold}, Min Silence: ${vadMinSilenceMs}ms, Padding: ${vadSpeechPadMs}ms`);
      console.log(` - Provider: ${onnxProvider}, Min Speech Duration: ${MIN_SPEECH_DURATION}ms`);

      req.pipe(vadStream)
        .on('error', (err) => {
          console.error(`\n!!! VAD Service: SileroVADStream Error: ${err.message}`);
          if (!res.writableEnded) {
              res.status(500).write(`VAD Processing Error: ${err.message}\n`);
              res.end();
          }
        })
        .on("data", async ({ speech: speechEvent, audioData: chunk }) => {

          if (speechEvent.start) {
            console.log(`(${new Date().toISOString()}) VAD Service: Speech Start Detected`);
            speechStartTime = Date.now();
            accumulatedSpeechBuffer = Buffer.alloc(0); // Reset accumulated buffer
            
            // Start accumulating audio data
            if (chunk && chunk.length > 0) {
              accumulatedSpeechBuffer = Buffer.concat([accumulatedSpeechBuffer, chunk]);
            }
          }

          if (speechEvent.state && chunk) {
            // Continue accumulating speech data
            accumulatedSpeechBuffer = Buffer.concat([accumulatedSpeechBuffer, chunk]);
          }

          if (speechEvent.end) {
            const speechDuration = speechStartTime ? Date.now() - speechStartTime : 0;
            console.log(`(${new Date().toISOString()}) VAD Service: Speech End Detected - Duration: ${(speechDuration / 1000).toFixed(2)}s`);

            speechStartTime = null;

            // Use accumulated audio buffer or the chunk from end event
            const finalAudioData = chunk && chunk.length > 0 ? chunk : accumulatedSpeechBuffer;

            if (!finalAudioData || finalAudioData.length === 0) {
              console.log("[VAD Service] No audio data available for transcription, discarding.");
              accumulatedSpeechBuffer = Buffer.alloc(0);
              return;
            }

            // Check minimum duration
            if (speechDuration >= MIN_SPEECH_DURATION) {
              console.log(`[VAD Service] Processing speech segment: ${(finalAudioData.length / 1024).toFixed(2)} KB`);

              try {
                // The audio from VAD is at 16kHz, send it directly
                const audioToSend = finalAudioData;
                const sampleRateToSend = outputSampleRate; // 16000Hz

                console.log(`[VAD Service] Sending audio to STT: ${(audioToSend.length / 1024).toFixed(2)} KB at ${sampleRateToSend}Hz`);

                const response = await axios.post(STT_URL, audioToSend, {
                    headers: {
                        'Content-Type': 'application/octet-stream',
                        'X-Sample-Rate': sampleRateToSend,
                        'X-Audio-Format': 'audio/x-signed-linear'
                    },
                    maxBodyLength: Infinity,
                    maxContentLength: Infinity,
                    timeout: 30000 // 30 second timeout
                });

                console.log(`[VAD Service] STT Response Status: ${response.status}`);

                const transcription = response.data && response.data.transcription;

                if (transcription && transcription.trim() && !res.writableEnded) {
                    console.log(`[VAD Service] ✓ Transcription: "${transcription}"`);
                    res.write(transcription.trim() + "\n");
                } else {
                     console.log("[VAD Service] ✗ Empty or no transcription received");
                     // Optionally send feedback to client
                     // res.write("[No speech detected]\n");
                }

              } catch (err) {
                console.error(`[VAD Service] ✗ STT Service Error:`);
                if (err.response) {
                    console.error(` - HTTP Status: ${err.response.status}`);
                    console.error(` - Response: ${JSON.stringify(err.response.data)}`);
                } else if (err.request) {
                    console.error(` - Network Error: ${err.message}`);
                    console.error(` - STT URL: ${STT_URL}`);
                } else {
                    console.error(` - Setup Error: ${err.message}`);
                }

                if (!res.writableEnded) {
                    res.write("[Transcription service error]\n");
                }
              }
            } else {
              console.log(`[VAD Service] ✗ Speech too short: ${(speechDuration / 1000).toFixed(2)}s (min: ${MIN_SPEECH_DURATION/1000}s)`);
            }

            // Reset accumulated buffer
            accumulatedSpeechBuffer = Buffer.alloc(0);
          }
        })
        .on('finish', () => {
          console.log(`(${new Date().toISOString()}) VAD Service: Stream finished`);
        });

  } catch (initError) {
      console.error(`[VAD Service] Initialization failed: ${initError.message}`);
      if (!res.writableEnded) {
          res.status(500).write(`VAD Init Error: ${initError.message}\n`);
          res.end();
      }
      return;
  }

  req.on("end", () => {
    console.log(`(${new Date().toISOString()}) VAD Service: Client disconnected`);
    if (!res.writableEnded) {
        res.end();
    }
  });

  req.on("error", (err) => {
    console.error(`(${new Date().toISOString()}) VAD Service: Request error:`, err.message);
    if (vadStream && !vadStream.destroyed) {
        vadStream.destroy(err);
    }
    if (!res.headersSent) {
        res.status(500).json({ message: "Error receiving audio stream" });
    } else if (!res.writableEnded) {
        res.end();
    }
  });
};

// --- Route Configuration ---
app.post('/speech-to-text-stream', handleAudioStream);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    service: 'VAD Service',
    timestamp: new Date().toISOString(),
    config: {
      minSpeechDuration: MIN_SPEECH_DURATION,
      sttUrl: STT_URL,
      vadThreshold: process.env.VAD_THRESHOLD || '0.3'
    }
  });
});

// Start the VAD server
const VAD_PORT = process.env.PORT || 6019;
app.listen(VAD_PORT, () => {
  console.log(`\n=== VAD Service Started ===`);
  console.log(`Timestamp: ${new Date().toISOString()}`);
  console.log(`Listening on port ${VAD_PORT}`);
  console.log(`STT Service: ${STT_URL}`);
  console.log(`Min Speech Duration: ${MIN_SPEECH_DURATION}ms`);
  console.log(`Input: ${INPUT_AUDIO_CONFIG.sampleRate}Hz -> VAD: 16kHz`);
  console.log(`========================\n`);
});
