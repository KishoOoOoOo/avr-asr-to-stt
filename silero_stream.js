const { Transform } = require('stream');
const ort = require('onnxruntime-node');

class SileroVADStream extends Transform {
  constructor(options = {}) {
    super({
      writableObjectMode: false,
      readableObjectMode: true,
    });

    // --- Configuration ---
    this.options = {
      // Input sample rate (from client, e.g., 8kHz from Asterisk)
      inputSampleRate: options.inputSampleRate || 8000,
      // Silero VAD model sample rate (always 16kHz)
      sampleRate: 16000,
      // Frame size for VAD processing
      frameSize: options.frameSize || 512,
      // VAD thresholds and timing - OPTIMIZED FOR ARABIC
      threshold: options.threshold || 0.3, // Lower threshold for better sensitivity
      minSilenceDurationMs: options.minSilenceDurationMs || 600, // Longer silence for Arabic pauses
      speechPadMs: options.speechPadMs || 400, // More padding for Arabic words
      // ONNX model path
      modelPath: options.modelPath || './silero_vad.onnx',
      // ONNX execution provider
      provider: options.provider || 'cpu',
      ...options
    };

    // --- VAD State ---
    this.vadModel = null;
    this.state = {
      state: null, // ONNX model state tensor
      sr: null,    // Sample rate tensor
      inputBuffer: Buffer.alloc(0),
      resampledBuffer: Buffer.alloc(0), // Buffer for resampled audio
      speechBuffer: [], // Buffer for detected speech frames
      isSpeaking: false,
      silenceFramesCount: 0,
      speechStartReported: false,
      totalFramesProcessed: 0,
    };

    // Calculate timing parameters
    this.inputSamplesPerMs = this.options.inputSampleRate / 1000;
    this.outputSamplesPerMs = this.options.sampleRate / 1000;
    this.frameSizeInBytes = this.options.frameSize * 2; // 16-bit PCM
    this.inputFrameSizeInBytes = Math.round(this.options.frameSize * this.options.inputSampleRate / this.options.sampleRate) * 2;
    
    // Silence and padding calculations based on output sample rate
    this.minSilenceFrames = Math.ceil((this.options.minSilenceDurationMs * this.outputSamplesPerMs) / this.options.frameSize);
    this.speechPadFrames = Math.ceil((this.options.speechPadMs * this.outputSamplesPerMs) / this.options.frameSize);

    this.isInitialized = false;
    this._initialize();
  }

  async _initialize() {
    try {
      console.log(`Initializing Silero VAD Stream...`);
      console.log(` - Input: ${this.options.inputSampleRate}Hz -> VAD: ${this.options.sampleRate}Hz`);
      console.log(` - Frame Size: ${this.options.frameSize} samples`);
      console.log(` - Threshold: ${this.options.threshold}`);
      console.log(` - Min Silence: ${this.options.minSilenceDurationMs}ms (${this.minSilenceFrames} frames)`);
      console.log(` - Speech Padding: ${this.options.speechPadMs}ms (${this.speechPadFrames} frames)`);

      this.vadModel = await ort.InferenceSession.create(this.options.modelPath, {
         executionProviders: [this.options.provider],
      });
      console.log('✓ ONNX VAD model loaded successfully');

      // Initialize state tensors
      const stateShape = [2, 1, 128];
      const stateSize = stateShape.reduce((a, b) => a * b, 1);
      this.state.state = new ort.Tensor('float32', new Float32Array(stateSize).fill(0), stateShape);
      this.state.sr = new ort.Tensor('int64', [BigInt(this.options.sampleRate)], [1]);

      console.log('✓ VAD state tensors initialized');
      this.isInitialized = true;
      this.emit('initialized');

    } catch (error) {
      console.error('✗ Failed to initialize Silero VAD:', error);
      this.emit('error', new Error(`VAD initialization failed: ${error.message}`));
    }
  }

  /**
   * Simple linear resampling from input rate to 16kHz
   */
  _resampleAudio(inputBuffer) {
    if (this.options.inputSampleRate === this.options.sampleRate) {
      return inputBuffer;
    }

    const inputSamples = [];
    for (let i = 0; i < inputBuffer.length; i += 2) {
      inputSamples.push(inputBuffer.readInt16LE(i));
    }

    const resampleRatio = this.options.sampleRate / this.options.inputSampleRate;
    const outputLength = Math.round(inputSamples.length * resampleRatio);
    const outputSamples = [];

    for (let i = 0; i < outputLength; i++) {
      const sourceIndex = i / resampleRatio;
      const leftIndex = Math.floor(sourceIndex);
      const rightIndex = Math.min(leftIndex + 1, inputSamples.length - 1);
      const fraction = sourceIndex - leftIndex;

      // Linear interpolation
      const interpolated = inputSamples[leftIndex] * (1 - fraction) + 
                          inputSamples[rightIndex] * fraction;
      outputSamples.push(Math.round(interpolated));
    }

    // Convert back to buffer
    const outputBuffer = Buffer.allocUnsafe(outputSamples.length * 2);
    for (let i = 0; i < outputSamples.length; i++) {
      outputBuffer.writeInt16LE(outputSamples[i], i * 2);
    }

    return outputBuffer;
  }

  async _transform(chunk, encoding, callback) {
    if (!this.isInitialized) {
      this.once('initialized', () => this._processChunk(chunk, callback));
      this.once('error', (err) => callback(err));
      return;
    }
    this._processChunk(chunk, callback);
  }

  async _processChunk(chunk, callback) {
    try {
      // Add to input buffer
      this.state.inputBuffer = Buffer.concat([this.state.inputBuffer, chunk]);

      // Resample the input buffer
      if (this.state.inputBuffer.length >= this.inputFrameSizeInBytes) {
        const toResample = this.state.inputBuffer.slice(0, this.inputFrameSizeInBytes);
        this.state.inputBuffer = this.state.inputBuffer.slice(this.inputFrameSizeInBytes);
        
        const resampled = this._resampleAudio(toResample);
        this.state.resampledBuffer = Buffer.concat([this.state.resampledBuffer, resampled]);
      }

      // Process resampled frames
      while (this.state.resampledBuffer.length >= this.frameSizeInBytes) {
        const audioFramePCM = this.state.resampledBuffer.slice(0, this.frameSizeInBytes);
        this.state.resampledBuffer = this.state.resampledBuffer.slice(this.frameSizeInBytes);

        // Convert to Float32Array for VAD processing
        const audioFrameFloat32 = this._bufferToFloat32(audioFramePCM);

        // Run VAD inference
        const inputs = {
          input: new ort.Tensor('float32', audioFrameFloat32, [1, this.options.frameSize]),
          state: this.state.state,
          sr: this.state.sr
        };

        const outputs = await this.vadModel.run(inputs);
        const probability = outputs.output.data[0];
        const newState = outputs.stateN;

        // Update state
        this.state.state = newState;
        this.state.totalFramesProcessed++;

        // Handle speech detection logic
        this._handleSpeechLogic(probability, audioFramePCM);
      }
      
      callback();
    } catch (error) {
      console.error('Error processing audio chunk:', error);
      callback(error);
    }
  }

  _handleSpeechLogic(probability, audioFramePCM) {
    const isSpeech = probability >= this.options.threshold;

    if (isSpeech) {
      this.state.silenceFramesCount = 0;
      
      if (!this.state.isSpeaking) {
        // Speech start detected
        this.state.isSpeaking = true;
        this.state.speechStartReported = false;
        
        console.log(`  VAD: Speech START (prob: ${probability.toFixed(3)}, frame: ${this.state.totalFramesProcessed})`);
        
        // Keep existing buffer content for pre-speech padding
        this.state.speechBuffer.push(audioFramePCM);
        
        if (!this.state.speechStartReported) {
          this.push({ speech: { start: true, probability: probability } });
          this.state.speechStartReported = true;
        }
      } else {
        // Continue speech
        this.state.speechBuffer.push(audioFramePCM);
        
        // Emit ongoing speech data
        this.push({ 
          speech: { state: true, probability: probability }, 
          audioData: audioFramePCM 
        });
      }
    } else {
      // Non-speech frame
      if (this.state.isSpeaking) {
        this.state.silenceFramesCount++;
        // Continue buffering during silence (for potential end padding)
        this.state.speechBuffer.push(audioFramePCM);

        if (this.state.silenceFramesCount >= this.minSilenceFrames) {
          // Speech end detected
          console.log(`  VAD: Speech END (prob: ${probability.toFixed(3)}, silence: ${this.state.silenceFramesCount} frames)`);
          
          this.state.isSpeaking = false;
          this.state.speechStartReported = false;
          this.state.silenceFramesCount = 0;

          // Combine all buffered speech data
          const speechAudioData = Buffer.concat(this.state.speechBuffer);
          this.state.speechBuffer = [];

          // Emit end event with complete audio segment
          this.push({ 
            speech: { end: true, probability: probability }, 
            audioData: speechAudioData 
          });
        }
      } else {
        // Silence continues - maintain a rolling buffer for pre-speech padding
        this.state.speechBuffer.push(audioFramePCM);
        
        // Keep only the padding frames to avoid memory buildup
        if (this.state.speechBuffer.length > this.speechPadFrames) {
          this.state.speechBuffer = this.state.speechBuffer.slice(-this.speechPadFrames);
        }
      }
    }
  }

  _bufferToFloat32(buffer) {
    const float32Array = new Float32Array(buffer.length / 2);
    for (let i = 0; i < float32Array.length; i++) {
      // Normalize to [-1, 1] range
      float32Array[i] = buffer.readInt16LE(i * 2) / 32768.0;
    }
    return float32Array;
  }

  _flush(callback) {
    console.log("VAD Stream: Flushing...");
    
    if (this.state.isSpeaking && this.state.speechBuffer.length > 0) {
      console.log("VAD Stream: Forcing end event for remaining speech");
      const speechAudioData = Buffer.concat(this.state.speechBuffer);
      this.push({ 
        speech: { end: true, probability: 0.0 }, 
        audioData: speechAudioData 
      });
      this.state.isSpeaking = false;
      this.state.speechBuffer = [];
    }
    
    console.log(`VAD Stream: Processed ${this.state.totalFramesProcessed} frames total`);
    callback();
  }
}

module.exports = SileroVADStream;
