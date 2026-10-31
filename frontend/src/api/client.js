import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
});

/**
 * Send a text-only chat message.
 */
export async function chatText(text, sessionId, llmProvider, llmApiKey) {
  const response = await api.post('/chat/text', {
    text,
    session_id: sessionId,
    llm_provider: llmProvider,
    llm_api_key: llmApiKey,
  });
  return response.data;
}

/**
 * Send an audio-only chat message.
 */
export async function chatAudio(audioBlob, sessionId, llmProvider, llmApiKey, sttProvider, groqApiKey) {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');
  formData.append('session_id', sessionId);
  formData.append('llm_provider', llmProvider);
  if (llmApiKey) formData.append('llm_api_key', llmApiKey);
  if (sttProvider) formData.append('stt_provider', sttProvider);
  if (groqApiKey) formData.append('groq_api_key', groqApiKey);

  const response = await api.post('/chat/audio', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

/**
 * Send a text + video chat message.
 */
export async function chatTextVideo(text, videoBlob, sessionId, llmProvider, llmApiKey) {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('video', videoBlob, 'video.webm');
  formData.append('session_id', sessionId);
  formData.append('llm_provider', llmProvider);
  if (llmApiKey) formData.append('llm_api_key', llmApiKey);

  const response = await api.post('/chat/text-video', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

/**
 * Send an audio + video chat message.
 */
export async function chatAudioVideo(audioBlob, videoBlob, sessionId, llmProvider, llmApiKey, sttProvider, groqApiKey) {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');
  formData.append('video', videoBlob, 'video.webm');
  formData.append('session_id', sessionId);
  formData.append('llm_provider', llmProvider);
  if (llmApiKey) formData.append('llm_api_key', llmApiKey);
  if (sttProvider) formData.append('stt_provider', sttProvider);
  if (groqApiKey) formData.append('groq_api_key', groqApiKey);

  const response = await api.post('/chat/audio-video', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

/**
 * Visual Chat endpoint — multimodal emotion analysis + LLM.
 *
 * Sends audio + optional webcam frames to POST /chat/visual.
 * Returns VisualChatResponse with the emotion signals, the fused read and
 * emotion(sentence) — or `no_speech: true` when the audio had no real speech.
 *
 * @param {Blob} audioBlob - Recorded audio (WebM/WAV).
 * @param {Blob|Blob[]|null} imageBlobs - Webcam frame(s) (JPEG/PNG), or null.
 * @param {string} sessionId
 * @param {string} llmProvider - 'groq' | 'openai' | 'gemini'
 * @param {string|null} llmApiKey
 * @param {string|null} sttProvider - 'groq' | 'whisper'
 * @param {string|null} groqApiKey
 */
export async function chatVisual(audioBlob, imageBlobs, sessionId, llmProvider, llmApiKey, sttProvider, groqApiKey, userText = null) {
  const formData = new FormData();
  if (audioBlob) {
    formData.append('audio', audioBlob, 'recording.webm');
  }
  const frames = (Array.isArray(imageBlobs) ? imageBlobs : [imageBlobs]).filter(Boolean);
  frames.forEach((blob, i) => {
    formData.append('images', blob, `frame_${i}.jpg`);
  });
  if (userText) {
    formData.append('user_text', userText);
  }
  formData.append('session_id', sessionId);
  formData.append('llm_provider', llmProvider);
  if (llmApiKey) formData.append('llm_api_key', llmApiKey);
  if (sttProvider) formData.append('stt_provider', sttProvider);
  if (groqApiKey) formData.append('groq_api_key', groqApiKey);

  const response = await api.post('/chat/visual', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 90000, // 90 s — allow for model loading + LLM
  });
  return response.data;
}

/**
 * Synthesize speech from text and emotion.
 * Returns { audio_url: string | null, error: string | null }
 */
export async function synthesizeSpeech(text, emotion, voiceParams) {
  const response = await api.post('/tts', {
    text,
    emotion,
    voice_params: voiceParams,
  });
  return response.data;
}

/**
 * Transcribe audio using the selected STT provider.
 * Returns { text: string, provider: string, error?: string }
 */
export async function transcribeAudio(audioBlob, provider = 'whisper', groqApiKey = '') {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');
  formData.append('provider', provider);
  if (groqApiKey) formData.append('groq_api_key', groqApiKey);

  const response = await api.post('/transcribe', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

/**
 * Health check.
 */
export async function healthCheck() {
  const response = await api.get('/health');
  return response.data;
}

/**
 * Detect face emotion from an image Blob/File.
 */
export async function detectFaceEmotion(imageBlobOrFile) {
  const formData = new FormData();
  formData.append('image', imageBlobOrFile, 'face.jpg');
  const response = await api.post('/face/detect', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

/**
 * Detect face emotion from a Base64-encoded image string.
 */
export async function detectFaceEmotionBase64(imageBase64) {
  const response = await api.post('/face/detect-base64', {
    image_base64: imageBase64,
  });
  return response.data;
}

/**
 * Log call completion session summary with turns and duration.
 */
export async function logCallComplete(sessionId, turns, durationSeconds) {
  try {
    const response = await api.post('/chat/call-complete', {
      session_id: sessionId,
      turns,
      duration_seconds: durationSeconds,
    });
    return response.data;
  } catch (err) {
    console.warn('[VisualChat] Failed to notify backend of call completion:', err);
    return null;
  }
}

export default api;

