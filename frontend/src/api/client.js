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
export async function chatAudio(audioBlob, sessionId, llmProvider, llmApiKey) {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');
  formData.append('session_id', sessionId);
  formData.append('llm_provider', llmProvider);
  formData.append('llm_api_key', llmApiKey);

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
  formData.append('llm_api_key', llmApiKey);

  const response = await api.post('/chat/text-video', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

/**
 * Send an audio + video chat message.
 */
export async function chatAudioVideo(audioBlob, videoBlob, sessionId, llmProvider, llmApiKey) {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');
  formData.append('video', videoBlob, 'video.webm');
  formData.append('session_id', sessionId);
  formData.append('llm_provider', llmProvider);
  formData.append('llm_api_key', llmApiKey);

  const response = await api.post('/chat/audio-video', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
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
 * Health check.
 */
export async function healthCheck() {
  const response = await api.get('/health');
  return response.data;
}

export default api;
