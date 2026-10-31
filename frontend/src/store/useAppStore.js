import { create } from 'zustand';
import { v4 as uuidv4 } from 'uuid';

// Load saved chat messages from localStorage
const getSavedMessages = () => {
  try {
    const saved = localStorage.getItem('chat_messages');
    return saved ? JSON.parse(saved) : [];
  } catch (e) {
    console.error('Failed to parse chat messages from localStorage:', e);
    return [];
  }
};

// Keep the session id across reloads so it matches the persisted chat history
// (the backend keeps conversation memory per session id).
const getSavedSessionId = () => {
  try {
    const saved = localStorage.getItem('session_id');
    if (saved) return saved;
    const fresh = uuidv4();
    localStorage.setItem('session_id', fresh);
    return fresh;
  } catch {
    return uuidv4();
  }
};

const useAppStore = create((set, get) => ({
  /* ── Input / Output Mode ── */
  inputMode: 'text', // 'text' | 'audio'
  videoEnabled: false,
  outputModes: {
    text: true,
    voice: false,
    avatar: true,
  },

  /* ── Messages ── */
  messages: getSavedMessages(),
  // message shape:
  // {
  //   id: string,
  //   role: 'user' | 'assistant',
  //   content: string,
  //   emotion: string,
  //   emotionState: object | null,
  //   segments: array | null,      // response segments from the API
  //   avatarCommands: object | null,
  //   voiceParams: object | null,
  //   timestamp: number,
  // }

  /* ── Recording / Processing State ── */
  isRecording: false,
  isProcessing: false,
  isSpeaking: false,

  /* ── Emotion / Avatar ── */
  currentEmotion: 'neutral',
  currentAvatarCommands: null,
  selectedAvatar: (() => {
    try {
      const saved = localStorage.getItem('selected_avatar');
      return saved === 'sky2' ? 'sky2' : 'sky';
    } catch {
      return 'sky';
    }
  })(), // 'sky' | 'sky2'

  /* ── Multimodal Emotion Signals (4-modality, Visual Chat) ── */
  multimodalEmotions: null,
  // shape: {
  //   text:  { emotion, confidence, available }
  //   audio: { emotion, confidence, available }
  //   face:  { emotion, confidence, available, face_detected }
  //   dsmr:  { emotion, confidence }
  // }

  /* ── Session ── */
  sessionId: getSavedSessionId(),

  /* ── LLM Configuration ── */
  llmProvider: 'groq',
  llmApiKey: '',

  /* ── STT Configuration ── */
  sttProvider: localStorage.getItem('stt_provider') || 'groq', // 'groq' (default) | 'whisper'
  groqApiKey: localStorage.getItem('groq_api_key') || '',

  /* ── UI State ── */
  settingsPanelOpen: false,
  isVisualChatActive: false,

  /* ── Actions ── */
  setInputMode: (mode) =>
    set({ inputMode: mode }),

  toggleVideo: () =>
    set((state) => ({ videoEnabled: !state.videoEnabled })),

  setOutputMode: (key, value) =>
    set((state) => ({
      outputModes: { ...state.outputModes, [key]: value },
    })),

  addMessage: (msg) =>
    set((state) => {
      const newMessages = [
        ...state.messages,
        {
          id: msg.id || uuidv4(),
          role: msg.role,
          content: msg.content || '',
          audioUrl: msg.audioUrl || null,
          emotion: msg.emotion || 'neutral',
          userEmotion: msg.userEmotion || null,
          userEmotionScore: msg.userEmotionScore !== undefined ? msg.userEmotionScore : null,
          emotionState: msg.emotionState || null,
          segments: msg.segments || null,
          avatarCommands: msg.avatarCommands || null,
          voiceParams: msg.voiceParams || null,
          timestamp: msg.timestamp || Date.now(),
        },
      ];
      localStorage.setItem('chat_messages', JSON.stringify(newMessages));
      return { messages: newMessages };
    }),

  updateMessage: (id, updates) =>
    set((state) => {
      const newMessages = state.messages.map((msg) =>
        msg.id === id ? { ...msg, ...updates } : msg
      );
      localStorage.setItem('chat_messages', JSON.stringify(newMessages));
      return { messages: newMessages };
    }),

  setIsRecording: (bool) =>
    set({ isRecording: bool }),

  setIsProcessing: (bool) =>
    set({ isProcessing: bool }),

  setIsSpeaking: (bool) =>
    set({ isSpeaking: bool }),

  setCurrentEmotion: (emotion) =>
    set({ currentEmotion: emotion }),

  setCurrentAvatarCommands: (cmds) =>
    set({ currentAvatarCommands: cmds }),

  setSelectedAvatar: (avatar) => {
    const valid = avatar === 'sky2' ? 'sky2' : 'sky';
    try {
      localStorage.setItem('selected_avatar', valid);
    } catch {}
    set({ selectedAvatar: valid });
  },

  setMultimodalEmotions: (emotions) =>
    set({ multimodalEmotions: emotions }),

  setLlmProvider: (provider) =>
    set({ llmProvider: provider }),

  setLlmApiKey: (key) =>
    set({ llmApiKey: key }),

  setSttProvider: (provider) => {
    localStorage.setItem('stt_provider', provider);
    set({ sttProvider: provider });
  },

  setGroqApiKey: (key) => {
    localStorage.setItem('groq_api_key', key);
    set({ groqApiKey: key });
  },

  setVisualChatActive: (active) =>
    set({ isVisualChatActive: active }),

  toggleSettingsPanel: () =>
    set((state) => ({ settingsPanelOpen: !state.settingsPanelOpen })),

  resetSession: () => {
    localStorage.removeItem('chat_messages');
    const sessionId = uuidv4();
    try { localStorage.setItem('session_id', sessionId); } catch {}
    set({
      messages: [],
      sessionId,
      multimodalEmotions: null,
      currentEmotion: 'neutral',
      currentAvatarCommands: null,
      isRecording: false,
      isProcessing: false,
    });
  },
}));

export default useAppStore;
