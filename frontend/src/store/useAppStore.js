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

  /* ── Session ── */
  sessionId: uuidv4(),

  /* ── LLM Configuration ── */
  llmProvider: 'openai',
  llmApiKey: '',

  /* ── UI State ── */
  settingsPanelOpen: false,

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

  setLlmProvider: (provider) =>
    set({ llmProvider: provider }),

  setLlmApiKey: (key) =>
    set({ llmApiKey: key }),

  toggleSettingsPanel: () =>
    set((state) => ({ settingsPanelOpen: !state.settingsPanelOpen })),

  resetSession: () => {
    localStorage.removeItem('chat_messages');
    set({
      messages: [],
      sessionId: uuidv4(),
      currentEmotion: 'neutral',
      currentAvatarCommands: null,
      isRecording: false,
      isProcessing: false,
    });
  },
}));

export default useAppStore;
