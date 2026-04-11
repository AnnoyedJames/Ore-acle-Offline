export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  images?: ImageResult[];
  timestamp: Date;
}

export interface Citation {
  id: number;
  page_title: string;
  page_url: string;
  section: string;
  cited_text: string;
}

export interface ImageResult {
  url: string;
  alt_text: string;
  section: string;
  caption?: string;
  page_title: string;
}

export interface ConversationState {
  messages: Message[];
  isLoading: boolean;
  conversationId?: string;
}

export interface LLMSettings {
  model: string;
  temperature: number;
  top_p: number;
  max_tokens: number;
  search_mode: 'semantic' | 'keyword' | 'hybrid';
  thinking: boolean;
}

export const DEFAULT_LLM_SETTINGS: LLMSettings = {
  model: 'gemini-flash-lite',
  temperature: 0.3,
  top_p: 0.95,
  max_tokens: 1024,
  search_mode: 'hybrid',
  thinking: true,
};
