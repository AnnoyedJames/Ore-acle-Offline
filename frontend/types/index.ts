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
  model: 'qwen3-4b',
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  search_mode: 'hybrid',
  thinking: true,
};
