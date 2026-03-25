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
