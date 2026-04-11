'use client';

import { useState, useRef, useEffect } from 'react';
import { Message, LLMSettings, DEFAULT_LLM_SETTINGS } from '@/types';
import MessageBubble from './MessageBubble';
import LoadingSpinner from './LoadingSpinner';
import LLMSettingsPanel from './LLMSettingsPanel';
import { Send, SlidersHorizontal, RefreshCw } from 'lucide-react';

interface ChatInterfaceProps {
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  initialPrompt?: string;
}

export default function ChatInterface({ messages, setMessages, initialPrompt }: ChatInterfaceProps) {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [settings, setSettings] = useState<LLMSettings>(DEFAULT_LLM_SETTINGS);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const hasSentInitialPrompt = useRef(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-send initial prompt from landing page
  useEffect(() => {
    if (initialPrompt && !hasSentInitialPrompt.current) {
      hasSentInitialPrompt.current = true;
      sendMessage(initialPrompt);
    }
  }, [initialPrompt]);

  const sendMessage = async (text: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
    content: text,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Build conversation history for context (last 10 exchanges to stay within token limits)
      const currentMessages = [...messages, userMessage];
      const history = currentMessages
        .slice(-20) // last 10 pairs max
        .map(m => ({ role: m.role, content: m.content }));

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          history,
          model: settings.model,
          temperature: settings.temperature,
          top_p: settings.top_p,
          max_tokens: settings.max_tokens,
          search_mode: settings.search_mode,
          thinking: settings.thinking,
        }),
      });

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response ?? data.error ?? 'No response from backend. Is the FastAPI server running at localhost:8000?',
        citations: data.citations,
        // Deduplicate images by URL so the same image doesn't appear multiple times
        // when it appears in multiple retrieved chunks
        images: data.images
          ? Array.from(new Map((data.images as any[]).map(img => [img.url, img])).values())
          : undefined,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    sendMessage(input);
  };

  const handleRegenerate = async () => {
    if (isLoading || messages.length === 0) return;
    
    // Find the last user message
    let lastUserMessageIndex = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUserMessageIndex = i;
        break;
      }
    }
    
    if (lastUserMessageIndex === -1) return;
    
    const lastUserText = messages[lastUserMessageIndex].content;
    
    // Slice off the last pair to act as an overwrite run
    setMessages(prev => prev.slice(0, lastUserMessageIndex));
    await sendMessage(lastUserText);
  };

  return (
    <div className="flex h-full min-w-0 w-full">

      {/* Main chat column */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden px-2 sm:px-4 py-4 sm:py-6 min-w-0 w-full">
          {messages.map((message, index) => (
            <div key={message.id}>
              <MessageBubble message={message} />
              {!isLoading && message.role === 'assistant' && index === messages.length - 1 && (
                <div className="flex justify-center -mt-2 mb-4">
                  <button
                    onClick={handleRegenerate}
                    className="flex items-center gap-2 px-3 py-1.5 text-xs text-gray-500 hover:text-diamond-blue transition-colors rounded-full border border-gray-300/50 hover:border-diamond-blue/50"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    Regenerate response
                  </button>
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start mb-4">
              <div className="glass glass-light dark:glass-dark rounded-lg p-4">
                <LoadingSpinner />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-gray-300 dark:border-gray-700 p-2 sm:p-4 glass glass-light dark:glass-dark">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about Minecraft..."
              className="flex-1 px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white/50 dark:bg-black/50 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-diamond-blue"
              disabled={isLoading}
            />
            <button
              type="button"
              onClick={() => setSettingsOpen(o => !o)}
              className={`px-3 py-2 rounded-lg border transition-colors ${
                settingsOpen
                  ? 'bg-diamond-blue/20 border-diamond-blue text-diamond-blue'
                  : 'border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-400 hover:border-diamond-blue hover:text-diamond-blue'
              }`}
              title="LLM Settings"
            >
              <SlidersHorizontal className="w-4 h-4" />
            </button>
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-4 py-2 rounded-lg bg-diamond-blue hover:bg-diamond-blue/80 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold transition-colors flex items-center gap-2"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </div>

      {/* Settings Panel */}
      {settingsOpen && (
        <LLMSettingsPanel settings={settings} onChange={setSettings} />
      )}
    </div>
  );
}
