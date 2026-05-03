import { useState, useRef, useEffect } from 'react';
import { Message, LLMSettings, DEFAULT_LLM_SETTINGS } from '@/types';
import MessageBubble from './MessageBubble';
import LoadingSpinner from './LoadingSpinner';
import LLMSettingsPanel from './LLMSettingsPanel';
import { Send, SlidersHorizontal, RefreshCw, Paperclip, X } from 'lucide-react';

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
  const [uploadedImages, setUploadedImages] = useState<{ filename: string; base64: string }[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const hasSentInitialPrompt = useRef(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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
      uploadedImages: uploadedImages.length > 0 ? [...uploadedImages] : undefined,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setUploadedImages([]);
    setIsLoading(true);

    const assistantId = (Date.now() + 1).toString();

    try {
      const currentMessages = [...messages, userMessage];
      const history = currentMessages
        .slice(-20)
        .map(m => ({ role: m.role, content: m.content }));

      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          history,
          images: userMessage.uploadedImages?.map(img => img.base64),
          model: settings.model,
          temperature: settings.temperature,
          top_p: settings.top_p,
          max_tokens: settings.max_tokens,
          search_mode: settings.search_mode,
          thinking: settings.thinking,
          reranker_key: settings.reranker_key || null,
          rerank_candidates: settings.rerank_candidates ?? null,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      // Create placeholder assistant message that will be updated as tokens arrive
      const placeholder: Message = {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, placeholder]);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let accContent = '';
      let citations: any[] | undefined;
      let images: any[] | undefined;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let eventType = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
          } else if (line.startsWith('data: ') && eventType) {
            const raw = line.slice(6);
            try {
              const parsed = JSON.parse(raw);
              if (eventType === 'citations') {
                citations = parsed.citations;
                images = parsed.images
                  ? Array.from(new Map((parsed.images as any[]).map((img: any) => [img.url, img])).values())
                  : undefined;
              } else if (eventType === 'token') {
                accContent += parsed;
                setMessages(prev =>
                  prev.map(m =>
                    m.id === assistantId
                      ? { ...m, content: accContent, citations, images }
                      : m,
                  ),
                );
              } else if (eventType === 'error') {
                accContent += `\n\n**Error:** ${parsed}`;
                setMessages(prev =>
                  prev.map(m =>
                    m.id === assistantId ? { ...m, content: accContent } : m,
                  ),
                );
              }
            } catch {
              // ignore malformed JSON lines
            }
            eventType = '';
          }
        }
      }

      // Final update with all metadata
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantId
            ? { ...m, content: accContent || 'No response from backend.', citations, images }
            : m,
        ),
      );
    } catch (error) {
      console.error('Error sending message:', error);
      // If placeholder was already added, update it; otherwise add new error msg
      setMessages(prev => {
        const hasPlaceholder = prev.some(m => m.id === assistantId);
        const errorMsg: Message = {
          id: assistantId,
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          timestamp: new Date(),
        };
        if (hasPlaceholder) {
          return prev.map(m => (m.id === assistantId ? errorMsg : m));
        }
        return [...prev, errorMsg];
      });
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
    
    let lastUserMessageIndex = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUserMessageIndex = i;
        break;
      }
    }
    
    if (lastUserMessageIndex === -1) return;
    
    const lastUserText = messages[lastUserMessageIndex].content;
    
    setMessages(prev => prev.slice(0, lastUserMessageIndex));
    await sendMessage(lastUserText);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    Array.from(files).forEach(file => {
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedImages(prev => [
          ...prev,
          { filename: file.name, base64: reader.result as string }
        ]);
      };
      reader.readAsDataURL(file);
    });
    
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const removeUploadedImage = (indexToRemove: number) => {
    setUploadedImages(prev => prev.filter((_, index) => index !== indexToRemove));
  };

  const isVisionModel = settings.model === 'gemini-flash-lite';

  return (
    <div className="flex h-full min-w-0 w-full">

      <div className="flex flex-col flex-1 min-w-0">
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

        <div className="border-t border-gray-300 dark:border-gray-700 p-2 sm:p-4 glass glass-light dark:glass-dark">
          
          {uploadedImages.length > 0 && (
            <div className="flex gap-2 mb-2 flex-wrap min-w-0">
              {uploadedImages.map((img, i) => (
                <div key={i} className="relative inline-block mt-2 ml-2">
                  <img 
                    src={img.base64} 
                    alt={img.filename} 
                    className="h-16 w-16 object-cover rounded-md border border-gray-300 dark:border-gray-600"
                  />
                  <button
                    onClick={() => removeUploadedImage(i)}
                    className="absolute -top-2 -right-2 bg-red-500 hover:bg-red-600 text-white rounded-full p-0.5 shadow-md"
                    title="Remove image"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <form onSubmit={handleSubmit} className="flex gap-2 items-center">
            
            <input
              type="file"
              accept="image/*"
              multiple
              ref={fileInputRef}
              className="hidden"
              onChange={handleImageUpload}
              disabled={isLoading || !isVisionModel}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading || !isVisionModel}
              title={isVisionModel ? "Upload picture for Gemini analysis" : "Current model does not support vision"}
              className={`p-2 rounded-lg border transition-colors flex-shrink-0 ${
                !isVisionModel
                  ? 'bg-gray-200 border-gray-200 text-gray-400 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-500 cursor-not-allowed opacity-50'
                  : 'bg-white dark:bg-black/50 border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-400 hover:text-diamond-blue hover:border-diamond-blue'
              }`}
            >
              <Paperclip className="w-4 h-4" />
            </button>

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

      {settingsOpen && (
        <LLMSettingsPanel settings={settings} onChange={setSettings} />
      )}
    </div>
  );
}
