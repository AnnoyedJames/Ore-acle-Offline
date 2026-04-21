import { useState, useEffect, useRef } from 'react';
import ChatInterface from '@/components/ChatInterface';
import Header from '@/components/Header';
import { Pickaxe, Compass, Book, Sparkles, SquarePen } from 'lucide-react';
import type { Message } from '@/types';

const EXAMPLE_PROMPTS = [
  { icon: Pickaxe, text: 'Where do I find diamonds?' },
  { icon: Compass, text: 'How do I make a compass?' },
  { icon: Book, text: 'What are enchantments?' },
  { icon: Sparkles, text: 'How does the End portal work?' },
];

const SESSION_KEY = 'ore-acle-chat';
const HISTORY_KEY = 'ore-acle-chat-history';

export default function Home() {
  const [showChat, setShowChat] = useState(false);
  const [selectedPrompt, setSelectedPrompt] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [hydrated, setHydrated] = useState(false);
  const [sessions, setSessions] = useState<{id: string, title: string}[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const isFirstRender = useRef(true);

  // Restore session on mount
  useEffect(() => {
    try {
      const saved = sessionStorage.getItem(SESSION_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.messages?.length) {
          setMessages(
            parsed.messages.map((m: Message & { timestamp: string | Date }) => ({
              ...m,
              timestamp: new Date(m.timestamp),
            }))
          );
        }
        if (parsed.showChat) {
          setShowChat(true);
        }
      }
    } catch {
      // Ignore corrupt session
    }
    try {
      const hist = localStorage.getItem(HISTORY_KEY);
      if (hist) setSessions(JSON.parse(hist));
    } catch {}
    setHydrated(true);
  }, []);

  // Persist session whenever state changes (after hydration)
  useEffect(() => {
    if (!hydrated) return;
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    try {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify({ messages, showChat }));
    } catch {}

    if (messages.length > 0 && showChat) {
      try {
        const currentSess = sessions.find(s => s.id === 'current') || { id: 'current', title: messages[0].content.substring(0, 30) + '...' };
        const newSessions = [currentSess, ...sessions.filter(s => s.id !== 'current')];
        setSessions(newSessions);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(newSessions));
      } catch {}
    }
  }, [messages, showChat, hydrated]);

  const handlePromptClick = (prompt: string) => {
    setSelectedPrompt(prompt);
    setShowChat(true);
  };

  const handleNewChat = () => {
    setMessages([]);
    setSelectedPrompt(null);
    setShowChat(false);
    sessionStorage.removeItem(SESSION_KEY);
  };

  return (
    <div className="h-screen flex flex-col overflow-x-hidden">
      <Header />

      <div className="flex-1 flex overflow-hidden w-full">
        {/* Sidebar — always available */}
        <div className={`transition-all duration-300 ${sidebarOpen ? 'w-64 max-w-[60vw]' : 'w-0'} bg-black/40 border-r border-white/10 overflow-hidden flex flex-col`}>
          <div className="p-4 font-bold text-white whitespace-nowrap border-b border-white/10 shrink-0">Chat History</div>
          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {sessions.map(s => (
              <button key={s.id} className="w-full text-left px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-white text-sm truncate">
                {s.title}
              </button>
            ))}
            {sessions.length === 0 && <div className="text-gray-400 text-sm p-2 text-center">No history yet</div>}
          </div>
        </div>

        <div className="flex-1 flex flex-col relative w-full overflow-hidden">
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="absolute left-2 top-2 z-10 p-2 bg-black/50 rounded-lg text-white hover:bg-black/80 flex items-center justify-center transition-colors" title="Toggle History">
            <Book className="w-4 h-4" />
          </button>

          {showChat && (
            <button onClick={handleNewChat} className="absolute left-12 top-2 z-10 p-2 bg-black/50 rounded-lg text-white hover:bg-black/80 flex items-center justify-center transition-colors" title="New Chat">
              <SquarePen className="w-4 h-4" />
            </button>
          )}

          {!showChat ? (
            // Landing page
            <div className="flex-1 flex items-center justify-center p-4 sm:p-8">
              <div className="max-w-3xl w-full">
                <div className="text-center mb-8 sm:mb-12">
                  <h2 className="text-3xl sm:text-5xl md:text-6xl font-bold mb-2 sm:mb-4 text-white text-stroke-sm sm:text-stroke">
                    Ore-acle
                  </h2>
                  <p className="text-sm sm:text-lg md:text-xl text-white text-stroke-sm">
                    Your Minecraft wiki companion with intelligent search
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 sm:gap-4 mb-6 sm:mb-8">
                  {EXAMPLE_PROMPTS.map((prompt, idx) => {
                    const Icon = prompt.icon;
                    return (
                      <button
                        key={idx}
                        onClick={() => handlePromptClick(prompt.text)}
                        className="glass glass-light dark:glass-dark rounded-lg p-3 sm:p-4 text-left hover:border-diamond-blue dark:hover:border-diamond-blue transition-all hover:scale-105 flex items-center gap-2 sm:gap-3"
                      >
                        <Icon className="w-5 h-5 text-diamond-blue flex-shrink-0" />
                        <span className="text-sm sm:text-base text-gray-900 dark:text-gray-100">{prompt.text}</span>
                      </button>
                    );
                  })}
                </div>

                <div className="glass glass-light dark:glass-dark rounded-lg p-2">
                  <input
                    type="text"
                    placeholder="Ask anything about Minecraft..."
                    className="w-full px-3 sm:px-4 py-2 sm:py-3 bg-transparent text-sm sm:text-base text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none"
                    onFocus={() => setShowChat(true)}
                  />
                </div>
              </div>
            </div>
          ) : (
            // Chat interface
            <div className="flex-1 flex overflow-hidden min-w-0 w-full">
              <div className="flex-1 flex flex-col min-w-0 w-full">
                <ChatInterface
                  messages={messages}
                  setMessages={setMessages}
                  initialPrompt={selectedPrompt ?? undefined}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
