'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { Message, Citation, ImageResult } from '@/types';
import SourceCard from './SourceCard';
import ImageGallery from './ImageGallery';
import ReactMarkdown, { Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { expandSemanticTags, parseMinecraftFormatting } from '@/lib/minecraft-colors';
import Image from 'next/image';
import { ChevronDown, BrainCircuit } from 'lucide-react';

// ---------------------------------------------------------------------------
// Thinking block — collapsible, styled like a dev panel
// ---------------------------------------------------------------------------
function ThinkingBlock({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mb-3 rounded-lg border border-diamond-blue/30 overflow-hidden text-xs">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-diamond-blue/10 hover:bg-diamond-blue/20 transition-colors text-diamond-blue font-medium"
      >
        <BrainCircuit className="w-3.5 h-3.5 shrink-0" />
        <span className="flex-1 text-left">Thinking process</span>
        <ChevronDown className={`w-3.5 h-3.5 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <pre className="px-3 py-2 text-gray-500 dark:text-gray-400 whitespace-pre-wrap font-mono leading-relaxed overflow-x-auto bg-black/5 dark:bg-white/5">
          {text.trim()}
        </pre>
      )}
    </div>
  );
}

interface MessageBubbleProps {
  message: Message;
}

interface TooltipState {
  citation: Citation;
  rect: DOMRect;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const [tappedOpen, setTappedOpen] = useState(false);
  const isUser = message.role === 'user';

  // Close on outside tap (mobile)
  useEffect(() => {
    if (!tappedOpen) return;
    const close = (e: MouseEvent | TouchEvent) => {
      if (!(e.target as HTMLElement).closest('.citation-trigger')) {
        setTooltip(null);
        setTappedOpen(false);
      }
    };
    document.addEventListener('mousedown', close);
    document.addEventListener('touchstart', close);
    return () => {
      document.removeEventListener('mousedown', close);
      document.removeEventListener('touchstart', close);
    };
  }, [tappedOpen]);

  const showTooltip = useCallback((citation: Citation, el: HTMLElement) => {
    setTooltip({ citation, rect: el.getBoundingClientRect() });
  }, []);

  const hideTooltip = useCallback(() => {
    if (!tappedOpen) setTooltip(null);
  }, [tappedOpen]);

  const toggleTooltip = useCallback((citation: Citation, el: HTMLElement) => {
    setTappedOpen(prev => {
      if (prev) {
        setTooltip(null);
        return false;
      }
      setTooltip({ citation, rect: el.getBoundingClientRect() });
      return true;
    });
  }, []);

  // Compute tooltip position relative to viewport
  const tooltipStyle = useCallback((): React.CSSProperties => {
    if (!tooltip) return {};
    const { rect } = tooltip;
    const openAbove = rect.top > 220;
    return {
      position: 'fixed' as const,
      left: Math.max(8, Math.min(rect.left, window.innerWidth - 420)),
      top: openAbove ? rect.top - 8 : rect.bottom + 8,
      transform: openAbove ? 'translateY(-100%)' : undefined,
      zIndex: 9999,
    };
  }, [tooltip]);

  const renderContentWithCitations = (content: string) => {
    // Extract <think>...</think> block (reasoning models like Qwen3)
    const thinkMatch = content.match(/^<think>([\s\S]*?)<\/think>\s*/i);
    const thinkText = thinkMatch ? thinkMatch[1] : null;
    const mainContent = thinkMatch ? content.slice(thinkMatch[0].length) : content;

    const processedContent = isUser ? mainContent : parseMinecraftFormatting(expandSemanticTags(mainContent));

    if (!message.citations || message.citations.length === 0) {
      return (
        <>
          {thinkText && !isUser && <ThinkingBlock text={thinkText} />}
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
            {processedContent}
          </ReactMarkdown>
        </>
      );
    }

    return (
      <>
        {thinkText && !isUser && <ThinkingBlock text={thinkText} />}
        <div className="prose prose-sm dark:prose-invert max-w-none min-w-0 w-full">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw]}
          urlTransform={(url) => {
            if (url.startsWith('image:')) return url;
            return url;
          }}
          components={{
            p: ({ children, ...props }) => {
              // Check if children contains a figure (our image component)
              // This is a naive check but works for react-markdown's structure
              const hasBlockImage = Array.isArray(children) 
                ? children.some((child: any) => child?.type === 'figure' || child?.props?.node?.tagName === 'img')
                : (children as any)?.type === 'figure' || (children as any)?.props?.node?.tagName === 'img';
                
              if (hasBlockImage) {
                return <div {...props} className="mb-4 last:mb-0 w-full">{processChildren(children)}</div>;
              }
              return <p {...props} className="mb-2 last:mb-0 break-words">{processChildren(children)}</p>;
            },
            li: ({ children, ...props }) => (
              <li {...props}>{processChildren(children)}</li>
            ),
            img: ({ src, alt, ...props }) => {
              if (src?.startsWith('image:')) {
                const id = parseInt(src.replace('image:', ''), 10);
                const imgData = message.images?.[id - 1];
                if (imgData) {
                  return (
                    <figure className="my-4 block w-full max-w-sm mx-auto">
                      <div className="relative aspect-[4/3] w-full bg-black/20 rounded-lg overflow-hidden border border-stone-300 dark:border-stone-700">
                        <img
                          src={imgData.url}
                          alt={alt || imgData.alt_text}
                          className="w-full h-full object-contain image-pixelated"
                          title={imgData.caption || imgData.alt_text}
                        />
                      </div>
                      {(imgData.caption || alt || imgData.alt_text) && (
                        <figcaption className="text-center text-xs text-stone-500 mt-2 italic px-2">
                          {imgData.caption || alt || imgData.alt_text}
                        </figcaption>
                      )}
                    </figure>
                  );
                }
                return null; // Don't render broken image if ID is invalid
              }
              // Normal markdown images
              return <img src={src} alt={alt} {...props} className="rounded-lg max-w-full h-auto" />;
            }
          }}
        >
          {processedContent}
        </ReactMarkdown>
      </div>
      </>
    );
  };

  const processChildren = (children: any): any => {
    if (typeof children === 'string') return injectCitations(children);
    if (Array.isArray(children)) {
      return children.map((child, i) =>
        typeof child === 'string' ? <span key={i}>{injectCitations(child)}</span> : child
      );
    }
    return children;
  };

  const injectCitations = (text: string) => {
    const regex = /\[(\d+)\]/g;
    const parts: JSX.Element[] = [];
    let last = 0;
    let m;
    let k = 0;

    while ((m = regex.exec(text)) !== null) {
      if (m.index > last) parts.push(<span key={k++}>{text.slice(last, m.index)}</span>);

      const num = parseInt(m[1]);
      const citation = message.citations?.find(c => c.id === num);

      if (citation) {
        const cit = citation;
        parts.push(
          <sup
            key={k++}
            className="citation-trigger cursor-pointer text-diamond-blue hover:text-diamond-blue/80 font-bold px-1 transition-colors"
            onMouseEnter={(e) => showTooltip(cit, e.currentTarget)}
            onMouseLeave={hideTooltip}
            onClick={(e) => { e.stopPropagation(); toggleTooltip(cit, e.currentTarget); }}
          >
            [{num}]
          </sup>
        );
      } else {
        parts.push(<span key={k++}>{m[0]}</span>);
      }
      last = m.index + m[0].length;
    }

    if (last < text.length) parts.push(<span key={k++}>{text.slice(last)}</span>);
    return parts.length > 0 ? parts : text;
  };

  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-4 min-w-0`}>
      <div
        className={`max-w-full sm:max-w-[80%] rounded-lg p-2 sm:p-4 overflow-hidden ${
          isUser
            ? 'book-theme dark:bg-diamond-blue/30 border-2 border-[#8B6914] dark:border-diamond-blue/50'
            : 'glass glass-light dark:glass-dark'
        }`}
      >
        <div className={`${isUser ? 'book-text' : 'text-gray-900 dark:text-gray-100'} break-words overflow-wrap-anywhere min-w-0 w-full`}>
          {renderContentWithCitations(message.content)}
        </div>

        {message.images && message.images.length > 0 && (
          <div className="mt-3">
            <ImageGallery images={message.images} />
          </div>
        )}
      </div>

      {/* Single tooltip — one per bubble, positioned via fixed coords */}
      {tooltip && (
        <div
          style={tooltipStyle()}
          onMouseEnter={() => setTooltip(tooltip)}
          onMouseLeave={hideTooltip}
        >
          <SourceCard citation={tooltip.citation} />
        </div>
      )}
    </div>
  );
}
