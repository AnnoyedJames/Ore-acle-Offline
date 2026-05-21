import { useState, useEffect, useCallback } from 'react';
import { Message, Citation, ImageResult } from '@/types';
import SourceCard from './SourceCard';
import ImageGallery from './ImageGallery';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { expandSemanticTags, parseMinecraftFormatting } from '@/lib/minecraft-colors';
import { ChevronDown, BrainCircuit } from 'lucide-react';
import CraftingGrid from './CraftingGrid';

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

function MarkdownImage({ message, node, src, alt, ...props }: any) {
  const [error, setError] = useState(false);

  if (error) return null;

  // 1. Internal image:N reference
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
              loading="lazy"
              title={imgData.caption || imgData.alt_text}
              onError={() => setError(true)}
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
    return null;
  }

  // 2. Resolve URL: if it's already a local /api/image/ path, use it.
  //    Otherwise try to match against message.images by alt text or
  //    filename so external wiki URLs fall back to local files.
  let resolvedSrc = src || '';
  if (resolvedSrc && !resolvedSrc.startsWith('/api/image/') && message.images?.length) {
    const match = message.images.find((img: any) => {
      if (alt && (img.alt_text === alt || img.caption === alt)) return true;
      // Match by filename stem in the URL
      if (src) {
        const srcFile = src.split('/').pop()?.split('?')[0]?.split('#')[0] || '';
        const imgFile = img.url.split('/').pop() || '';
        if (srcFile && imgFile && srcFile === imgFile) return true;
      }
      return false;
    });
    if (match) resolvedSrc = match.url;
  }

  return (
    <img
      src={resolvedSrc}
      alt={alt}
      loading="lazy"
      onError={() => setError(true)}
      className="rounded-lg max-w-full h-auto inline-block"
    />
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
    const thinkMatch = content.match(/^<think>([\s\S]*?)<\/think>\s*/i);
    const thinkText = thinkMatch ? thinkMatch[1] : null;
    const mainContent = thinkMatch ? content.slice(thinkMatch[0].length) : content;

    const processCraftingRecipes = (text: string) => {
      if (!text) return text;
      
      // 1. Process inline array-based recipes
      let processed = text.replace(
        /\[(?:Crafting Recipe|Crafting):\s*\[(.*?)\]\s*\[(.*?)\]\s*\[(.*?)\]\s*->\s*(.*?)\]/gi,
        (match, row1, row2, row3, result) => {
          const extract = (row: string) => row.split(',').map(s => s.trim().replace(/"/g, '&quot;'));
          const r1 = extract(row1);
          const r2 = extract(row2);
          const r3 = extract(row3);
          const res = result?.trim().replace(/"/g, '&quot;') || '';
          const s0 = r1[0] || ''; const s1 = r1[1] || ''; const s2 = r1[2] || '';
          const s3 = r2[0] || ''; const s4 = r2[1] || ''; const s5 = r2[2] || '';
          const s6 = r3[0] || ''; const s7 = r3[1] || ''; const s8 = r3[2] || '';
          return `<crafting-grid data-s0="${s0}" data-s1="${s1}" data-s2="${s2}" data-s3="${s3}" data-s4="${s4}" data-s5="${s5}" data-s6="${s6}" data-s7="${s7}" data-s8="${s8}" data-result="${res}"></crafting-grid>`;
        }
      );

      // 2. Process Markdown table-based recipes
      // Allows for formatting like **Result:** Enchanting Table
      const tablePattern = /\|\s*(?:Crafting Grid|Crafting recipe|Ingredients)\s*\|[^\|\n]*\|[^\|\n]*\|\s*\n\|\s*[-:]+\s*\|\s*[-:]+\s*\|\s*[-:]+\s*\|\s*\n\|\s*([^\|\n]+)\s*\|\s*([^\|\n]+)\s*\|\s*([^\|\n]+)\s*\|\s*\n\|\s*([^\|\n]+)\s*\|\s*([^\|\n]+)\s*\|\s*([^\|\n]+)\s*\|\s*\n\|\s*([^\|\n]+)\s*\|\s*([^\|\n]+)\s*\|\s*([^\|\n]+)\s*\|[\s\S]{0,150}?(?:Result|Output|Item|Crafting Result)(?:[^A-Za-z0-9_]|<[^>]*>)*([A-Za-z0-9_ '"-]+)/gi;
      
      processed = processed.replace(tablePattern, (match, s0, s1, s2, s3, s4, s5, s6, s7, s8, res) => {
        const clean = (s: string) => s ? s.trim().replace(/"/g, '&quot;').replace(/\./g, '') : '';
        const resultItem = clean(res) || 'Crafting Result';
        return `\n\n<crafting-grid data-s0="${clean(s0)}" data-s1="${clean(s1)}" data-s2="${clean(s2)}" data-s3="${clean(s3)}" data-s4="${clean(s4)}" data-s5="${clean(s5)}" data-s6="${clean(s6)}" data-s7="${clean(s7)}" data-s8="${clean(s8)}" data-result="${resultItem}"></crafting-grid>\n\n`;
      });
      
      return processed;
    };

    const formattedContent = isUser ? mainContent : parseMinecraftFormatting(expandSemanticTags(mainContent));
    const processedContent = processCraftingRecipes(formattedContent);

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
            p: ({ node, children, ...props }) => {
              const hasBlockImage = Array.isArray(children) 
                ? children.some((child: any) => child?.type === 'figure' || child?.props?.node?.tagName === 'img')
                : (children as any)?.type === 'figure' || (children as any)?.props?.node?.tagName === 'img';
                
              if (hasBlockImage) {
                return <div {...props} className="mb-4 last:mb-0 w-full">{processChildren(children)}</div>;
              }
              return <p {...props} className="mb-2 last:mb-0 break-words">{processChildren(children)}</p>;
            },
            li: ({ node, children, ...props }) => (
              <li {...props}>{processChildren(children)}</li>
            ),
            td: ({ node, children, ...props }) => (
              <td {...props}>{processChildren(children)}</td>
            ),
            th: ({ node, children, ...props }) => (
              <th {...props}>{processChildren(children)}</th>
            ),
            table: ({ node, children, ...props }) => {
              const tableChildren = (Array.isArray(children) ? children : [children]).filter((c: any) => c && typeof c !== 'string' || (typeof c === 'string' && c.trim() !== ''));
              const thead = tableChildren.find((c: any) => c.type === 'thead' || c.props?.node?.tagName === 'thead');
              const tbody = tableChildren.find((c: any) => c.type === 'tbody' || c.props?.node?.tagName === 'tbody');
              
              if (thead && tbody) {
                const headRow = Array.isArray(thead.props.children) ? thead.props.children[0] : thead.props.children;
                if (headRow) {
                  const thCols = Array.isArray(headRow.props.children) ? headRow.props.children : [headRow.props.children];
                  const extractText = (node: any): string => {
                    if (!node) return '';
                    if (typeof node === 'string') return node;
                    if (Array.isArray(node)) return node.map(extractText).join('');
                    if (node.props && node.props.children) return extractText(node.props.children);
                    return '';
                  };
                  const firstColText = extractText(thCols[0]);
                  
                  if (firstColText.includes('Crafting Grid') || firstColText.includes('Crafting recipe') || firstColText.includes('Ingredients')) {
                    const rows = (Array.isArray(tbody.props.children) ? tbody.props.children : [tbody.props.children])
                      .filter((c: any) => c && typeof c !== 'string' || (typeof c === 'string' && c.trim() !== ''));
                    
                    if (rows.length === 3) {
                      const gridCols: string[][] = [];
                      let is3x3 = true;
                      for (const row of rows) {
                        const cols = (Array.isArray(row.props.children) ? row.props.children : [row.props.children])
                          .filter((c: any) => c && typeof c !== 'string' || (typeof c === 'string' && c.trim() !== ''));
                        if (cols.length !== 3) { is3x3 = false; break; }
                        gridCols.push(cols.map((col: any) => {
                          const val = extractText(col);
                            return val ? val.trim() : 'Empty';
                        }));
                      }
                      
                        if (is3x3) {
                          return (
                            <CraftingGrid 
                              s0={gridCols[0][0]} s1={gridCols[0][1]} s2={gridCols[0][2]}
                              s3={gridCols[1][0]} s4={gridCols[1][1]} s5={gridCols[1][2]}
                              s6={gridCols[2][0]} s7={gridCols[2][1]} s8={gridCols[2][2]}
                              result='Crafting Result'
                            />
                          );
                        }
                      }
                    }
                  }
                }
                
                return (
                  <div className="overflow-x-auto relative w-full mb-4">
                    <table {...props}>
                    {processChildren(children)}
                  </table>
                </div>
              );
            },
            img: (props) => <MarkdownImage message={message} {...props} />,              'crafting-grid': ({ node, ...props }: any) => {
                return (
                  <CraftingGrid 
                    s0={props['data-s0']} s1={props['data-s1']} s2={props['data-s2']} 
                    s3={props['data-s3']} s4={props['data-s4']} s5={props['data-s5']} 
                    s6={props['data-s6']} s7={props['data-s7']} s8={props['data-s8']} 
                    result={props['data-result'] || 'Result'} 
                  />
                );
              }
          } as any}
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
    // Match multiple citation formats:
    //   ([6] Armor materials) — decorated variant, swallow the label
    //   [1]            — canonical bracket format
    //   Source #4:     — cloud-style with colon
    //   (Source #4)    — parenthetical variant
    const regex = /(?:\(\s*\[(\d+)\][^)\n]{0,80}\)|\[(\d+)\]|Source\s*#(\d+)[:.)]?|\((?:Source\s*#)?(\d+)\))/g;
    const parts: JSX.Element[] = [];
    let last = 0;
    let m;
    let k = 0;

    while ((m = regex.exec(text)) !== null) {
      if (m.index > last) parts.push(<span key={k++}>{text.slice(last, m.index)}</span>);

      const num = parseInt(m[1] || m[2] || m[3] || m[4]);
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
        {isUser && message.uploadedImages && message.uploadedImages.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {message.uploadedImages.map((img, i) => (
              <img 
                key={i}
                src={img.base64}
                alt={img.filename}
                className="max-h-48 rounded-md border border-gray-300 dark:border-gray-600 shadow-sm object-contain"
              />
            ))}
          </div>
        )}

        <div className={`${isUser ? 'book-text' : 'text-gray-900 dark:text-gray-100'} break-words overflow-wrap-anywhere min-w-0 w-full`}>
          {renderContentWithCitations(message.content)}
        </div>

        {message.images && message.images.length > 0 && (
          <div className="mt-3">
            <ImageGallery images={message.images} />
          </div>
        )}
      </div>

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
