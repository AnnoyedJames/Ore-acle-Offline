'use client';

import Header from '@/components/Header';
import {
  Database,
  Search,
  Brain,
  Image as ImageIcon,
  FileText,
  Layers,
  ArrowRight,
  ExternalLink,
  Github,
  Globe,
  Cpu,
  HardDrive,
  Scissors,
  BookOpen,
  MessageSquare,
  AlertTriangle,
} from 'lucide-react';

/* ------------------------------------------------------------------ */
/*  Data & Types                                                       */
/* ------------------------------------------------------------------ */

interface PipelineStep {
  icon: React.ElementType;
  title: string;
  subtitle: string;
  description: string;
  color: string;
}

interface TechCard {
  name: string;
  role: string;
  description: string;
  tier: string;
  link: string;
  color: string;
}

interface Stat {
  value: string;
  label: string;
}

const PIPELINE_STEPS: PipelineStep[] = [
  {
    icon: Globe,
    title: 'Scraping',
    subtitle: 'wiki_scraper.py',
    description:
      '12,487 pages crawled from minecraft.wiki via sitemap. Rate-limited to 1 req/s with resume support and periodic state saves.',
    color: 'text-emerald-green',
  },
  {
    icon: ImageIcon,
    title: 'Image Download',
    subtitle: 'image_downloader.py',
    description:
      '61,248 images downloaded from the wiki CDN at 5 req/s. Content-addressed by MD5 hash, converted to WebP (quality 80, max 1280px).',
    color: 'text-ore-gold',
  },
  {
    icon: Scissors,
    title: 'Text Cleaning',
    subtitle: 'text_cleaner.py',
    description:
      'HTML → structured JSON. Strips navboxes, edit links, and references while preserving prose, tables (as Markdown), infoboxes (as dicts), and image context.',
    color: 'text-diamond-blue',
  },
  {
    icon: Layers,
    title: 'Chunking',
    subtitle: 'chunker.py',
    description:
      'Recursive text splitting at 512 tokens with 50-token overlap. Each chunk retains page title, URL, section heading, and verbatim text for NotebookLM-style citations.',
    color: 'text-redstone-red',
  },
  {
    icon: Brain,
    title: 'Embedding',
    subtitle: 'generator.py',
    description:
      'Multilingual-E5-Large generates 1024-dimensional vectors. Matched with Pinecone Inference API for serverless query embedding.',
    color: 'text-purple-400',
  },
  {
    icon: Database,
    title: 'Upload',
    subtitle: 'uploader.py + upload_images_r2.py',
    description:
      'Vectors → Pinecone, text/metadata → Supabase, images → Cloudflare R2. Dual-target upload with retry logic and checkpoint saves.',
    color: 'text-emerald-green',
  },
];

const TECH_STACK: TechCard[] = [
  {
    name: 'Pinecone',
    role: 'Vector Database & Inference',
    description:
      'Stores 1024d vectors and serves as the embedding provider via Pinecone Inference API (serverless embedding generation).',
    tier: 'Free (Serverless)',
    link: 'https://www.pinecone.io/',
    color: 'border-purple-400',
  },
  {
    name: 'Supabase',
    role: 'Keyword Search Index',
    description:
      'Stores tsvector keyword index with page metadata only (no chunk text). Returns matching IDs that are then hydrated from Pinecone. Keeps DB under 200 MB on the free tier.',
    tier: 'Free (500 MB)',
    link: 'https://supabase.com/',
    color: 'border-emerald-green',
  },
  {
    name: 'Cloudflare R2',
    role: 'Image Hosting',
    description:
      '61k+ WebP images served via S3-compatible storage with zero egress fees. Content-addressed by MD5 hash.',
    tier: 'Free (10 GB)',
    link: 'https://www.cloudflare.com/developer-platform/r2/',
    color: 'border-ore-gold',
  },
  {
    name: 'Multilingual-E5-Large',
    role: 'Embedding Model',
    description:
      'State-of-the-art 1024-dimensional text embedding model. Generated locally during indexing, and via Pinecone Inference API for live queries.',
    tier: 'Open Source / API',
    link: 'https://huggingface.co/intfloat/multilingual-e5-large',
    color: 'border-diamond-blue',
  },
  {
    name: 'DeepSeek',
    role: 'Chat Completion',
    description:
      'Generates answers with inline citations from retrieved context. System prompt enforces Minecraft expertise and citation formatting.',
    tier: 'API Credits',
    link: 'https://www.deepseek.com/',
    color: 'border-redstone-red',
  },
  {
    name: 'Next.js + Vercel',
    role: 'Frontend & Hosting',
    description:
      'React 18 with App Router, Tailwind glassmorphism theme, and Minecraft-style typography. Deployed on Vercel with Turbopack.',
    tier: 'Free (Hobby)',
    link: 'https://vercel.com/',
    color: 'border-white/50',
  },
];

const STATS: Stat[] = [
  { value: '12,487', label: 'Wiki Pages' },
  { value: '61,248', label: 'Images' },
  { value: '1024d', label: 'Embedding Dimensions' },
  { value: '~340k', label: 'Text Chunks' },
];

/* ------------------------------------------------------------------ */
/*  Subcomponents                                                      */
/* ------------------------------------------------------------------ */

function PipelineStepCard({ step, index, isLast }: { step: PipelineStep; index: number; isLast: boolean }) {
  const Icon = step.icon;
  return (
    <div className="flex items-start gap-3 sm:gap-4">
      {/* Step indicator */}
      <div className="flex flex-col items-center flex-shrink-0">
        <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-lg glass glass-light dark:glass-dark flex items-center justify-center border border-gray-300 dark:border-gray-600">
          <Icon className={`w-5 h-5 sm:w-6 sm:h-6 ${step.color}`} />
        </div>
        {!isLast && (
          <div className="w-0.5 h-6 sm:h-8 bg-gradient-to-b from-gray-400 to-transparent dark:from-gray-600 mt-1" />
        )}
      </div>
      {/* Content */}
      <div className="pb-6 sm:pb-8 min-w-0 flex-1">
        <div className="flex items-center gap-2 flex-wrap">
          <h3 className={`font-bold text-sm sm:text-base ${step.color}`}>
            {index + 1}. {step.title}
          </h3>
          <span className="text-[10px] sm:text-xs text-gray-500 dark:text-gray-400 font-mono">
            {step.subtitle}
          </span>
        </div>
        <p className="text-xs sm:text-sm text-gray-700 dark:text-gray-300 mt-1 leading-relaxed">
          {step.description}
        </p>
      </div>
    </div>
  );
}

function TechStackCard({ card }: { card: TechCard }) {
  return (
    <a
      href={card.link}
      target="_blank"
      rel="noopener noreferrer"
      className={`block glass glass-light dark:glass-dark rounded-lg p-3 sm:p-4 border-l-4 ${card.color} hover:scale-[1.02] transition-transform`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="font-bold text-sm sm:text-base text-gray-900 dark:text-gray-100">
            {card.name}
          </h3>
          <p className="text-[10px] sm:text-xs text-gray-500 dark:text-gray-400">
            {card.role} · {card.tier}
          </p>
        </div>
        <ExternalLink className="w-3 h-3 sm:w-4 sm:h-4 text-gray-400 flex-shrink-0 mt-1" />
      </div>
      <p className="text-xs sm:text-sm text-gray-700 dark:text-gray-300 mt-2 leading-relaxed">
        {card.description}
      </p>
    </a>
  );
}

function StatBadge({ stat }: { stat: Stat }) {
  return (
    <div className="glass glass-light dark:glass-dark rounded-lg p-3 sm:p-4 text-center select-none hover:scale-105 transition-transform duration-300">
      <div className="text-lg sm:text-2xl font-bold text-ore-gold drop-shadow-[0_1px_1px_rgba(0,0,0,0.8)]">
        {stat.value}
      </div>
      <div className="text-[10px] sm:text-xs text-gray-600 dark:text-gray-300 mt-0.5 font-medium drop-shadow-[0_1px_0_rgba(255,255,255,0.5)] dark:drop-shadow-[0_1px_1px_rgba(0,0,0,0.8)]">
        {stat.label}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Page                                                               */
/* ------------------------------------------------------------------ */

export default function AboutPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 py-6 sm:py-10 space-y-10 sm:space-y-14">
          {/* ---- Hero ---- */}
          <section className="text-center">
            <h1 className="text-3xl sm:text-5xl font-bold text-white text-stroke-sm sm:text-stroke mb-3">
              How Ore-acle Works
            </h1>
            <p className="text-sm sm:text-lg text-gray-200 max-w-2xl mx-auto leading-relaxed">
              A Retrieval-Augmented Generation system that answers Minecraft
              questions using real wiki data — with{' '}
              <span className="text-ore-gold font-bold">NotebookLM-style citations</span>{' '}
              back to the source.
            </p>
          </section>

          {/* ---- Stats ---- */}
          <section>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3 text-shadow-sm">
              {STATS.map((s) => (
                <StatBadge key={s.label} stat={s} />
              ))}
            </div>
          </section>

          {/* ---- Architecture Diagram (text-based) ---- */}
          <section className="glass glass-light dark:glass-dark rounded-xl p-4 sm:p-6">
            <h2 className="text-lg sm:text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
              <Search className="w-5 h-5 text-diamond-blue" />
              Search Architecture
            </h2>
            <div className="space-y-3 text-xs sm:text-sm text-gray-700 dark:text-gray-300">
              <p>When you ask a question, Ore-acle runs a <span className="text-diamond-blue font-bold">hybrid search</span> combining two strategies:</p>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 mt-3">
                {/* Semantic */}
                <div className="glass glass-light dark:glass-dark rounded-lg p-3 border border-purple-400/30">
                  <h4 className="font-bold text-purple-400 mb-1 flex items-center gap-1.5">
                    <Brain className="w-4 h-4" /> Semantic Search
                  </h4>
                  <div className="space-y-1 text-[11px] sm:text-xs leading-relaxed">
                    <p>1. Your query is embedded via Pinecone Inference</p>
                    <p>2. Pinecone finds the 20 closest vectors</p>
                    <p>3. Full chunk content is returned from Pinecone metadata</p>
                  </div>
                </div>
                {/* Keyword */}
                <div className="glass glass-light dark:glass-dark rounded-lg p-3 border border-emerald-green/30">
                  <h4 className="font-bold text-emerald-green mb-1 flex items-center gap-1.5">
                    <FileText className="w-4 h-4" /> Keyword Search
                  </h4>
                  <div className="space-y-1 text-[11px] sm:text-xs leading-relaxed">
                    <p>1. Your query hits Supabase&apos;s tsvector index</p>
                    <p>2. PostgreSQL full-text search ranks matches</p>
                    <p>3. Matching IDs are returned and hydrated from Pinecone</p>
                  </div>
                </div>
              </div>

              <div className="glass glass-light dark:glass-dark rounded-lg p-3 border border-ore-gold/30 mt-3">
                <h4 className="font-bold text-ore-gold mb-1 flex items-center gap-1.5">
                  <Layers className="w-4 h-4" /> Reciprocal Rank Fusion (k=60)
                </h4>
                <p className="text-[11px] sm:text-xs leading-relaxed">
                  Results from both strategies are merged with RRF scoring. Each result gets a score of 1/(k + rank) from
                  each list, and the combined scores determine the final top-10 chunks sent to the language model.
                </p>
              </div>

              <div className="glass glass-light dark:glass-dark rounded-lg p-3 border border-redstone-red/30 mt-3">
                <h4 className="font-bold text-redstone-red mb-1 flex items-center gap-1.5">
                  <MessageSquare className="w-4 h-4" /> Answer Generation
                </h4>
                <p className="text-[11px] sm:text-xs leading-relaxed">
                  The top chunks are formatted as numbered sources and sent to DeepSeek with a system
                  prompt enforcing inline citations like [1][2]. The response preserves source traceability — hover any
                  citation to see the exact wiki text it came from.
                </p>
              </div>
            </div>
          </section>

          {/* ---- Data Pipeline ---- */}
          <section className="glass glass-light dark:glass-dark rounded-xl p-4 sm:p-6">
            <h2 className="text-lg sm:text-2xl font-bold text-gray-900 dark:text-gray-100 mb-6 flex items-center gap-2">
              <HardDrive className="w-5 h-5 text-ore-gold" />
              Data Pipeline
            </h2>
            <div>
              {PIPELINE_STEPS.map((step, i) => (
                <PipelineStepCard
                  key={step.title}
                  step={step}
                  index={i}
                  isLast={i === PIPELINE_STEPS.length - 1}
                />
              ))}
            </div>
          </section>

          {/* ---- Tech Stack ---- */}
          <section>
            <h2 className="text-lg sm:text-2xl font-bold text-white text-stroke-sm sm:text-stroke mb-4 flex items-center gap-2">
              <Cpu className="w-5 h-5 text-diamond-blue" />
              Tech Stack
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
              {TECH_STACK.map((card) => (
                <TechStackCard key={card.name} card={card} />
              ))}
            </div>
          </section>

          {/* ---- Design Decisions ---- */}
          <section className="glass glass-light dark:glass-dark rounded-xl p-4 sm:p-6">
            <h2 className="text-lg sm:text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-emerald-green" />
              Design Decisions
            </h2>
            <div className="space-y-3 text-xs sm:text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
              <div>
                <h4 className="font-bold text-gray-900 dark:text-gray-100">Why hybrid search?</h4>
                <p>
                  Semantic search finds conceptually similar content (&ldquo;hostile mobs that spawn in the dark&rdquo;), 
                  but keyword search is better for exact terms (&ldquo;Ender Dragon health points&rdquo;). 
                  Reciprocal Rank Fusion combines both for robust retrieval without tuning a weight parameter.
                </p>
              </div>
              <div>
                <h4 className="font-bold text-gray-900 dark:text-gray-100">Why Multilingual-E5-Large?</h4>
                <p>
                  We switched from Nomic v1.5 (768d) to Multilingual-E5-Large (1024d) because it ranks significantly 
                  higher on the MTEB leaderboard for retrieval tasks. The larger vector size captures more semantic 
                  nuance, and Pinecone&apos;s serverless infrastructure handles the 1024 dimensions effortlessly.
                </p>
              </div>
              <div>
              </div>
              <div>
                <h4 className="font-bold text-gray-900 dark:text-gray-100">Why NotebookLM-style citations?</h4>
                <p>
                  Every answer links back to the exact wiki section it drew from. This makes the system transparent 
                  and verifiable — you can check the source with one click, just like Google&apos;s NotebookLM.
                </p>
              </div>
            </div>
          </section>

          {/* ---- Limitations ---- */}
          <section className="glass glass-light dark:glass-dark rounded-xl p-4 sm:p-6 mb-8">
            <h2 className="text-lg sm:text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-red-500" />
              Limitations
            </h2>
            <div className="space-y-3 text-xs sm:text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
              <div>
                <h4 className="font-bold text-gray-900 dark:text-gray-100">Knowledge Cutoff</h4>
                <p>
                  The database is a static snapshot from early 2026. Future Minecraft updates (e.g., 1.23+) will not be
                  reflected unless the wiki data is re-scraped and re-indexed.
                </p>
              </div>
              <div>
                <h4 className="font-bold text-gray-900 dark:text-gray-100">Context Window Constraints</h4>
                <p>
                  While RAG is powerful, it has a finite context window (top 10 retrieved chunks). Extremely broad
                  questions like &ldquo;List every item in the game&rdquo; may result in incomplete lists as not all relevant
                  chunks can fit into the prompt.
                </p>
              </div>
              <div>
                <h4 className="font-bold text-gray-900 dark:text-gray-100">No Vision Analysis</h4>
                <p>
                  Images are retrieved based on their metadata (captions, alt text, surrounding text), not by analyzing the
                  pixel data itself. The system cannot &ldquo;see&rdquo; the image content directly.
                </p>
              </div>
              <div>
                <h4 className="font-bold text-gray-900 dark:text-gray-100">Wiki Dependence</h4>
                <p>
                  Answers are grounded strictly in the Minecraft Wiki text. If the wiki contains an error or outdated
                  information, the model may propagate that inaccuracy.
                </p>
              </div>
            </div>
          </section>

          {/* ---- Footer ---- */}
          <footer className="text-center pb-6 sm:pb-10 space-y-2">
            <a
              href="https://github.com/annoyedjames/Ore-acle"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-sm text-gray-300 hover:text-ore-gold transition-colors"
            >
              <Github className="w-4 h-4" />
              View on GitHub
            </a>
            <p className="text-[10px] sm:text-xs text-gray-500">
              Not affiliated with Mojang, Microsoft, Oracle, or Minecraft.
              <br />
              Wiki content sourced from{' '}
              <a
                href="https://minecraft.wiki"
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-ore-gold"
              >
                minecraft.wiki
              </a>{' '}
              under CC BY-SA 3.0.
            </p>
          </footer>
        </div>
      </main>
    </div>
  );
}
