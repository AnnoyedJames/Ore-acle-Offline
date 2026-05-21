import { Citation } from '@/types';
import { ExternalLink } from 'lucide-react';

interface SourceCardProps {
  citation: Citation;
}

export default function SourceCard({ citation }: SourceCardProps) {
  const lead = citation.lead_text?.trim();
  const highlight = citation.highlight_text?.trim() || citation.cited_text;
  const trail = citation.trail_text?.trim();

  return (
    <div className="glass glass-light dark:glass-dark rounded-lg p-3 sm:p-4 shadow-lg min-w-0 w-[calc(100vw-2rem)] sm:min-w-[300px] sm:w-auto max-w-[400px]">
      <div className="flex items-start justify-between gap-2 mb-2">
        <h4 className="font-bold text-sm text-gray-900 dark:text-gray-100">
          {citation.page_title}
        </h4>
        <a
          href={citation.page_url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-diamond-blue hover:text-diamond-blue/80 transition-colors flex-shrink-0"
          aria-label="Open source article"
        >
          <ExternalLink className="w-4 h-4" />
        </a>
      </div>
      
      {citation.section && (
        <p className="text-[11px] uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-2">
          {citation.section}
        </p>
      )}

      <div className="border-l-2 border-ore-gold pl-3 space-y-1">
        {lead && (
          <p className="text-xs text-gray-500 dark:text-gray-400">
            ...{lead}
          </p>
        )}
        <blockquote className="text-sm text-gray-800 dark:text-gray-200 font-medium not-italic">
          {highlight}
        </blockquote>
        {trail && (
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {trail}...
          </p>
        )}
      </div>
    </div>
  );
}
