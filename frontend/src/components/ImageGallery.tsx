import { useState } from 'react';
import { ImageResult } from '@/types';
import { X } from 'lucide-react';

interface ImageGalleryProps {
  images: ImageResult[];
}

export default function ImageGallery({ images }: ImageGalleryProps) {
  const [expandedImage, setExpandedImage] = useState<ImageResult | null>(null);

  if (images.length === 0) return null;

  const displayImages = images.slice(0, 6);

  return (
    <>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mt-4">
        {displayImages.map((image, idx) => (
          <div 
            key={idx}
            className="relative group cursor-pointer overflow-hidden rounded-lg border border-stone-300 dark:border-stone-700 hover:border-diamond-blue transition-colors bg-black/5"
            onClick={() => setExpandedImage(image)}
          >
            <div className="relative w-full aspect-[4/3] bg-black/5 dark:bg-black/20 flex items-center justify-center">
              <img
                src={image.url}
                alt={image.alt_text}
                loading="lazy"
                className="max-w-full max-h-full object-contain p-2 image-pixelated"
              />
            </div>
            <div className="absolute inset-x-0 bottom-0 bg-black/60 p-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <p className="text-xs text-white truncate text-center">{image.caption || image.alt_text}</p>
            </div>
          </div>
        ))}
      </div>

      {expandedImage && (
        <div 
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 p-4"
          onClick={() => setExpandedImage(null)}
        >
          <div className="relative max-w-4xl max-h-[90vh] glass glass-dark rounded-lg p-4">
            <button
              onClick={() => setExpandedImage(null)}
              className="absolute top-2 right-2 p-2 rounded-full bg-gray-800/80 hover:bg-gray-700 transition-colors"
              aria-label="Close"
            >
              <X className="w-5 h-5 text-white" />
            </button>
            
            <div className="w-full h-[60vh] flex items-center justify-center">
              <img
                src={expandedImage.url}
                alt={expandedImage.caption || expandedImage.alt_text}
                className="max-w-full max-h-full object-contain image-pixelated"
              />
            </div>
            
            <div className="mt-4 text-white">
              <p className="text-sm font-bold">{expandedImage.page_title}</p>
              {expandedImage.section && (
                <p className="text-xs text-gray-300">Section: {expandedImage.section}</p>
              )}
              {expandedImage.caption && (
                <p className="text-sm mt-2 italic">{expandedImage.caption}</p>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
