import React from 'react';
import { getTextureUrl } from '@/lib/texture-mapper';

interface CraftingRecipeProps {
  s0?: string; s1?: string; s2?: string;
  s3?: string; s4?: string; s5?: string;
  s6?: string; s7?: string; s8?: string;
  result: string;
}

export default function CraftingGrid({ s0, s1, s2, s3, s4, s5, s6, s7, s8, result }: CraftingRecipeProps) {
  const slots = [s0, s1, s2, s3, s4, s5, s6, s7, s8];

  const handleImageError = (e: React.SyntheticEvent<HTMLImageElement, Event>) => {
    console.error('Failed to load image:', e.currentTarget.src, 'for item:', e.currentTarget.alt);
    // Hide the broken image to show a gray invisible box or perhaps an error icon
    e.currentTarget.style.opacity = '0';
  };

  React.useEffect(() => {
    console.log('CraftingGrid rendered with:', { s0, s1, s2, s3, s4, s5, s6, s7, s8, result });
  }, [s0, s1, s2, s3, s4, s5, s6, s7, s8, result]);

  return (
    <div className="my-6 inline-flex flex-col sm:flex-row items-center justify-center p-4 rounded-md bg-[#C6C6C6] border-2 border-t-[#FFFFFF] border-l-[#FFFFFF] border-r-[#555555] border-b-[#555555] shadow-lg select-none font-sans">
      
      {/* 3x3 Input Grid */}
      <div className="grid grid-cols-3 gap-0.5 p-1">
        {slots.map((item, idx) => (
          <div 
            key={idx} 
            className="w-10 h-10 bg-[#8B8B8B] border border-t-[#373737] border-l-[#373737] border-r-[#FFFFFF] border-b-[#FFFFFF] flex items-center justify-center relative cursor-help group"
          >
            {item && item !== '.' && item !== '' && (
              <>
                <img 
                  src={getTextureUrl(item)} 
                  alt={item}
                  className="w-8 h-8 object-contain transition-opacity"
                  style={{ imageRendering: 'pixelated' }}
                  onError={handleImageError}
                />
                
                {/* Custom Tooltip */}
                <div className="absolute opacity-0 group-hover:opacity-100 z-50 bottom-[120%] pointer-events-none whitespace-nowrap bg-[#100010] text-[#AAAAAA] text-xs px-2 py-1 border-2 border-purple-900/50 shadow-md drop-shadow-[0_1px_1px_rgba(0,0,0,0.8)]">
                  {item}
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      {/* Arrow Graphic */}
      <div className="mx-6 my-4 sm:my-0 flex shrink-0 items-center justify-center text-[#555555] font-bold text-3xl">
         {/* Using an svg that looks like the classic progression arrow if needed, but a right chevron works */}
         <svg width="32" height="24" viewBox="0 0 32 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M0 8H16V0L32 12L16 24V16H0V8Z" fill="#8B8B8B" stroke="#373737" strokeWidth="2"/>
            <path d="M0 8H16V0L32 12V12.5L16 24V16H0V8Z" fill="#D2D2D2" />
         </svg>
      </div>

      {/* Result Slot */}
      <div className="p-1 relative flex items-center justify-center">
        <div className="w-[52px] h-[52px] bg-[#8B8B8B] border border-t-[#373737] border-l-[#373737] border-r-[#FFFFFF] border-b-[#FFFFFF] flex items-center justify-center relative cursor-help group">
          {result && result !== '.' && result !== '' && (
            <>
              <img 
                src={getTextureUrl(result)} 
                alt={result}
                className="w-10 h-10 object-contain transition-opacity"
                style={{ imageRendering: 'pixelated' }}
                onError={handleImageError}
              />
              <div className="absolute opacity-0 group-hover:opacity-100 z-50 bottom-[120%] pointer-events-none whitespace-nowrap bg-[#100010] text-[#FFFFFF] font-bold text-xs px-2 py-1 border-2 border-purple-900/50 shadow-md drop-shadow-[0_1px_1px_rgba(0,0,0,0.8)]">
                {result}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
