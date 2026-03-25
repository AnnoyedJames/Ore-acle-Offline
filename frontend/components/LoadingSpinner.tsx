'use client';

export default function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center space-x-2">
      <div 
        className="w-6 h-6 animate-float bg-cover bg-center"
        style={{ 
          backgroundImage: 'url(/textures/bookshelf.png)',
          imageRendering: 'pixelated'
        }}
      />
      <div 
        className="w-6 h-6 animate-float bg-cover bg-center"
        style={{ 
          backgroundImage: 'url(/textures/crafting_table_side.png)',
          animationDelay: '0.2s',
          imageRendering: 'pixelated'
        }}
      />
      <div 
        className="w-6 h-6 animate-float bg-cover bg-center"
        style={{ 
          backgroundImage: 'url(/textures/furnace_front.png)',
          animationDelay: '0.4s',
          imageRendering: 'pixelated'
        }}
      />
    </div>
  );
}
