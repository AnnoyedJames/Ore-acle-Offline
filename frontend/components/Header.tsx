'use client';

import { useTheme } from 'next-themes';
import { Sun, Moon, Info, MessageSquare } from 'lucide-react';
import { useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';
import Image from 'next/image';
import Link from 'next/link';

export default function Header() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const pathname = usePathname();
  const isAboutPage = pathname === '/about';

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <header className="flex items-center justify-between p-2 sm:p-4 glass glass-light dark:glass-dark border-b border-gray-300 dark:border-gray-700">
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="w-6 h-6 sm:w-8 sm:h-8" />
          <h1 className="text-lg sm:text-2xl font-bold">Ore-acle</h1>
        </div>
        <div className="flex items-center gap-1.5 sm:gap-2">
          <div className="w-8 h-8 sm:w-10 sm:h-10" />
          <div className="w-8 h-8 sm:w-10 sm:h-10" />
        </div>
      </header>
    );
  }

  return (
    <header className="flex items-center justify-between p-2 sm:p-4 glass glass-light dark:glass-dark border-b border-gray-300 dark:border-gray-700">
      {/* Logo — links home */}
      <Link href="/" className="flex items-center gap-2 sm:gap-3 min-w-0 hover:opacity-80 transition-opacity">
        <Image 
          src="/textures/grass_block_side.png" 
          alt="Grass Block" 
          width={32} 
          height={32}
          className="pixelated w-6 h-6 sm:w-8 sm:h-8"
        />
        <div className="min-w-0">
          <h1 className="text-lg sm:text-2xl font-bold text-gray-900 dark:text-gray-100 truncate">
            Ore-acle
          </h1>
          <p className="text-[8px] sm:text-[10px] text-gray-500 dark:text-gray-400 -mt-1 truncate">
            Not affiliated with Mojang, Microsoft, Oracle, or Minecraft
          </p>
        </div>
      </Link>
      <div className="flex-1 sm:flex-none flex items-center justify-end gap-1.5 sm:gap-2 ml-2 sm:ml-0">
        {isAboutPage ? (
          <Link
            href="/"
            className="flex-1 sm:flex-none flex items-center justify-center gap-1.5 px-3 py-1 sm:px-3 sm:py-2 rounded-lg glass glass-light dark:glass-dark border border-gray-300 dark:border-gray-600 hover:border-ore-gold dark:hover:border-ore-gold transition-colors text-gray-700 dark:text-gray-200"
          >
            <MessageSquare className="w-3 h-3 sm:w-4 sm:h-4 text-diamond-blue flex-shrink-0" />
            <span className="text-xs sm:text-sm font-medium truncate">Chat</span>
          </Link>
        ) : (
          <Link
            href="/about"
            className="flex-1 sm:flex-none flex items-center justify-center gap-1.5 px-3 py-1 sm:px-3 sm:py-2 rounded-lg glass glass-light dark:glass-dark border border-gray-300 dark:border-gray-600 hover:border-ore-gold dark:hover:border-ore-gold transition-colors text-gray-700 dark:text-gray-200"
          >
            <Info className="w-3 h-3 sm:w-4 sm:h-4 text-diamond-blue flex-shrink-0" />
            <span className="text-xs sm:text-sm font-medium truncate">About this project</span>
          </Link>
        )}
        <button
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className="p-1.5 sm:p-2 rounded-lg glass glass-light dark:glass-dark border border-gray-300 dark:border-gray-600 hover:border-ore-gold dark:hover:border-ore-gold transition-colors"
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? (
            <Sun className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-400" />
          ) : (
            <Moon className="w-4 h-4 sm:w-5 sm:h-5 text-blue-600" />
          )}
        </button>
      </div>
    </header>
  );
}
