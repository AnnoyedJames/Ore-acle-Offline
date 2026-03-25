import type { Metadata } from 'next'
import { ThemeProvider } from 'next-themes'
import OverworldBackground from '@/components/OverworldBackground'
import './globals.css'

export const metadata: Metadata = {
  title: 'Ore-acle | Minecraft Wiki RAG',
  description: 'Ask questions about Minecraft and get answers with citations',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Preload critical UI textures */}
        <link rel="preload" as="image" href="/textures/bookshelf.png" />
        <link rel="preload" as="image" href="/textures/crafting_table_side.png" />
        <link rel="preload" as="image" href="/textures/furnace_front.png" />
        {/* Preload background textures to speed up OverworldBackground feeling */}
        <link rel="preload" as="image" href="/textures/stone.png" />
        <link rel="preload" as="image" href="/textures/deepslate.png" />
        <link rel="preload" as="image" href="/textures/dirt.png" />
      </head>
      <body>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
          <OverworldBackground />
          <div className="fixed inset-0 z-0 bg-black/40 dark:bg-black/65 transition-colors duration-300 pointer-events-none" />
          <div className="relative z-10 w-full overflow-x-hidden">
            {children}
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}
