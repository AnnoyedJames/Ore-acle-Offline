import { Routes, Route } from 'react-router-dom';
import OverworldBackground from '@/components/OverworldBackground';
import Home from '@/pages/Home';
import About from '@/pages/About';

export default function App() {
  return (
    <>
      <OverworldBackground />
      <div className="fixed inset-0 z-0 bg-black/40 dark:bg-black/65 transition-colors duration-300 pointer-events-none" />
      <div className="relative z-10 w-full overflow-x-hidden">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </div>
    </>
  );
}
