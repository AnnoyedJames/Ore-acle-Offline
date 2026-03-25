'use client';

import { useEffect, useRef } from 'react';

// ================ Configuration ================

const NATIVE_BLOCK = 16;  // Native Minecraft texture size (px)
const GRID_COLS = 64;     // World width in blocks

// Layer heights (in blocks, top to bottom) — no sky
const SURFACE_ROWS = 3;   // grass + dirt/sand
const STONE_ROWS = 14;    // deepslate moved down 4 blocks total
const DEEPSLATE_ROWS = 7;
const BEDROCK_ROWS = 1;
const TOTAL_ROWS = SURFACE_ROWS + STONE_ROWS + DEEPSLATE_ROWS + BEDROCK_ROWS; // 25

// Scroll speed: pixels per second at 1x scale (smooth, ambient)
const SCROLL_PPS = 20;

// ================ Ore Vein Configuration ================

interface OreConfig {
  key: string;
  veinCount: number;
  minRow: number;   // relative to layer start
  maxRow: number;   // relative to layer start (inclusive)
  maxVeinSize: number;
}

// Stone layer ores — mimics real Minecraft Y-level distribution
const STONE_ORES: OreConfig[] = [
  { key: 'coal_ore',     veinCount: 12, minRow: 0, maxRow: 7, maxVeinSize: 3 },
  { key: 'iron_ore',     veinCount: 8,  minRow: 0, maxRow: 7, maxVeinSize: 3 },
  { key: 'copper_ore',   veinCount: 5,  minRow: 1, maxRow: 5, maxVeinSize: 2 },
  { key: 'gold_ore',     veinCount: 3,  minRow: 3, maxRow: 7, maxVeinSize: 2 },
  { key: 'lapis_ore',    veinCount: 3,  minRow: 2, maxRow: 6, maxVeinSize: 2 },
  { key: 'redstone_ore', veinCount: 4,  minRow: 5, maxRow: 7, maxVeinSize: 3 },
  { key: 'emerald_ore',  veinCount: 1,  minRow: 0, maxRow: 7, maxVeinSize: 1 },
  { key: 'diamond_ore',  veinCount: 2,  minRow: 6, maxRow: 7, maxVeinSize: 2 },
];

// Deepslate layer ores
const DEEPSLATE_ORES: OreConfig[] = [
  { key: 'deepslate_coal_ore',     veinCount: 4,  minRow: 0, maxRow: 2, maxVeinSize: 2 },
  { key: 'deepslate_iron_ore',     veinCount: 7,  minRow: 0, maxRow: 6, maxVeinSize: 3 },
  { key: 'deepslate_copper_ore',   veinCount: 3,  minRow: 0, maxRow: 3, maxVeinSize: 2 },
  { key: 'deepslate_gold_ore',     veinCount: 3,  minRow: 0, maxRow: 6, maxVeinSize: 2 },
  { key: 'deepslate_lapis_ore',    veinCount: 2,  minRow: 0, maxRow: 4, maxVeinSize: 2 },
  { key: 'deepslate_redstone_ore', veinCount: 5,  minRow: 0, maxRow: 6, maxVeinSize: 3 },
  { key: 'deepslate_emerald_ore',  veinCount: 1,  minRow: 0, maxRow: 6, maxVeinSize: 1 },
  { key: 'deepslate_diamond_ore',  veinCount: 3,  minRow: 0, maxRow: 6, maxVeinSize: 2 },
];

// All texture filenames needed
const TEXTURE_NAMES = [
  'stone', 'deepslate', 'dirt', 'sand', 'grass_block_side', 'bedrock',
  'coal_ore', 'iron_ore', 'copper_ore', 'gold_ore',
  'lapis_ore', 'redstone_ore', 'emerald_ore', 'diamond_ore',
  'deepslate_coal_ore', 'deepslate_iron_ore', 'deepslate_copper_ore',
  'deepslate_gold_ore', 'deepslate_lapis_ore', 'deepslate_redstone_ore',
  'deepslate_emerald_ore', 'deepslate_diamond_ore',
];

// ================ Helpers ================

/** Simple seedable PRNG (Lehmer / Park-Miller) */
function seededRandom(seed: number) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return function (): number {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

/** Load an image, returning null on failure */
function loadImage(src: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => {
      console.warn(`Failed to load texture: ${src}`);
      resolve(null);
    };
    img.src = src;
  });
}

// ================ Grid Generation ================

type Block = string; // texture key

function generateGrid(rand: () => number): Block[][] {
  const grid: Block[][] = [];

  // === Base layers (sharp boundaries) ===
  
  // Surface row 0: grass block side
  grid.push(new Array(GRID_COLS).fill('grass_block_side'));

  // Surface rows 1-2: dirt with some sand patches
  for (let r = 0; r < SURFACE_ROWS - 1; r++) {
    const row: Block[] = [];
    for (let c = 0; c < GRID_COLS; c++) {
      row.push(rand() < 0.75 ? 'dirt' : 'sand');
    }
    grid.push(row);
  }

  // Stone rows
  for (let r = 0; r < STONE_ROWS; r++) {
    grid.push(new Array(GRID_COLS).fill('stone'));
  }

  // Deepslate rows
  for (let r = 0; r < DEEPSLATE_ROWS; r++) {
    grid.push(new Array(GRID_COLS).fill('deepslate'));
  }

  // Bedrock bottom row
  grid.push(new Array(GRID_COLS).fill('bedrock'));

  // === Layer bleeding ===
  
  // Dirt bleeding into stone (rows 2-3 can have some dirt)
  const dirtStoneTransition = SURFACE_ROWS; // row 2 starts stone
  for (let c = 0; c < GRID_COLS; c++) {
    const bleedDepth = rand() < 0.6 ? 1 : (rand() < 0.3 ? 2 : 0); // 1 common, 2 max
    for (let d = 0; d < bleedDepth; d++) {
      const row = dirtStoneTransition + d;
      if (row < grid.length && grid[row][c] === 'stone') {
        grid[row][c] = 'dirt';
      }
    }
  }

  // Deepslate bleeding into stone (some deepslate rises 1-2 blocks)
  const stoneDeepslateTransition = SURFACE_ROWS + STONE_ROWS; // row 12
  for (let c = 0; c < GRID_COLS; c++) {
    const bleedHeight = rand() < 0.6 ? 1 : (rand() < 0.3 ? 2 : 0); // 1 common, 2 max
    for (let d = 1; d <= bleedHeight; d++) {
      const row = stoneDeepslateTransition - d;
      if (row >= SURFACE_ROWS && grid[row][c] === 'stone') {
        grid[row][c] = 'deepslate';
      }
    }
  }

  // Random bedrock in second-to-bottom deepslate row
  const secondToBottom = TOTAL_ROWS - 2; // row 18
  for (let c = 0; c < GRID_COLS; c++) {
    if (rand() < 0.25 && grid[secondToBottom][c] === 'deepslate') { // 25% chance
      grid[secondToBottom][c] = 'bedrock';
    }
  }

  // === Place ore veins ===
  const stoneStart = SURFACE_ROWS;
  placeOreVeins(grid, stoneStart, STONE_ROWS, STONE_ORES, rand);

  const deepslateStart = stoneStart + STONE_ROWS;
  placeOreVeins(grid, deepslateStart, DEEPSLATE_ROWS, DEEPSLATE_ORES, rand);

  return grid;
}

function placeOreVeins(
  grid: Block[][],
  layerStart: number,
  layerHeight: number,
  ores: OreConfig[],
  rand: () => number,
) {
  for (const ore of ores) {
    for (let v = 0; v < ore.veinCount; v++) {
      // Random position within the ore's valid depth range
      const row = layerStart + ore.minRow +
        Math.floor(rand() * (ore.maxRow - ore.minRow + 1));
      const col = Math.floor(rand() * GRID_COLS);

      if (row >= grid.length) continue;
      grid[row][col] = ore.key;

      // Grow vein: 1–maxVeinSize additional blocks in random adjacent directions
      const veinSize = Math.max(1, Math.ceil(rand() * ore.maxVeinSize));
      let curRow = row;
      let curCol = col;

      for (let i = 1; i < veinSize; i++) {
        const dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]];
        const [dr, dc] = dirs[Math.floor(rand() * dirs.length)];
        const newRow = curRow + dr;
        const newCol = (curCol + dc + GRID_COLS) % GRID_COLS; // wrap horizontally

        if (
          newRow >= layerStart + ore.minRow &&
          newRow <= layerStart + ore.maxRow &&
          newRow < grid.length
        ) {
          grid[newRow][newCol] = ore.key;
          curRow = newRow;
          curCol = newCol;
        }
      }
    }
  }
}

// ================ Component ================

export default function OverworldBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let cancelled = false;
    let resizeHandler: (() => void) | null = null;

    async function init() {
      // Load all textures in parallel
      const textures: Record<string, HTMLImageElement> = {};
      await Promise.all(
        TEXTURE_NAMES.map(async (name) => {
          const img = await loadImage(`/textures/${name}.png`);
          if (img) textures[name] = img;
        })
      );

      if (cancelled) return;

      // Generate the world grid
      const rand = seededRandom(Date.now());
      const grid = generateGrid(rand);

      // Build offscreen canvas at native texture resolution
      // Double the width so we can scroll seamlessly without slicing
      const tileW = GRID_COLS * NATIVE_BLOCK;
      const offW = tileW * 2;
      const offH = TOTAL_ROWS * NATIVE_BLOCK;
      const offscreen = document.createElement('canvas');
      offscreen.width = offW;
      offscreen.height = offH;
      const offCtx = offscreen.getContext('2d')!;
      offCtx.imageSmoothingEnabled = false;

      // Draw terrain blocks (two copies side by side for seamless wrap)
      for (let copy = 0; copy < 2; copy++) {
        const xOff = copy * tileW;
        for (let r = 0; r < TOTAL_ROWS; r++) {
          for (let c = 0; c < GRID_COLS; c++) {
            const block = grid[r][c];
            const tex = textures[block];
            if (tex) {
              offCtx.drawImage(
                tex,
                xOff + c * NATIVE_BLOCK, r * NATIVE_BLOCK,
                NATIVE_BLOCK, NATIVE_BLOCK,
              );
            }
          }
        }
      }

      if (cancelled) return;

      // Resize canvas to fill viewport
      const resize = () => {
        if (!canvas) return;
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      };
      resize();
      window.addEventListener('resize', resize);
      resizeHandler = resize;

      // Animation loop — sub-pixel smooth scrolling
      let offset = 0;
      let lastTime = performance.now();

      function animate(time: number) {
        if (cancelled || !canvas || !ctx) return;

        const delta = (time - lastTime) / 1000;
        lastTime = time;

        const vw = canvas.width;
        const vh = canvas.height;

        // Scale so the grid fills viewport height exactly
        const scale = vh / offH;
        const scaledTileW = tileW * scale;

        // Advance scroll offset in screen-space (sub-pixel precision)
        offset = (offset + SCROLL_PPS * delta) % scaledTileW;

        // Draw — single drawImage with sub-pixel offset for buttery smoothness
        ctx.clearRect(0, 0, vw, vh);
        ctx.imageSmoothingEnabled = false;

        // The offscreen canvas is 2x wide, so we just shift it left
        // and it always has enough content to fill the viewport
        ctx.drawImage(
          offscreen,
          0, 0, offW, offH,
          -offset, 0, offW * scale, vh,
        );

        animRef.current = requestAnimationFrame(animate);
      }

      animRef.current = requestAnimationFrame(animate);
    }

    init();

    return () => {
      cancelled = true;
      cancelAnimationFrame(animRef.current);
      if (resizeHandler) {
        window.removeEventListener('resize', resizeHandler);
      }
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 -z-10"
      style={{ imageRendering: 'pixelated' }}
    />
  );
}
