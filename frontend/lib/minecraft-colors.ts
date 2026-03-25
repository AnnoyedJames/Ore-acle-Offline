// Minecraft color code mappings (§ formatting codes)
// Reference: https://minecraft.wiki/w/Formatting_codes

export const MINECRAFT_COLORS: Record<string, string> = {
  // Color codes
  '0': '#000000', // Black
  '1': '#0000AA', // Dark Blue
  '2': '#00AA00', // Dark Green
  '3': '#00AAAA', // Dark Aqua
  '4': '#AA0000', // Dark Red
  '5': '#AA00AA', // Dark Purple
  '6': '#FFAA00', // Gold
  '7': '#AAAAAA', // Gray
  '8': '#555555', // Dark Gray
  '9': '#5555FF', // Blue
  'a': '#55FF55', // Green
  'b': '#55FFFF', // Aqua
  'c': '#FF5555', // Red
  'd': '#FF55FF', // Light Purple
  'e': '#FFFF55', // Yellow
  'f': '#FFFFFF', // White
};

export const MINECRAFT_FORMATS: Record<string, string> = {
  'l': 'bold',      // Bold
  'm': 'line-through', // Strikethrough
  'n': 'underline', // Underline
  'o': 'italic',    // Italic
  'r': 'reset',     // Reset
};

/**
 * Semantic tag → §-code mapping.
 * The LLM writes [tag]…[/tag]; the renderer expands them to §-codes
 * before parseMinecraftFormatting() turns those into styled HTML spans.
 *
 * Tags:
 *   [heading]  → §6§l…§r  Gold bold   — section titles
 *   [sub]      → §e…§r    Yellow      — sub-headings, numbers, Y-levels
 *   [term]     → §b…§r    Aqua        — items, blocks, mobs, technical terms
 *   [tip]      → §a…§r    Green       — helpful advice
 *   [warning]  → §c…§r    Red         — dangers, cautions
 *   [note]     → §7…§r    Gray        — asides, muted context
 */
const SEMANTIC_TAGS: Record<string, { prefix: string; suffix: string }> = {
  heading: { prefix: '§6§l', suffix: '§r' },
  sub:     { prefix: '§e',   suffix: '§r' },
  term:    { prefix: '§b',   suffix: '§r' },
  tip:     { prefix: '§a',   suffix: '§r' },
  warning: { prefix: '§c',   suffix: '§r' },
  note:    { prefix: '§7',   suffix: '§r' },
};

/**
 * Expand semantic tags ([tag]…[/tag]) into raw §-codes.
 * Must be called BEFORE parseMinecraftFormatting().
 * No nesting — inner tags are left as-is (the LLM is prompted not to nest).
 */
export function expandSemanticTags(text: string): string {
  return text.replace(
    /\[(heading|sub|term|tip|warning|note)\]([\s\S]*?)\[\/\1\]/g,
    (_match, tag: string, content: string) => {
      const mapping = SEMANTIC_TAGS[tag];
      if (!mapping) return content;
      return `${mapping.prefix}${content}${mapping.suffix}`;
    },
  );
}

/**
 * Parse Minecraft formatting codes (§x) in text and convert to HTML spans
 * Example: "§aGreen text§r with §lbold§r" → "<span style='color:#55FF55'>Green text</span> with <span style='font-weight:bold'>bold</span>"
 */
export function parseMinecraftFormatting(text: string): string {
  // Split by § character
  const parts = text.split('§');
  if (parts.length === 1) return text; // No formatting codes
  
  let result = parts[0]; // First part has no code
  let currentColor = '';
  let currentFormats: string[] = [];
  
  for (let i = 1; i < parts.length; i++) {
    const code = parts[i][0]?.toLowerCase();
    const content = parts[i].slice(1);
    
    if (!code) {
      result += content;
      continue;
    }
    
    // Reset all formatting
    if (code === 'r') {
      if (currentColor || currentFormats.length > 0) {
        result += '</span>';
      }
      currentColor = '';
      currentFormats = [];
      result += content;
      continue;
    }
    
    // Close previous span if exists
    if (currentColor || currentFormats.length > 0) {
      result += '</span>';
    }
    
    // Handle color codes
    if (MINECRAFT_COLORS[code]) {
      currentColor = MINECRAFT_COLORS[code];
    }
    
    // Handle format codes (accumulate, don't replace)
    if (MINECRAFT_FORMATS[code] && MINECRAFT_FORMATS[code] !== 'reset') {
      if (!currentFormats.includes(MINECRAFT_FORMATS[code])) {
        currentFormats.push(MINECRAFT_FORMATS[code]);
      }
    }
    
    // Open new span with styles
    if (currentColor || currentFormats.length > 0) {
      const styles: string[] = [];
      if (currentColor) styles.push(`color: ${currentColor}`);
      if (currentFormats.includes('bold')) styles.push('font-weight: bold');
      if (currentFormats.includes('italic')) styles.push('font-style: italic');
      if (currentFormats.includes('underline')) styles.push('text-decoration: underline');
      if (currentFormats.includes('line-through')) styles.push('text-decoration: line-through');
      
      result += `<span style="${styles.join('; ')}">`;
    }
    
    result += content;
  }
  
  // Close any remaining span
  if (currentColor || currentFormats.length > 0) {
    result += '</span>';
  }
  
  return result;
}
