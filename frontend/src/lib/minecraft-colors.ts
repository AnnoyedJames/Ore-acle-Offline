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

const SEMANTIC_TAGS: Record<string, { prefix: string; suffix: string }> = {
  heading: { prefix: '§6§l', suffix: '§r' },
  sub:     { prefix: '§e',   suffix: '§r' },
  term:    { prefix: '§b',   suffix: '§r' },
  tip:     { prefix: '§a',   suffix: '§r' },
  warning: { prefix: '§c',   suffix: '§r' },
  note:    { prefix: '§7',   suffix: '§r' },
};

export function expandSemanticTags(text: string): string {
  if (!text) return text ?? '';
  return text.replace(
    /\[(heading|sub|term|tip|warning|note)\]([\s\S]*?)\[\/\1\]/g,
    (_match, tag: string, content: string) => {
      const mapping = SEMANTIC_TAGS[tag];
      if (!mapping) return content;
      return `${mapping.prefix}${content}${mapping.suffix}`;
    },
  );
}

export function parseMinecraftFormatting(text: string): string {
  const parts = text.split('§');
  if (parts.length === 1) return text;
  
  let result = parts[0];
  let currentColor = '';
  let currentFormats: string[] = [];
  
  for (let i = 1; i < parts.length; i++) {
    const code = parts[i][0]?.toLowerCase();
    const content = parts[i].slice(1);
    
    if (!code) {
      result += content;
      continue;
    }
    
    if (code === 'r') {
      if (currentColor || currentFormats.length > 0) {
        result += '</span>';
      }
      currentColor = '';
      currentFormats = [];
      result += content;
      continue;
    }
    
    if (currentColor || currentFormats.length > 0) {
      result += '</span>';
    }
    
    if (MINECRAFT_COLORS[code]) {
      currentColor = MINECRAFT_COLORS[code];
    }
    
    if (MINECRAFT_FORMATS[code] && MINECRAFT_FORMATS[code] !== 'reset') {
      if (!currentFormats.includes(MINECRAFT_FORMATS[code])) {
        currentFormats.push(MINECRAFT_FORMATS[code]);
      }
    }
    
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
  
  if (currentColor || currentFormats.length > 0) {
    result += '</span>';
  }
  
  return result;
}
