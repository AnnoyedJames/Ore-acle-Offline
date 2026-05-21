// Mapping from canonical item names (lowercase_underscored) to actual texture filenames
// when the simple normalization doesn't match the file on disk.
const TEXTURE_OVERRIDES: Record<string, string> = {
  'redstone_dust': 'redstone_dust_dot',
  'compass': 'compass_00',
  'clock': 'clock_00',
  'crossbow': 'crossbow_standby',
};

export function getTextureUrl(itemName: string): string {
  if (!itemName || itemName === '.') return '';
  
  // Normalize string for filenames (e.g. "Iron Ingot" -> "iron_ingot")
  let normalized = itemName
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '');

  // Check override map first
  if (TEXTURE_OVERRIDES[normalized]) {
    return '/textures/all_blocks/' + TEXTURE_OVERRIDES[normalized] + '.png';
  }

  // Items known to have no texture file — returns empty (grid shows no icon, just tooltip)
  if (normalized === 'shield') return '';

  return '/textures/all_blocks/' + normalized + '.png';
}
