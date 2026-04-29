export function getTextureUrl(itemName: string): string {
  if (!itemName || itemName === '.') return '';
  
  // Normalize string for filenames (e.g. "Iron Ingot" -> "iron_ingot")
  // also handle "Block of Iron" -> "iron_block" (if you map them later)
  let normalized = itemName
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '');

  // For now, since everything you provided is in 'all_blocks'
  return '/textures/all_blocks/' + normalized + '.png';
}
