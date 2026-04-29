import os
file_path = 'frontend/src/components/CraftingGrid.tsx'
with open(file_path, 'rb') as f:
    content = f.read()

# Let's decode it safely
if content.startswith(b'\xff\xfe'):
    text = content.decode('utf-16-le')
elif content.startswith(b'\xfe\xff'):
    text = content.decode('utf-16-be')
elif content.startswith(b'\xef\xbb\xbf'):
    text = content.decode('utf-8-sig')
else:
    # Try utf-8
    text = content.decode('utf-8', errors='ignore')

# Strip any garbage character before "import"
idx = text.find('import React')
if idx != -1:
    text = text[idx:]

with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(text)

print('SUCCESS')
