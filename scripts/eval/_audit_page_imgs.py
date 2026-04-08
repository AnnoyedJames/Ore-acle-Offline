import json, os
data = json.load(open('data/processed/metadata.json', encoding='utf-8'))
pages = {p['title']: p for p in data['pages']}
targets = ['Creeper', 'Zombie', 'Diamond', 'Crafting', 'Nether']
for t in targets:
    p = pages.get(t, {})
    imgs = p.get('images', [])
    with_text = [i for i in imgs if i.get('caption') or i.get('alt_text')]
    on_disk = [i for i in imgs if os.path.exists(i.get('file_path', ''))]
    print(f"{t}: {len(imgs)} total, {len(with_text)} with caption/alt, {len(on_disk)} on disk")
    for i in with_text[:3]:
        print(f"  {i.get('local_filename','')} | cap={i.get('caption','')[:60]} | alt={i.get('alt_text','')[:60]}")
