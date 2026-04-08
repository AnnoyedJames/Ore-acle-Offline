import sys; sys.path.insert(0, '.')
from backend.database.local_stores import SQLiteStore
s = SQLiteStore()
r = s.search('how do i get a water bucket in minecraft', limit=5)
print(f'Results: {len(r)}')
for x in r:
    print(f"  {x['chunk_id']} | {x['page_title']} | rank={x['rank']:.4f}")
