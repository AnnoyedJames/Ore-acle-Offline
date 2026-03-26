"""
Ore-acle Offline: Top Pages Heuristic

This script calculates a popularity heuristic for Minecraft Wiki pages using a custom 
algorithm that combines PageRank with content density and metadata modifiers.

Formula:
$$ \text{Score} = \log_{10}(PR \times 10^6 + 1) \times \log_{10}(\max(WC, 1)) \times M_{content} \times P_{meta} $$

Where:
- $PR$ is the raw PageRank score computed via power iteration on interlinks.
- $WC$ is the word count of the page text.
- $M_{content} = 1.0 + 0.5 \cdot I_{infobox} + 0.01 \cdot \min(N_{images}, 50)$
- $I_{infobox} \in \{0, 1\}$
- $P_{meta} = 0.3$ if the page is a meta-entity (e.g., Editions, Launchers, snapshot numbers), else $1.0$.
"""

import json
import math
from collections import defaultdict
from typing import Dict, List

def compute_pagerank(interlinks: Dict[str, List[str]], damping_factor: float = 0.85, max_iterations: int = 50, tol: float = 1e-6) -> Dict[str, float]:
    nodes = set(interlinks.keys())
    for targets in interlinks.values():
        nodes.update(targets)
    
    N = len(nodes)
    if N == 0:
        return {}

    pr = {node: 1.0 / N for node in nodes}
    out_degree = {node: len(targets) for node, targets in interlinks.items()}
    
    incoming = defaultdict(list)
    for src, targets in interlinks.items():
        for target in targets:
            incoming[target].append(src)
            
    dangling_nodes = [n for n in nodes if out_degree.get(n, 0) == 0]

    for i in range(max_iterations):
        new_pr = {}
        dangling_sum = sum(pr[n] for n in dangling_nodes)
        base_pr = (1.0 - damping_factor) / N + damping_factor * (dangling_sum / N)
        
        diff = 0.0
        for node in nodes:
            sum_in = 0.0
            for inc in incoming[node]:
                sum_in += pr[inc] / out_degree[inc]
            
            new_pr[node] = base_pr + damping_factor * sum_in
            diff += abs(new_pr[node] - pr[node])
            
        pr = new_pr
        if diff < tol:
            print(f"PageRank converged at iteration {i+1}")
            break
            
    return pr

print("Loading data...")
with open('data/processed/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)["pages"]
    
with open('data/processed/interlinks.json', 'r', encoding='utf-8') as f:
    interlinks = json.load(f)["graph"]

print("Computing PageRank...")
pr_scores = compute_pagerank(interlinks)

word_counts = {p['title']: p.get('word_count', 0) for p in metadata}

import re

results = []
for title, pr in pr_scores.items():
    wc = word_counts.get(title, 0)
    
    # 1. Log-scale both PageRank and Word Count to flatten extreme outliers
    pr_factor = math.log10((pr * 1000000) + 1)
    wc_factor = math.log10(max(wc, 1))
    
    # 2. Extract content richness from metadata
    # We find the matching metadata dict for this title
    page_meta = next((p for p in metadata if p['title'] == title), {})
    has_infobox = 1 if page_meta.get('infobox') else 0
    img_count = len(page_meta.get('images', []))
    
    # Add a multiplier if it has an infobox, and up to a 50% boost for having images
    content_multiplier = 1.0 + (has_infobox * 0.5) + (min(img_count, 50) * 0.01)
    
    # 3. Penalize meta-pages, editions, and specific game versions
    is_meta = bool(re.search(r'(?i)edition\b|\b1\.\d+|\b\d{2}w\d+[a-z]?|launcher|mojang|protocol|tracker', title))
    penalty = 0.3 if is_meta else 1.0
    
    # Calculate final score
    final_score = pr_factor * wc_factor * content_multiplier * penalty
    scaled_pr = pr * 100000 
    
    results.append({
        'title': title,
        'pr_score': scaled_pr,
        'word_count': wc,
        'final_score': final_score,
        'images': img_count,
        'infobox': has_infobox
    })
    
results.sort(key=lambda x: x['final_score'], reverse=True)

print(f"\n{'#':<4} {'Title':<41} {'Score':>7} {'PR':>6} {'Words':>8} {'Img':>3} {'IB':>2}")
print("-" * 80)
for i, res in enumerate(results[:50], 1):
    print(f"{i:<4} {res['title'][:40]:<41} {res['final_score']:>7.2f} {res['pr_score']:>6.0f} {res['word_count']:>8} {res['images']:>3} {res['infobox']:>2}")
