import json
import math
import re
from collections import defaultdict

def compute_pagerank(interlinks, damping_factor=0.85, max_iterations=50, tol=1e-6):
    nodes = set(interlinks.keys())
    for targets in interlinks.values():
        nodes.update(targets)
    
    N = len(nodes)
    if N == 0: return {}

    pr = {node: 1.0 / N for node in nodes}
    out_degree = {node: len(targets) for node, targets in interlinks.items()}
    
    incoming = defaultdict(list)
    for src, targets in interlinks.items():
        for target in targets:
            incoming[target].append(src)
            
    dangling_nodes = [n for n in nodes if out_degree.get(n, 0) == 0]

    for _ in range(max_iterations):
        new_pr = {}
        dangling_sum = sum(pr[n] for n in dangling_nodes)
        base_pr = (1.0 - damping_factor) / N + damping_factor * (dangling_sum / N)
        diff = 0.0
        for node in nodes:
            sum_in = sum(pr[inc] / out_degree[inc] for inc in incoming[node])
            new_pr[node] = base_pr + damping_factor * sum_in
            diff += abs(new_pr[node] - pr[node])
        pr = new_pr
        if diff < tol:
            break
    return pr

def main():
    print("Loading data... (This takes a few seconds)")
    with open('data/processed/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)["pages"]
    with open('data/processed/interlinks.json', 'r', encoding='utf-8') as f:
        interlinks = json.load(f)["graph"]

    pr_scores = compute_pagerank(interlinks)
    word_counts = {p['title']: p.get('word_count', 0) for p in metadata}

    def print_stage(title, entries, headers, format_str, header_format_str):
        print("\n\n" + "="*80)
        print(f" STAGE: {title} ".center(80, "="))
        print("="*80 + "\n")
        print(header_format_str.format(*headers))
        print("-" * 80)
        for i, row in enumerate(entries[:15], 1):
            print(format_str.format(i, *row))

    # STAGE 1: Raw PageRank Only
    s1 = []
    for p in metadata:
        t = p['title']
        pr = pr_scores.get(t, 0)
        s1.append((t[:40], pr * 100000))
    s1.sort(key=lambda x: x[1], reverse=True)
    print_stage("1. Raw PageRank Only", s1, ("#", "Title", "Raw PR Score"), "{:<4} {:<45} {:>15.2f}", "{:<4} {:<45} {:>15}")

    # STAGE 2: PageRank * Linear Word Count
    s2 = []
    for p in metadata:
        t = p['title']
        pr = pr_scores.get(t, 0)
        wc = word_counts.get(t, 0)
        score = pr * wc 
        s2.append((t[:40], score, pr * 100000, wc))
    s2.sort(key=lambda x: x[1], reverse=True)
    print_stage("2. PageRank × Linear Word Count", s2, ("#", "Title", "Score", "PR", "Words"), "{:<4} {:<35} {:>12.2f} {:>10.2f} {:>10}", "{:<4} {:<35} {:>12} {:>10} {:>10}")

    # STAGE 3: Log10(PageRank) * Log10(Word Count)
    s3 = []
    for p in metadata:
        t = p['title']
        pr = pr_scores.get(t, 0)
        wc = word_counts.get(t, 0)
        pr_factor = math.log10((pr * 1000000) + 1)
        wc_factor = math.log10(max(wc, 1))
        score = pr_factor * wc_factor
        s3.append((t[:40], score, pr_factor, wc_factor))
    s3.sort(key=lambda x: x[1], reverse=True)
    print_stage("3. Log10(PageRank) × Log10(Word Count)", s3, ("#", "Title", "Log Score", "Log(PR)", "Log(Words)"), "{:<4} {:<35} {:>12.2f} {:>10.2f} {:>10.2f}", "{:<4} {:<35} {:>12} {:>10} {:>10}")

    # STAGE 4: Category/Metadata Modifiers (Final)
    s4 = []
    for p in metadata:
        t = p['title']
        pr = pr_scores.get(t, 0)
        wc = word_counts.get(t, 0)
        pr_factor = math.log10((pr * 1000000) + 1)
        wc_factor = math.log10(max(wc, 1))
        
        has_infobox = 1 if p.get('infobox') else 0
        img_count = len(p.get('images', []))
        content_multiplier = 1.0 + (has_infobox * 0.5) + (min(img_count, 50) * 0.01)
        
        is_meta = bool(re.search(r'(?i)edition\b|\b1\.\d+|\b\d{2}w\d+[a-z]?|launcher|mojang|protocol|tracker', t))
        penalty = 0.3 if is_meta else 1.0
        
        score = pr_factor * wc_factor * content_multiplier * penalty
        s4.append((t[:35], score, pr_factor*wc_factor, content_multiplier, penalty))
    s4.sort(key=lambda x: x[1], reverse=True)
    print_stage("4. Log Bounds + Metadata Multis + Penalties (Final)", s4, ("#", "Title", "Final Score", "Base Log", "Multi", "Penalty"), "{:<4} {:<35} {:>11.2f} {:>10.2f} {:>8.2f} {:>8.1f}", "{:<4} {:<35} {:>11} {:>10} {:>8} {:>8}")
    
    print("\n")

if __name__ == '__main__':
    main()