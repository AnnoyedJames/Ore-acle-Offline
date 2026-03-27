import re
with open('scripts/eval/generate_questionset.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Imports
code = code.replace('import argparse', 'import argparse\nimport base64')

# 2. QAPair
code = code.replace('    relevant_links: List[str]\n    difficulty: str',
                    '    relevant_links: List[str]\n    relevant_images: List[str] = []\n    difficulty: str')

# 3. SYSTEM_PROMPT
old_sp = """- Look at the provided "Outgoing Links" array. For each question, identify up to 3 links from that array that are highly relevant to the specific question topic to serve as gold-standard retrieval targets. Return them as full URLs (e.g. "https://minecraft.wiki/w/Diamond")."""
new_sp = old_sp + """\n- If relevant images are provided in the user prompt array visually, you may identify their "image_hash" (up to 2) that are highly relevant to the question. Return them in the "relevant_images" list. If no image is relevant, leave it empty."""
code = code.replace(old_sp, new_sp)

# 4. build_user_prompt
code = code.replace('def build_user_prompt(page: dict, interlinks: Dict[str, List[str]]) -> str:',
                    'def build_user_prompt(page: dict, interlinks: Dict[str, List[str]], images_to_show: List[dict]) -> str:')

old_bp_str = """Outgoing Links context: {json.dumps(ranked_outgoing)}\n\n--- FULL WIKI PAGE TEXT START ---"""
new_bp_str = """Outgoing Links context: {json.dumps(ranked_outgoing)}\nImages context (hashes): {json.dumps([img['image_hash'] for img in images_to_show])}\n\n--- FULL WIKI PAGE TEXT START ---"""
code = code.replace(old_bp_str, new_bp_str)

old_bp_json = '{"items": [{"question": "...", "answer": "...", "relevant_links": ["url1", "url2"], "difficulty": "easy|medium|hard"}]}'
new_bp_json = '{"items": [{"question": "...", "answer": "...", "relevant_links": ["url1"], "relevant_images": ["hash1"], "difficulty": "easy|medium|hard"}]}'
code = code.replace(old_bp_json, new_bp_json)

# 5. generate_qa_pairs
old_gq = 'def generate_qa_pairs(page: dict, interlinks: Dict[str, List[str]], model: str) -> List[dict]:\n    user_prompt = build_user_prompt(page, interlinks)'
new_gq = '''def generate_qa_pairs(page: dict, interlinks: Dict[str, List[str]], image_metadata: Dict[str, list], model: str) -> List[dict]:
    # Find matching images
    page_filename = Path(page.get("file_path", "")).name
    images_for_page = image_metadata.get(page_filename, [])[:10]  # Cap at 10 to avoid payload explosion
    
    user_prompt = build_user_prompt(page, interlinks, images_for_page)'''
code = code.replace(old_gq, new_gq)

old_gq2 = '''    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]'''
new_gq2 = '''    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    
    user_content = [{"type": "text", "text": user_prompt}]
    
    # Add images
    for img in images_for_page:
        try:
            with open(img["file_path"], "rb") as image_file:
                b64_img = base64.b64encode(image_file.read()).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{b64_img}"}
                })
        except Exception as e:
            logger.warning(f"Could not load image {img['file_path']}: {e}")
    
    messages.append({"role": "user", "content": user_content})'''
code = code.replace(old_gq2, new_gq2)

old_gq3 = '''            "difficulty": item.get("difficulty", "medium"),
            "relevant_links": safe_links,
            "source_page": page.get("title", ""),'''
new_gq3 = '''            "difficulty": item.get("difficulty", "medium"),
            "relevant_links": safe_links,
            "relevant_images": item.get("relevant_images", []),
            "source_page": page.get("title", ""),'''
code = code.replace(old_gq3, new_gq3)


# 6. main
old_main = '''        with open('data/processed/interlinks.json', 'r', encoding='utf-8') as f:
            interlinks = json.load(f)["graph"]
    except Exception as e:'''
new_main = '''        with open('data/processed/interlinks.json', 'r', encoding='utf-8') as f:
            interlinks = json.load(f)["graph"]
            
        with open('data/processed/image_metadata.json', 'r', encoding='utf-8') as f:
            raw_image_data = json.load(f)["images"]
            
        image_metadata = defaultdict(list)
        for img in raw_image_data:
            for sp in img.get("source_pages", []):
                image_metadata[sp].append(img)
    except Exception as e:'''
code = code.replace(old_main, new_main)

old_main2 = 'pairs = generate_qa_pairs(page, interlinks, model=args.model)'
new_main2 = 'pairs = generate_qa_pairs(page, interlinks, image_metadata, model=args.model)'
code = code.replace(old_main2, new_main2)

with open('scripts/eval/generate_questionset.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('Done!')
