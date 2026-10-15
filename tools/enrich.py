"""Polish training captions with an OpenAI-compatible service configured at runtime."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re


def caption_key(sentences):
    captions = [sentence["raw"].strip().lower() for sentence in sentences]
    return " | ".join(sorted(captions)), captions


def polish_captions(captions, prompt_template, client_factory, model):
    """Return five validated captions, or None so a failed request can be retried."""
    prompt = prompt_template.replace("{captions_json}", json.dumps(captions))
    try:
        with client_factory() as client:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
        content = response.choices[0].message.content or ""
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match is None:
            return None
        polished = json.loads(match.group(0))
        if not isinstance(polished, list) or len(polished) != 5:
            return None
        if not all(isinstance(item, dict) and isinstance(item.get("raw"), str)
                   and item["raw"].strip() for item in polished):
            return None
        return polished
    except Exception:
        return None


def save_json(path, data):
    """Replace a JSON file only after a complete temporary file has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def enrich_document(data, cache, prompt_template, client_factory, model, workers=50, save_cache=None):
    """Preserve image metadata and original non-training captions."""
    tasks = {}
    for item in data["images"]:
        if item.get("split") != "train":
            continue
        key, captions = caption_key(item["sentences"])
        entry = cache.setdefault(key, {"original_list": captions, "polished_list": []})
        if not entry.get("polished_list"):
            tasks[key] = captions

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(polish_captions, captions, prompt_template, client_factory, model): key
            for key, captions in tasks.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            polished = future.result()
            if polished is not None:
                cache[key]["polished_list"] = polished
            if save_cache is not None and completed % 50 == 0:
                save_cache(cache)

    fallback_groups = set()
    for item in data["images"]:
        if item.get("split") != "train":
            continue
        key, captions = caption_key(item["sentences"])
        polished = cache[key].get("polished_list")
        if polished:
            item["sentences"] = polished
        else:
            item["sentences"] = [{"raw": caption} for caption in captions]
            fallback_groups.add(key)
    return data, len(fallback_groups)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--prompt", type=Path,
                        default=Path(__file__).resolve().parents[1] / "prompts" / "enrich.txt")
    parser.add_argument("--model", default=os.environ.get("ENRICH_MODEL", "deepseek-ai/DeepSeek-V3"))
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive.")
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        parser.error("Set OPENAI_API_KEY and OPENAI_BASE_URL in your environment.")
    try:
        from openai import OpenAI
    except ImportError:
        parser.error("Install the optional requirements-enrich.txt dependencies first.")

    def client_factory():
        return OpenAI(api_key=api_key, base_url=base_url)

    data = json.loads(args.input.read_text(encoding="utf-8-sig"))
    cache_path = args.cache or args.output.with_name(args.output.stem + ".cache.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    prompt_template = args.prompt.read_text(encoding="utf-8")
    try:
        output, fallback_count = enrich_document(
            data, cache, prompt_template, client_factory, args.model, args.workers,
            save_cache=lambda current: save_json(cache_path, current),
        )
    finally:
        save_json(cache_path, cache)
    save_json(args.output, output)
    print(f"Saved {len(output['images'])} image records; {fallback_count} training caption groups used raw captions.")


if __name__ == "__main__":
    main()
