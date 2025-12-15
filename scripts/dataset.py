#!/usr/bin/env python3
# dataset_parallel.py - improved: generates informative completions (SSOMAR-aware)

import os
import glob
import json
import re
import argparse
import time
from pathlib import Path
from html import unescape
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, as_completed

SENTENCE_END_RE = re.compile(r'([.!?]+)\s+')

def read_file(path):
    b = open(path, "rb").read()
    for enc in ("utf-8", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            pass
    return b.decode("utf-8", "ignore")

def html_to_sections_fast(html_text):
    soup = BeautifulSoup(html_text, "lxml")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    headings = soup.find_all(re.compile("^h[1-6]$"))
    sections = []
    if headings:
        for i, h in enumerate(headings):
            heading_text = h.get_text(" ", strip=True)
            content_parts = []
            for el in h.next_siblings:
                if getattr(el, "name", None) and re.match("^h[1-6]$", el.name):
                    break
                content_parts.append(str(el))
            content_html = "".join(content_parts)
            tmp = BeautifulSoup(content_html, "lxml")
            codes = [c.get_text("\n") for c in tmp.find_all(["pre", "code"])]
            text = tmp.get_text("\n")
            sections.append({"heading": heading_text, "text": text.strip(), "codes": codes})
    else:
        text = soup.get_text("\n")
        codes = [c.get_text("\n") for c in soup.find_all(["pre", "code"])]
        sections.append({"heading": title or "content", "text": text.strip(), "codes": codes})
    return sections

def split_into_sentences(text):
    parts = SENTENCE_END_RE.split(text)
    if not parts:
        return [text]
    sentences = []
    i = 0
    while i < len(parts):
        if i+1 < len(parts):
            sent = parts[i].strip() + parts[i+1]
            sentences.append(sent.strip())
            i += 2
        else:
            tail = parts[i].strip()
            if tail:
                sentences.append(tail)
            i += 1
    return [s for s in sentences if s]

def chunk_text_by_sentences(text, max_chars=1200, overlap_chars=200):
    sentences = split_into_sentences(text)
    chunks = []
    cur = ""
    for s in sentences:
        if not cur:
            cur = s
            continue
        if len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur.strip())
            overlap = cur[-overlap_chars:] if overlap_chars > 0 else ""
            cur = (overlap + " " + s).strip() if overlap else s
        else:
            cur = cur + " " + s
    if cur:
        chunks.append(cur.strip())
    return chunks

def extract_summary_from_text(text, max_chars=500, max_sentences=3):
    sents = split_into_sentences(text)
    if not sents:
        return text.strip()[:max_chars]
    out = " ".join(sents[:max_sentences])
    if len(out) > max_chars:
        return out[:max_chars].rsplit(" ", 1)[0] + "..."
    return out

# heurystyki
PERM_RE = re.compile(r'\b([a-zA-Z_]+\.)+[a-zA-Z0-9_*{}-]+\b')
YAML_LIKE_RE = re.compile(r'^[\s-]*[a-zA-Z0-9_]+:\s', re.M)
JSON_LIKE_RE = re.compile(r'^\s*[{[]', re.M)

CMD_RE = re.compile(r'\b(luckperms|/lp|/permissionsex|/pex|/perm)\b', re.I)
def detect_permission_patterns(text):
    return list(set(PERM_RE.findall(text)))

def has_permission_token(text):
    return bool(re.search(r'\b[eE]i\.nocd\b|\b\.nocd\b|\bexecutableitems\b', text))

def detect_yaml_json(text):
    if JSON_LIKE_RE.search(text):
        return "json"
    if YAML_LIKE_RE.search(text):
        return "yaml"
    return None

def build_permission_completion(perm_text):
    # perm_text example: ei.nocd.{id} or ei.nocd.*
    p = perm_text.strip()
    # normalize tokens
    desc = []
    desc.append(f"Uprawnienie `{p}` dotyczy tej konfiguracji/permission w pluginie SSOMAR.")
    # give specific cases
    if p.endswith(".*") or p.endswith(".*`") :
        desc.append("`*` oznacza wildcard (wszystkie elementy tej kategorii). Daje to uprawnienie globalne.")
    if "{id}" in p or re.search(r'\{.+\}', p):
        desc.append("Jeżeli w nazwie występuje `{id}`, to należy zastąpić go konkretnym ID ExecutableItem (np. `Excalibur`, `super_pickaxe`).")
    desc.append("Typowe użycia:")
    desc.append("- Nadanie adminom prawa do używania przedmiotów bez cooldownu.")
    desc.append("- Nadanie VIP-om wybranych przedmiotów z brakiem limitu.")
    examples = []
    examples.append("Przykład nadania uprawnienia przez LuckPerms:")
    examples.append("`/lp user <nick> permission set ei.nocd.Excalibur true`")
    examples.append("Przykład ustawienia wildcard:")
    examples.append("`/lp group admin permission set ei.nocd.* true`")
    return "\\n".join(desc + [""] + examples)

def build_config_snippet_completion(code_text):
    # tworzymy przykładowy, poprawny fragment YAML/JSON bazując na wykrytych kluczach
    lines = []
    # spróbuj wyciągnąć klucz id/material/commands
    m_id = re.search(r'\bid[:]\s*([A-Za-z0-9_"\']+)', code_text, re.I)
    if m_id:
        id_val = m_id.group(1).strip('"\'')
    else:
        id_val = "ExampleItem"
    # generate yaml example and short explanation
    explanation = (
        f"Przykładowy poprawny fragment konfiguracji ExecutableItem o ID `{id_val}` "
        "oraz krótkie wyjaśnienie pól:"
    )
    yaml_example = [
        f"id: {id_val}",
        "displayname: \"Super Kilof\"",
        "material: DIAMOND_PICKAXE",
        "lore:",
        "  - \"Specjalny kilof\"",
        "commands:",
        "  - 'give %player% diamond 1'",
        "cooldown: 10",
        "permissions:",
        "  - 'ei.use.ExampleItem'"
    ]
    usage = [
        "Wyjaśnienie:",
        "- `id` — identyfikator przedmiotu, używany w uprawnieniach i odwołaniach.",
        "- `commands` — lista komend uruchamianych przy użyciu itemu. `%player%` zostanie zastąpione graczem.",
        "- `cooldown` — czas odnowienia w sekundach.",
        "- `permissions` — wymagane permisje do użycia itemu."
    ]
    return explanation + "\n\n" + "\n".join(yaml_example) + "\n\n" + "\n".join(usage)

def generate_examples_from_section(section, source_meta, max_chunk_chars=1200, overlap_chars=200, min_chunk_len=60):
    results = []
    text = section.get("text", "").strip()
    heading = section.get("heading", "")
    codes = section.get("codes", [])
    if not text and not codes:
        return results

    text_chunks = chunk_text_by_sentences(text, max_chars=max_chunk_chars, overlap_chars=overlap_chars) if text else []
    for i, chunk in enumerate(text_chunks):
        if len(chunk) < min_chunk_len:
            continue
        summary = extract_summary_from_text(chunk, max_chars=480, max_sentences=3)
        # build a richer completion: short explanation + example(s)
        completion_parts = []
        completion_parts.append(summary)
        # if found permission-like tokens, explain them
        perms = re.findall(r'\b[a-zA-Z_]+\.[a-zA-Z0-9_.*{}\-\:]+\b', chunk)
        detected_perms = [p for p in perms if p.count('.')>=1]
        if detected_perms:
            for p in detected_perms[:4]:
                completion_parts.append("")
                completion_parts.append(build_permission_completion(p))
        # small helpful example
        if "command" in chunk.lower() or "komend" in chunk.lower() or "commands" in chunk.lower():
            completion_parts.append("")
            completion_parts.append("Przykładowa komenda / zastosowanie:")
            completion_parts.append("/give %player% diamond 1")
        completion = "\n".join(completion_parts).strip()
        if not completion:
            completion = summary
        prompt = (
            "Jesteś ekspertem administracji serwerów Minecraft i wtyczek. "
            "Udziel rzeczowej, pomocnej odpowiedzi po polsku. "
            "Użyj poniższego kontekstu i jeśli to możliwe podaj konkretny przykład komendy lub fragment konfiguracji.\n\n"
            f"Kontekst (źródło: {source_meta}):\n{chunk}\n\n"
            "Pytanie: Podsumuj najważniejsze informacje z powyższego kontekstu i podaj przykład praktycznego zastosowania (komenda/konfiguracja) jeśli to sensowne.\n\nOdpowiedź:"
        )
        results.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}], "meta": {"source": source_meta, "heading": heading, "chunk_id": i}})

    # process code blocks (config/code)
    for j, code in enumerate(codes):
        if len(code.strip()) < 4:
            continue
        # detect permission token like ei.nocd.{id} or ei.nocd.*
        perms = re.findall(r'\b[eE]i\.[a-zA-Z0-9_.\{\}\*\-]+\b', code)
        if perms:
            for p in perms:
                prompt = (
                    "Jesteś ekspertem administracji serwerów Minecraft. Na podstawie fragmentu konfiguracji poniżej, wyjaśnij krótko co robi i jakie ma opcje. Odpowiedz po polsku.\n\n"
                    f"Kontekst (źródło: {source_meta}):\n{p}\n\n"
                    "Pytanie: Co robi ten fragment konfiguracji i jakie są typowe zastosowania?\n\nOdpowiedź:"
                )
                completion = build_permission_completion(p)
                results.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}], "meta": {"source": source_meta, "heading": heading, "code_block": j}})
            continue

        # if code looks like YAML/JSON -> produce example snippet + explanation
        yamljson = detect_yaml_json(code)
        if yamljson:
            prompt = (
                "Jesteś ekspertem administracji serwerów Minecraft i wtyczek. Na podstawie fragmentu konfiguracji poniżej wyjaśnij co robi i zaproponuj poprawny przykład konfiguracji (zachowaj format YAML/JSON). Odpowiedz po polsku.\n\n"
                f"Kontekst (źródło: {source_meta}):\n{code}\n\n"
                "Pytanie: Co to jest i jak tego użyć?\n\nOdpowiedź:"
            )
            completion = build_config_snippet_completion(code)
            results.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}], "meta": {"source": source_meta, "heading": heading, "code_block": j}})
            continue

        # fallback: short summary of code
        prompt = (
            "Jesteś ekspertem administracji serwerów Minecraft. Na podstawie fragmentu konfiguracji poniżej, wyjaśnij krótko co robi i jakie ma opcje. Odpowiedz po polsku.\n\n"
            f"Kontekst (źródło: {source_meta}):\n{code}\n\n"
            "Pytanie: Co robi ten fragment konfiguracji i jakie są typowe zastosowania?\n\nOdpowiedź:"
        )
        completion = extract_summary_from_text(code, max_chars=500, max_sentences=4)
        results.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}], "meta": {"source": source_meta, "heading": heading, "code_block": j}})

    return results

def process_file_worker(path, args):
    try:
        text = read_file(path)
        ext = os.path.splitext(path)[1].lower()
        if ext in [".md", ".txt"]:
            raw = text
            parts = re.split(r'\n#{1,6}\s+', raw)
            sections = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                first_line = p.splitlines()[0].strip()
                # try to extract code fences
                codes = re.findall(r'```(?:[a-zA-Z0-9]+)?\n(.*?)\n```', p, re.S)
                sections.append({"heading": first_line[:80], "text": re.sub(r'```(?:[a-zA-Z0-9]+)?\n(.*?)\n```', '', p, flags=re.S).strip(), "codes": codes})
        else:
            sections = html_to_sections_fast(text)
        src = os.path.basename(path)
        examples = []
        for sec in sections:
            exs = generate_examples_from_section(sec, src,
                                                max_chunk_chars=args.max_chunk_chars,
                                                overlap_chars=args.overlap_chars,
                                                min_chunk_len=args.min_chunk_len)
            if exs:
                examples.extend(exs)
        return (path, examples, None)
    except Exception as e:
        return (path, [], str(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="raw_sources", help="Katalog z pobranymi plikami (zawiera /raw/ i /meta/)")
    parser.add_argument("--out_file", type=str, default="sft_dataset.jsonl")
    parser.add_argument("--max_chunk_chars", type=int, default=1200)
    parser.add_argument("--overlap_chars", type=int, default=200)
    parser.add_argument("--min_chunk_len", type=int, default=80)
    parser.add_argument("--limit_files", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0, help="Ilość workerów (0 = cpu_count())")
    parser.add_argument("--max_file_size_mb", type=int, default=0, help="Pomijaj pliki większe niż X MB (0 = off)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir_raw = raw_dir / "raw"
    if not raw_dir_raw.exists():
        print("Katalog z pobranymi plikami nie istnieje:", raw_dir_raw)
        return

    patterns = [str(raw_dir_raw / "*" / "*")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(files)
    if args.limit_files and args.limit_files > 0:
        files = files[:args.limit_files]

    if args.max_file_size_mb and args.max_file_size_mb > 0:
        max_bytes = args.max_file_size_mb * 1024 * 1024
        files = [f for f in files if os.path.getsize(f) <= max_bytes]

    print(f"Znaleziono {len(files)} plików do przetworzenia w {raw_dir_raw}")
    start_all = time.perf_counter()

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 2)
    seen = set()
    written = 0

    with open(args.out_file, "w", encoding="utf-8") as fw, ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_file_worker, f, args): f for f in files}
        for fut in as_completed(futures):
            path = futures[fut]
            t0 = time.perf_counter()
            try:
                p, examples, err = fut.result()
                if err:
                    print(f"ERR processing {p}: {err}")
                    continue
                for ex_item in examples:
                    key = (ex_item["messages"][0]["content"][:400], ex_item["messages"][1]["content"][:400])
                    if key in seen:
                        continue
                    seen.add(key)
                    out = {"messages": ex_item["messages"]}
                    fw.write(json.dumps(out, ensure_ascii=False) + "\n")
                    written += 1
                t1 = time.perf_counter()
                print(f"[OK] {os.path.basename(p)} -> {len(examples)} exs (czas pliku: {t1-t0:.2f}s), zapisanych: {written}")
            except Exception as e:
                print(f"FUTURE ERR for {path}: {e}")

    total_time = time.perf_counter() - start_all
    print(f"Zapisano {written} unikatowych przykładów do: {args.out_file}")
    print(f"Czas całkowity: {total_time:.2f}s")

if __name__ == "__main__":
    main()