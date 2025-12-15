#!/usr/bin/env python3
# scraper.py - crawler + dataset generator (universal, supports --make-dataset)
# Zastępuje wcześniejszą wersję: dodaje generowanie sft_dataset.jsonl po pobraniu

import argparse
import asyncio
import aiohttp
import aiofiles
import hashlib
import json
import logging
import mimetypes
import os
import random
import re
import time
from datetime import datetime
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
]

BLACKLIST_KEYWORDS = [
    'discord', 'twitter.com', 'facebook.com', 'instagram.com', 'linkedin.com', 'youtube.com',
    'mailto:', 'tel:', 'github.com', 'rss', 'googletagmanager', 'google-analytics', 'gstatic',
    'cdn.', 'cdn-', 'doubleclick', 'analytics', 'pinterest', 'tiktok', 'spotify', 'slack', 'meet'
]

SENTENCE_END_RE = re.compile(r'([.!?]+)\s+')
HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', flags=re.IGNORECASE)
SRC_RE = re.compile(r'src=["\']([^"\']+)["\']', flags=re.IGNORECASE)
JSON_LIKE_RE = re.compile(r'^\s*[{[]', re.M)
YAML_LIKE_RE = re.compile(r'^[\s-]*[a-zA-Z0-9_]+:\s', re.M)
PERM_RE = re.compile(r'\b([a-zA-Z_]+\.)+[a-zA-Z0-9_*{}-]+\b')

def safe_filename_from_url(url: str, maxlen: int = 180) -> str:
    p = urlparse(url)
    base = (p.netloc + p.path).strip('/')
    safe = ''.join(c if (c.isalnum() or c in '-._') else '_' for c in base)
    return safe[:maxlen] or hashlib.md5(url.encode()).hexdigest()[:10]

def sha1_hex(b: bytes) -> str:
    import hashlib as _h
    return _h.sha1(b).hexdigest()

def guess_extension(content_type: str, url: str = None) -> str:
    if content_type:
        ct = content_type.split(';',1)[0].lower()
        if 'html' in ct: return 'html'
        if 'pdf' in ct: return 'pdf'
        if 'json' in ct: return 'json'
        if 'xml' in ct: return 'xml'
        if 'plain' in ct: return 'txt'
        ext = mimetypes.guess_extension(ct)
        if ext: return ext.lstrip('.')
    if url:
        p = urlparse(url).path
        ext = os.path.splitext(p)[1]
        if ext: return ext.lstrip('.')
    return 'html'

def ensure_dirs(out_dir: str):
    os.makedirs(os.path.join(out_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'meta'), exist_ok=True)

async def fetch_one(session: aiohttp.ClientSession, url: str, user_agent: str, proxy: str = None, max_retries: int = 3, timeout: int = 30):
    """
    Robust fetch:
    - if 'Session is closed' -> create a temporary session and retry once
    - handle common aiohttp exceptions with retries/backoff
    - always return dict {'status': ..., 'content': bytes, 'headers': ...} or error info
    """
    attempt = 0
    while True:
        ua = user_agent if attempt == 0 else random.choice(USER_AGENTS)
        headers = {
            'User-Agent': ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,pl;q=0.8',
            'Referer': f'https://{urlparse(url).netloc}/',
        }
        try:
            # primary path: use provided session
            async with session.get(url, headers=headers, timeout=timeout, proxy=proxy) as resp:
                status = resp.status
                content = await resp.read()
                return {'status': status, 'content': content, 'headers': dict(resp.headers)}
        except RuntimeError as re:
            # common message: "Session is closed"
            msg = str(re)
            logging.warning("RuntimeError while fetching %s: %s", url, msg)
            if 'Session is closed' in msg:
                # try one-time fetch with a temporary session
                logging.info("Opening temporary session for %s due to closed session", url)
                try:
                    tmp_timeout = aiohttp.ClientTimeout(total=timeout)
                    async with aiohttp.ClientSession(timeout=tmp_timeout) as tmp_sess:
                        async with tmp_sess.get(url, headers=headers, timeout=timeout, proxy=proxy) as resp:
                            status = resp.status
                            content = await resp.read()
                            return {'status': status, 'content': content, 'headers': dict(resp.headers)}
                except Exception as e2:
                    attempt += 1
                    if attempt > max_retries:
                        logging.warning("Failed fetch %s after %d attempts (temp session): %s", url, attempt, e2)
                        return {'status': None, 'error': str(e2), 'content': b''}
                    backoff = (2 ** (attempt-1)) + random.random()
                    logging.info("Temp-session fetch error for %s -> retry in %.1fs (attempt %d)", url, backoff, attempt)
                    await asyncio.sleep(backoff)
                    continue
            else:
                # unexpected runtime error -> treat as other exception
                attempt += 1
                if attempt > max_retries:
                    return {'status': None, 'error': msg, 'content': b''}
                backoff = (2 ** (attempt-1)) + random.random()
                await asyncio.sleep(backoff)
                continue
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, aiohttp.ServerDisconnectedError, asyncio.TimeoutError) as e:
            attempt += 1
            if attempt > max_retries:
                logging.warning("Failed fetch %s after %d attempts: %s", url, attempt, e)
                return {'status': None, 'error': str(e), 'content': b''}
            backoff = (2 ** (attempt-1)) + random.random()
            logging.info("Fetch error for %s -> retry in %.1fs (attempt %d)", url, backoff, attempt)
            await asyncio.sleep(backoff)
            continue
        except Exception as e:
            # catch-all for other unexpected exceptions
            attempt += 1
            logging.warning("Unexpected error fetching %s: %s (attempt %d)", url, e, attempt)
            if attempt > max_retries:
                return {'status': None, 'error': str(e), 'content': b''}
            await asyncio.sleep((2 ** (attempt-1)) + random.random())
            continue


async def fetch_with_playwright(url: str, proxy: str = None, timeout: int = 30):
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except Exception as e:
        logging.error("Playwright not installed or import failed: %s", e)
        return {'status': None, 'error': 'playwright_missing', 'content': b''}
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, proxy={"server": proxy} if proxy else None)
            page = await browser.new_page()
            await page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
            await page.goto(url, wait_until='networkidle', timeout=timeout*1000)
            content = await page.content()
            await browser.close()
            return {'status': 200, 'content': content.encode('utf-8'), 'headers': {}}
    except Exception as e:
        logging.warning("Playwright fetch failed: %s", e)
        return {'status': None, 'error': str(e), 'content': b''}

async def save_response(out_dir: str, url: str, raw_bytes: bytes, headers: dict, status: int = None, seed: str = None):
    fn_base = safe_filename_from_url(url)
    id_hash = sha1_hex(url.encode('utf-8'))
    ext = guess_extension(headers.get('content-type', ''), url)
    domain = urlparse(url).netloc or 'unknown'
    domain_dir = os.path.join(out_dir, 'raw', domain)
    os.makedirs(domain_dir, exist_ok=True)
    fname = f"{fn_base}_{id_hash[:8]}.{ext}"
    path = os.path.join(domain_dir, fname)
    async with aiofiles.open(path, 'wb') as f:
        await f.write(raw_bytes)
    meta = {
        'id': id_hash,
        'url': url,
        'domain': domain,
        'saved_path': os.path.relpath(path, out_dir),
        'fetched_at': datetime.utcnow().isoformat() + 'Z',
        'content_type': headers.get('content-type'),
        'http_status': status,
        'size_bytes': len(raw_bytes),
        'headers': headers,
        'seed': seed
    }
    meta_path = os.path.join(out_dir, 'meta', f"{id_hash}.json")
    async with aiofiles.open(meta_path, 'w', encoding='utf-8') as fm:
        await fm.write(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta

def extract_links(html: str, base_url: str):
    found = set()
    for m in HREF_RE.findall(html):
        if not m: continue
        if m.startswith('#'): continue
        try:
            joined = urljoin(base_url, m)
            found.add(joined.split('#')[0])
        except Exception:
            pass
    for m in SRC_RE.findall(html):
        if not m: continue
        try:
            joined = urljoin(base_url, m)
            found.add(joined.split('#')[0])
        except Exception:
            pass
    return list(found)

def is_blacklisted_url(u: str) -> bool:
    low = u.lower()
    for k in BLACKLIST_KEYWORDS:
        if k in low: return True
    return False

async def crawl_site(seed_url: str, session: aiohttp.ClientSession, user_agent: str, args) -> list:
    parsed_seed = urlparse(seed_url)
    root_domain = parsed_seed.netloc
    segments = [s for s in parsed_seed.path.split('/') if s]
    prefix = '/' + segments[0] + '/' if segments else '/'
    scope = args.site_scope
    to_visit = [(seed_url, 0)]
    visited = set()
    discovered = []
    while to_visit and len(discovered) < args.max_pages_per_seed:
        url, depth = to_visit.pop(0)
        if url in visited: continue
        if depth > args.crawl_depth: continue
        visited.add(url)
        if is_blacklisted_url(url):
            logging.info("Skipping blacklisted URL during crawl: %s", url)
            continue
        logging.info("Crawling (%d/%d) %s", len(discovered)+1, args.max_pages_per_seed, url)
        res = await fetch_one(session, url, user_agent, proxy=args.proxy, max_retries=2, timeout=args.timeout)
        content = b''
        if res.get('status') == 200 and res.get('content'):
            content = res['content']
            try:
                meta = await save_response(args.out, url, content, res.get('headers', {}), status=res.get('status'), seed=seed_url)
                discovered.append(meta['url'])
            except Exception as e:
                logging.warning("Saving during crawl failed: %s", e)
        else:
            if args.use_playwright:
                pr = await fetch_with_playwright(url, proxy=args.proxy, timeout=args.timeout)
                if pr.get('status') == 200 and pr.get('content'):
                    content = pr['content']
                    try:
                        meta = await save_response(args.out, url, content, pr.get('headers', {}), status=200, seed=seed_url)
                        discovered.append(meta['url'])
                    except Exception as e:
                        logging.warning("Saving PW during crawl failed: %s", e)
        if content:
            try:
                text = content.decode('utf-8', errors='ignore')
            except Exception:
                text = ''
            links = extract_links(text, url)
            for link in links:
                if len(discovered) + len(to_visit) >= args.max_pages_per_seed: break
                try:
                    p = urlparse(link)
                except Exception:
                    continue
                if p.scheme not in ('http', 'https', ''): continue
                if is_blacklisted_url(link): continue
                if p.netloc and p.netloc != root_domain: continue
                if scope == 'prefix' and not p.path.startswith(prefix): continue
                if link not in visited:
                    to_visit.append((link, depth+1))
        await asyncio.sleep(random.uniform(0.2, 0.8))
    return discovered

async def worker_fetch_and_save(url: str, session: aiohttp.ClientSession, out_dir: str, user_agent: str, semaphore: asyncio.Semaphore, proxy: str = None, use_playwright: bool = False, seed: str = None):
    async with semaphore:
        await asyncio.sleep(random.uniform(0.2, 1.0))
        res = await fetch_one(session, url, user_agent, proxy=proxy)
        if res.get('status') == 200 and res.get('content'):
            meta = await save_response(out_dir, url, res['content'], res.get('headers', {}), status=res.get('status'), seed=seed)
            logging.info("Saved %s -> %s", url, meta['saved_path'])
            return meta
        if res.get('status') in (403, 429) or (res.get('status') and res['status'] >= 500) or (not res.get('status') and use_playwright):
            logging.info("Status %s for %s — attempting fallbacks", res.get('status'), url)
            res2 = await fetch_one(session, url, random.choice(USER_AGENTS), proxy=proxy, max_retries=1)
            if res2.get('status') == 200 and res2.get('content'):
                meta = await save_response(out_dir, url, res2['content'], res2.get('headers', {}), status=res2.get('status'), seed=seed)
                logging.info("Saved after UA rotation %s -> %s", url, meta['saved_path'])
                return meta
            if use_playwright:
                pr = await fetch_with_playwright(url, proxy=proxy)
                if pr.get('status') == 200 and pr.get('content'):
                    meta = await save_response(out_dir, url, pr.get('content'), pr.get('headers', {}), status=200, seed=seed)
                    logging.info("Saved via Playwright %s -> %s", url, meta['saved_path'])
                    return meta
        logging.warning("Skipping %s (status=%s, error=%s)", url, res.get('status'), res.get('error'))
        return None

# ---------- HTML -> sections ----------
def html_to_sections_fast(html_text):
    soup = BeautifulSoup(html_text, "lxml")
    for s in soup(["script", "style", "noscript"]): s.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    headings = soup.find_all(re.compile("^h[1-6]$"))
    sections = []
    if headings:
        for h in headings:
            heading_text = h.get_text(" ", strip=True)
            content_parts = []
            for el in h.next_siblings:
                if getattr(el, "name", None) and re.match("^h[1-6]$", el.name): break
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
    if not parts: return [text]
    sentences = []
    i = 0
    while i < len(parts):
        if i+1 < len(parts):
            sent = parts[i].strip() + parts[i+1]
            sentences.append(sent.strip())
            i += 2
        else:
            tail = parts[i].strip()
            if tail: sentences.append(tail)
            i += 1
    return [s for s in sentences if s]

def chunk_text_by_sentences(text, max_chars=1200, overlap_chars=200):
    sentences = split_into_sentences(text)
    chunks = []
    cur = ""
    for s in sentences:
        if not cur:
            cur = s; continue
        if len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur.strip())
            overlap = cur[-overlap_chars:] if overlap_chars > 0 else ""
            cur = (overlap + " " + s).strip() if overlap else s
        else:
            cur = cur + " " + s
    if cur: chunks.append(cur.strip())
    return chunks

def extract_summary_from_text(text, max_chars=500, max_sentences=3):
    sents = split_into_sentences(text)
    if not sents: return text.strip()[:max_chars]
    out = " ".join(sents[:max_sentences])
    if len(out) > max_chars: return out[:max_chars].rsplit(" ", 1)[0] + "..."
    return out

# ---------- generic completion builders ----------
def build_permission_completion(p):
    parts = []
    parts.append(f"Uprawnienie `{p}` — krótki opis i typowe zastosowania:")
    parts.append("- Opis: dotyczy konkretnej funkcji/config lub grupy uprawnień.")
    parts.append("- Typowe użycia: nadawanie adminowi/VIP, testy, debug.")
    parts.append("")
    parts.append("Przykład (LuckPerms):")
    parts.append(f"/lp user <nick> permission set {p} true")
    return "\n".join(parts)

def build_config_snippet_completion(code_text):
    m_id = re.search(r'\bid[:]\s*([A-Za-z0-9_"\']+)', code_text, re.I)
    id_val = m_id.group(1).strip('"\'') if m_id else "ExampleItem"
    yaml_example = [
        f"id: {id_val}",
        "displayname: \"Example Item\"",
        "material: DIAMOND",
        "commands:",
        "  - 'say %player% used ExampleItem'",
        "cooldown: 10",
    ]
    explanation = [
        "Poprawny przykład konfiguracji (YAML) oraz krótkie wyjaśnienie pól:",
        "- `id` — identyfikator elementu",
        "- `commands` — lista komend uruchamianych przy użyciu",
        "- `cooldown` — czas odnowienia (sekundy)",
    ]
    return "\n".join(explanation + [""] + yaml_example)

def generate_examples_from_sections(sections, source_meta, max_chunk_chars=1200, overlap_chars=200, min_chunk_len=60):
    results = []
    for sec in sections:
        text = sec.get("text","").strip()
        heading = sec.get("heading","")
        codes = sec.get("codes",[])
        if text:
            chunks = chunk_text_by_sentences(text, max_chars=max_chunk_chars, overlap_chars=overlap_chars)
            for chunk in chunks:
                if len(chunk) < min_chunk_len: continue
                summary = extract_summary_from_text(chunk, max_chars=480, max_sentences=3)
                perms = PERM_RE.findall(chunk)
                has_cmds = bool(re.search(r'\b/', chunk))
                completion_parts = [summary]
                if perms:
                    unique_perms = sorted(set(perms))[:4]
                    for p in unique_perms:
                        completion_parts.append("")
                        completion_parts.append(build_permission_completion(p))
                if has_cmds:
                    completion_parts.append("")
                    completion_parts.append("Przykładowa komenda/przykładowe użycie:")
                    completion_parts.append("/give %player% diamond 1")
                completion = "\n".join(completion_parts)
                prompt = (
                    "Jesteś ekspertem technicznym. Na podstawie poniższego fragmentu strony, podaj krótkie podsumowanie najważniejszych informacji i — jeśli sensowne — praktyczny przykład (komenda, fragment konfiguracji). Odpowiedz po polsku.\n\n"
                    f"Kontekst (źródło: {source_meta}, sekcja: {heading}):\n{chunk}\n\n"
                    "Pytanie: Podsumuj i podaj przykład praktycznego zastosowania.\n\nOdpowiedź:"
                )
                results.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]})
        for code in codes:
            if not code.strip(): continue
            code_text = code.strip()
            perms = re.findall(r'\b([a-zA-Z_]+\.[a-zA-Z0-9_.*{}\-\:]+)\b', code_text)
            if perms:
                for p in sorted(set(perms))[:3]:
                    prompt = (
                        "Jesteś ekspertem technicznym. Na podstawie poniższego fragmentu (uprawnienia/permission), wyjaśnij krótko co to robi i podaj przykład zastosowania. Odpowiedz po polsku.\n\n"
                        f"Kontekst (źródło: {source_meta}, sekcja: {heading}):\n{p}\n\n"
                        "Pytanie: Co robi ten fragment i jak go użyć?\n\nOdpowiedź:"
                    )
                    completion = build_permission_completion(p)
                    results.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]})
                continue
            if JSON_LIKE_RE.search(code_text) or YAML_LIKE_RE.search(code_text):
                prompt = (
                    "Jesteś ekspertem technicznym. Na podstawie fragmentu konfiguracji poniżej, opisz co robi i podaj poprawny przykład (zachowaj format YAML/JSON). Odpowiedz po polsku.\n\n"
                    f"Kontekst (źródło: {source_meta}, sekcja: {heading}):\n{code_text}\n\n"
                    "Pytanie: Co to jest i jak tego użyć?\n\nOdpowiedź:"
                )
                completion = build_config_snippet_completion(code_text)
                results.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]})
                continue
            prompt = (
                "Jesteś ekspertem technicznym. Na podstawie fragmentu kodu/komend poniżej, wytłumacz krok po kroku jak wykonać te polecenia lub jak zastosować ten fragment konfiguracji. Odpowiedz po polsku.\n\n"
                f"Kontekst (źródło: {source_meta}, sekcja: {heading}):\n{code_text}\n\n"
                "Pytanie: Jak wykonać/zaaplikować powyższe komendy/konfigurację?\n\nOdpowiedź:"
            )
            completion = extract_summary_from_text(code_text, max_chars=500, max_sentences=4)
            results.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]})
    return results

def load_saved_files_and_make_dataset(out_dir, out_file="sft_dataset.jsonl", max_chunk_chars=1200, overlap_chars=200, min_chunk_len=80, limit_files=0):
    raw_dir = os.path.join(out_dir, "raw")
    if not os.path.isdir(raw_dir):
        logging.error("Nie znaleziono katalogu raw: %s", raw_dir)
        return 0
    files = []
    for domain in sorted(os.listdir(raw_dir)):
        d = os.path.join(raw_dir, domain)
        if not os.path.isdir(d): continue
        for fn in sorted(os.listdir(d)):
            files.append(os.path.join(d, fn))
    if limit_files and limit_files>0:
        files = files[:limit_files]
    written = 0
    seen = set()
    with open(out_file, "w", encoding="utf-8") as fw:
        for fpath in files:
            try:
                b = open(fpath, "rb").read()
                txt = b.decode("utf-8", errors="ignore")
                sections = html_to_sections_fast(txt)
                meta_src = os.path.basename(fpath)
                examples = generate_examples_from_sections(sections, meta_src, max_chunk_chars=max_chunk_chars, overlap_chars=overlap_chars, min_chunk_len=min_chunk_len)
                for ex in examples:
                    key = (ex["messages"][0]["content"][:400], ex["messages"][1]["content"][:400])
                    if key in seen: continue
                    seen.add(key)
                    fw.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    written += 1
            except Exception as e:
                logging.warning("Error processing %s: %s", fpath, e)
    logging.info("Wygenerowano %d rekordów do %s", written, out_file)
    return written

# ---------- main pipeline ----------
async def main_async(args):
    ensure_dirs(args.out)
    all_urls = []
    seeds = list(args.urls) if args.urls else []
    connector = aiohttp.TCPConnector(limit_per_host=args.per_host_limit, limit=0)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if args.crawl and seeds:
            for s in seeds:
                logging.info("Starting crawl for seed: %s", s)
                discovered = await crawl_site(s, session, args.user_agent, args)
                all_urls.append(s)
                all_urls.extend(discovered)
                await asyncio.sleep(random.uniform(0.5, 1.5))
        else:
            all_urls.extend(seeds)

    seen = set(); dedup = []
    for u in all_urls:
        if u not in seen:
            seen.add(u); dedup.append(u)
    all_urls = dedup[:args.max_urls] if args.max_urls and args.max_urls > 0 else dedup

    logging.info("Total URLs to fetch: %d", len(all_urls))
    if not all_urls:
        logging.info("No URLs to process. Exiting.")
        return

    sem = asyncio.Semaphore(args.global_concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [worker_fetch_and_save(url, session, args.out, args.user_agent, sem, proxy=args.proxy, use_playwright=args.use_playwright, seed=url) for url in all_urls]
        results = []
        for coro in asyncio.as_completed(tasks):
            try:
                r = await coro
                if r: results.append(r)
            except Exception as e:
                logging.error("Worker error: %s", e)

    index_path = os.path.join(args.out, 'collection.jsonl')
    async with aiofiles.open(index_path, 'w', encoding='utf-8') as idxf:
        for item in results:
            await idxf.write(json.dumps(item, ensure_ascii=False) + '\n')
    logging.info("Saved collection index to %s", index_path)

    if args.make_dataset:
        out_file = args.dataset_out or os.path.join(args.out, "sft_dataset.jsonl")
        written = load_saved_files_and_make_dataset(args.out, out_file=out_file, max_chunk_chars=args.max_chunk_chars, overlap_chars=args.overlap_chars, min_chunk_len=args.min_chunk_len, limit_files=args.limit_files)
        logging.info("Dataset generation finished: %d records -> %s", written, out_file)

def parse_args():
    p = argparse.ArgumentParser(description="Crawler + dataset generator (generic) for SFT")
    p.add_argument("--urls", nargs='+', help="Seed URLs to fetch and crawl", required=True)
    p.add_argument("--out", type=str, default="./raw_sources", help="Output dir")
    p.add_argument("--global-concurrency", type=int, default=4, help="Global concurrent fetches")
    p.add_argument("--per-host-limit", type=int, default=3, help="Per-host connection limit (aiohttp)")
    p.add_argument("--timeout", type=int, default=60, help="Per-request total timeout (seconds)")
    p.add_argument("--user-agent", type=str, default=USER_AGENTS[0], help="User-Agent header")
    p.add_argument("--proxy", type=str, default=None, help="Optional proxy")
    p.add_argument("--max-urls", type=int, default=0, help="Cap total urls (0 = no cap)")
    p.add_argument("--use-playwright", action='store_true', help="Use Playwright fallback for JS pages")
    p.add_argument("--max-pages-per-seed", type=int, default=200, help="Max pages to discover per seed")
    p.add_argument("--crawl-depth", type=int, default=3, help="Max crawl depth from seed")
    p.add_argument("--site-scope", choices=['prefix','domain'], default='prefix', help="Crawl scope: prefix/domain")
    p.add_argument("--crawl", action='store_true', help="Enable crawling from seed URLs (default enabled)")
    p.add_argument("--make-dataset", action='store_true', help="Generate sft JSONL after fetch")
    p.add_argument("--dataset-out", type=str, default=None, help="Path for output dataset (default: <out>/sft_dataset.jsonl)")
    p.add_argument("--max_chunk_chars", type=int, default=1200)
    p.add_argument("--overlap_chars", type=int, default=200)
    p.add_argument("--min_chunk_len", type=int, default=80)
    p.add_argument("--limit_files", type=int, default=0)
    p.add_argument("--workers", type=int, default=0, help="(unused for fetch; reserved)")
    return p.parse_args()

def main():
    args = parse_args()
    if args.urls and not args.crawl:
        args.crawl = True
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logging.info("Interrupted by user")

if __name__ == "__main__":
    main()