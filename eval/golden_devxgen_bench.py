"""Real-world golden benchmark — OmniFuse vs synaptic-memory on a live xgen
retrieval collection (dev-xgen), end to end:

  1. download a collection's chunks from the xgen retrieval API,
  2. generate one natural question per document (LLM, from the chunk body only),
  3. run OmniFuse.retrieve() and synaptic's own FTS runner on the same corpus,
     queries and qrels, and report MRR/nDCG/Recall@10.

Nothing here ships the private corpus or any secret — credentials come from the
environment, and the built golden set stays local. This is the reproducer for
``eval/results/golden_devxgen.json``.

    export XGEN_BASE=https://dev-xgen.x2bee.com
    export XGEN_EMAIL=...   XGEN_PASSWORD=...        # xgen login (password hashed here)
    export OPENAI_API_KEY=... [OPENAI_MODEL=gpt-4o-mini]
    export SYNAPTIC_REPO=/path/to/synaptic-memory     # for the head-to-head
    python eval/golden_devxgen_bench.py --collection-id 42 --max-docs 220 --num-queries 215
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, os, re, ssl, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

_CTX = ssl.create_default_context(); _CTX.check_hostname = False; _CTX.verify_mode = ssl.CERT_NONE
_META = re.compile(r"<Document-Metadata>.*?</Document-Metadata>", re.S)


def _req(base, method, path, tok=None, body=None, retries=3):
    for i in range(retries):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(base + path, data=data, method=method)
        r.add_header("Content-Type", "application/json")
        if tok:
            r.add_header("Authorization", f"Bearer {tok}")
        try:
            with urllib.request.urlopen(r, context=_CTX, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5)


def download_corpus(base, email, password, coll_id, max_docs):
    tok = _req(base, "POST", "/api/auth/login",
               body={"email": email, "password": hashlib.sha256(password.encode()).hexdigest()})["access_token"]
    cols = _req(base, "GET", "/api/retrieval/collections", tok)
    cn = next(c["collection_name"] for c in cols if c["id"] == coll_id)
    e = quote(cn)
    ids, page = [], 1
    while len(ids) < max_docs:
        d = _req(base, "GET", f"/api/retrieval/collections/{e}/documents?page={page}&page_size=20", tok)
        docs = d.get("documents", [])
        if not docs:
            break
        ids += [(x["document_id"], x["file_name"]) for x in docs]
        if not d.get("pagination", {}).get("has_next"):
            break
        page += 1
    corpus = []
    for did, fname in ids[:max_docs]:
        doc = _req(base, "GET", f"/api/retrieval/collections/{e}/documents/{did}", tok)
        for ch in doc.get("chunks", []):
            txt = (ch.get("chunk_text") or "").strip()
            if len(txt) >= 40:
                corpus.append({"id": ch["chunk_id"], "title": fname, "text": txt, "doc_id": did})
    return corpus


_SYS = ("너는 검색 벤치마크용 질문 생성기다. 주어진 한국어 문단만 보고 답할 수 있는, 그 문단의 "
        "구체적 사실을 묻는 자연스러운 질문 1개를 만든다. 문단 문장을 그대로 복사하지 말고 다르게 "
        "표현하고, '이 문단/위 내용' 같은 메타 표현은 금지한다. JSON {\"q\":\"질문\"} 만 출력.")
_BAD = ("이 문단", "위 내용", "해당 문서", "위 문단", "본 문서", "제시된")


def _gen_q(text, key, model):
    body = {"model": model, "temperature": 0.4, "max_tokens": 200,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": _SYS}, {"role": "user", "content": text[:3000]}]}
    r = urllib.request.Request("https://api.openai.com/v1/chat/completions",
                               data=json.dumps(body).encode(), method="POST")
    r.add_header("Content-Type", "application/json"); r.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return json.loads(json.loads(resp.read().decode())["choices"][0]["message"]["content"]).get("q", "").strip()
    except Exception:
        return ""


def make_golden(corpus, key, model, num_queries):
    best = {}
    for ch in corpus:
        b = _META.sub("", ch["text"]).strip()
        if 300 <= len(b) <= 2500 and (ch["doc_id"] not in best or len(b) > len(_META.sub("", best[ch["doc_id"]]["text"]))):
            best[ch["doc_id"]] = ch
    cands = list(best.values())[: max(num_queries * 2, num_queries)]
    pairs = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for ch, q in zip(cands, ex.map(lambda c: _gen_q(_META.sub("", c["text"]).strip(), key, model), cands)):
            if len(q) >= 10 and "?" in q and not any(b in q for b in _BAD):
                pairs.append((ch["id"], q))
            if len(pairs) >= num_queries:
                break
    return {
        "corpus": {ch["id"]: {"title": ch["title"], "text": ch["text"]} for ch in corpus},
        "queries": {f"q{i}": q for i, (cid, q) in enumerate(pairs)},
        "relevant_docs": {f"q{i}": [cid] for i, (cid, q) in enumerate(pairs)},
    }


def evaluate(golden_path, synaptic_repo):
    sys.path.insert(0, str(synaptic_repo))
    sys.path.insert(0, str(Path(synaptic_repo) / "tests" / "benchmark"))
    from metrics import BenchmarkResult
    from eval.run_all import DatasetConfig, run_public_dataset
    from omnifuse import build_inmemory

    data = json.load(open(golden_path, encoding="utf-8"))
    raw = data["corpus"]
    corpus = [(d, v.get("title", ""), v.get("text", "")) for d, v in raw.items()]
    ql = [(q, data["queries"][q], set(data["relevant_docs"][q])) for q in data["queries"]]

    of = build_inmemory([], [], [{"id": d, "title": t, "text": x} for d, t, x in corpus])
    b = BenchmarkResult()
    for qid, text, rel in ql:
        seen = []
        for cid in [c.id for c, _ in of.retrieve(text, limit=20)]:
            if cid not in seen:
                seen.append(cid)
        b.add(query_id=qid, query=text, retrieved=seen[:10], relevant=rel, k=10)
    o = b.summary()
    r = asyncio.run(run_public_dataset(DatasetConfig(name="golden", path=Path(golden_path), quick=True),
                                       embedder=None, reranker=None))
    print(f"synaptic FTS : MRR={r.mrr:.4f} nDCG={r.ndcg:.4f} R@10={r.r_at_k:.4f}")
    print(f"OmniFuse     : MRR={o['mrr']:.4f} nDCG={o['mean_ndcg@k']:.4f} R@10={o['mean_recall@k']:.4f}")
    print(f"winner: {'OmniFuse' if o['mrr'] > r.mrr else 'synaptic'} (Δ MRR {o['mrr']-r.mrr:+.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection-id", type=int, default=42)
    ap.add_argument("--max-docs", type=int, default=220)
    ap.add_argument("--num-queries", type=int, default=215)
    ap.add_argument("--golden", default="golden_devxgen.json")
    a = ap.parse_args()
    gp = Path(a.golden)
    if not gp.exists():
        base = os.environ["XGEN_BASE"]; email = os.environ["XGEN_EMAIL"]; pw = os.environ["XGEN_PASSWORD"]
        key = os.environ["OPENAI_API_KEY"]; model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        print("downloading corpus …", flush=True)
        corpus = download_corpus(base, email, pw, a.collection_id, a.max_docs)
        print(f"  {len(corpus)} chunks; generating {a.num_queries} questions …", flush=True)
        json.dump(make_golden(corpus, key, model, a.num_queries), open(gp, "w", encoding="utf-8"), ensure_ascii=False)
    evaluate(gp, Path(os.environ["SYNAPTIC_REPO"]))


if __name__ == "__main__":
    main()
