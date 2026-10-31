import argparse
import fnmatch
import hashlib
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path

import numpy as np

WIKI_ROOT = Path(__file__).parent / "wiki"
INDEX_DIR = WIKI_ROOT / ".index"
EMBED_MODEL = "mlx-community/embeddinggemma-2-bf16"

QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: {title} | text: "

CHUNK_CHARS = 800
EMBED_BATCH = 8
FIND_MIN_RATIO = 0.4

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class WikiError(Exception):
    pass


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# ---------- 경로 ----------


def _resolve(path: str) -> tuple[str, Path]:
    """위키 안의 상대경로(`.md` 보장)와 실제 경로를 돌려준다. 밖으로 나가는 경로는 거부한다."""
    rel = _nfc(path.strip()).replace("\\", "/")
    if not rel:
        raise WikiError("경로가 비었습니다")
    if not rel.endswith(".md"):
        rel += ".md"
    full = (WIKI_ROOT / rel).resolve()
    root = WIKI_ROOT.resolve()
    if not full.is_relative_to(root):
        raise WikiError(f"위키 밖의 경로입니다: {path}")
    parts = full.relative_to(root).parts
    if any(p.startswith(".") for p in parts):
        raise WikiError(f"숨김 경로는 쓸 수 없습니다: {path}")
    return "/".join(parts), full


def _all_files() -> dict[str, Path]:
    files = {}
    for p in sorted(WIKI_ROOT.rglob("*.md")):
        rel = p.relative_to(WIKI_ROOT)
        if any(part.startswith(".") for part in rel.parts):
            continue
        files[_nfc(rel.as_posix())] = p
    return files


# ---------- 임베딩 ----------


@cache
def _load_model():
    import mlx.core as mx  # noqa: F401  (import 순서: mlx 먼저)
    from mlx_vlm.embedding_loader import load_embedding_model
    from mlx_vlm.utils import get_model_path
    from transformers import AutoTokenizer

    path = get_model_path(EMBED_MODEL)
    return load_embedding_model(path), AutoTokenizer.from_pretrained(path)


class Embedder:
    """EmbeddingGemma 2. 출력은 정규화된 768차원이라 내적이 곧 코사인이다."""

    def _embed(self, texts: list[str]) -> np.ndarray:
        import mlx.core as mx

        model, tokenizer = _load_model()
        out = []
        for i in range(0, len(texts), EMBED_BATCH):
            inputs = tokenizer(
                texts[i : i + EMBED_BATCH],
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="np",
            )
            emb = model(**{k: mx.array(v) for k, v in inputs.items()}).text_embeds
            out.append(np.array(emb.astype(mx.float32)))  # bf16은 그대로 쓰고 결과만 변환
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([QUERY_PREFIX + t for t in texts])

    def embed_documents(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return self._embed([DOC_PREFIX.format(title=t) + x for t, x in pairs])


# ---------- 청킹 ----------


def _chunk(path: str, text: str) -> list[dict]:
    """헤딩 단위로 나누고 긴 섹션은 문단 기준으로 CHUNK_CHARS 근처에서 자른다."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            sections.append((m.group(2).strip(), [line]))
        else:
            sections[-1][1].append(line)

    chunks = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        buf = ""
        for para in re.split(r"\n\s*\n", body):
            if buf and len(buf) + len(para) > CHUNK_CHARS:
                chunks.append({"path": path, "heading": heading, "text": buf})
                buf = ""
            buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            chunks.append({"path": path, "heading": heading, "text": buf})
    return chunks


def _title(chunk: dict) -> str:
    stem = Path(chunk["path"]).stem
    return f"{stem} - {chunk['heading']}" if chunk["heading"] else stem


# ---------- 인덱스 ----------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_index() -> tuple[dict, np.ndarray]:
    meta_path, emb_path = INDEX_DIR / "meta.json", INDEX_DIR / "embeddings.npy"
    empty = {"model": EMBED_MODEL, "files": {}, "chunks": []}
    if not (meta_path.exists() and emb_path.exists()):
        return empty, np.zeros((0, 768), np.float32)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        emb = np.load(emb_path)
    except (OSError, ValueError):
        return empty, np.zeros((0, 768), np.float32)
    if meta.get("model") != EMBED_MODEL or len(meta["chunks"]) != len(emb):
        return empty, np.zeros((0, 768), np.float32)
    return meta, emb


def _save_index(meta: dict, emb: np.ndarray) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", emb)
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )


def sync(full: bool = False) -> int:
    """바뀐·새 파일만 다시 임베딩하고 사라진 파일의 청크를 지운다. 다시 계산한 파일 수를 돌려준다."""
    meta, emb = ({"model": EMBED_MODEL, "files": {}, "chunks": []}, np.zeros((0, 768), np.float32)) \
        if full else _load_index()
    files = _all_files()
    hashes = {rel: _sha(p) for rel, p in files.items()}
    stale = {rel for rel, h in hashes.items() if meta["files"].get(rel) != h}
    gone = set(meta["files"]) - set(files)
    if not stale and not gone:
        return 0

    keep = [i for i, c in enumerate(meta["chunks"]) if c["path"] not in stale | gone]
    chunks = [meta["chunks"][i] for i in keep]
    emb = emb[keep]

    new_chunks = []
    for rel in sorted(stale):
        new_chunks += _chunk(rel, files[rel].read_text(encoding="utf-8"))
    if new_chunks:
        new_emb = Embedder().embed_documents([(_title(c), c["text"]) for c in new_chunks])
        emb = np.concatenate([emb, new_emb])
        chunks += new_chunks

    _save_index({"model": EMBED_MODEL, "files": hashes, "chunks": chunks}, emb)
    return len(stale)


def warm() -> int:
    """통화 전에 인덱스를 맞추고 임베딩 모델을 올려 둔다. 문서가 없으면 아무것도 하지 않는다."""
    if not _all_files():
        return 0
    n = sync()
    _load_model()
    return n


# ---------- 검색 ----------


def search(query: str, k: int = 5) -> list[dict]:
    """임베딩 검색. 파일마다 가장 점수가 높은 청크 하나만 남긴다."""
    sync()
    meta, emb = _load_index()
    if not len(emb):
        return []
    q = Embedder().embed_queries([query])[0]
    scores = emb @ q
    results, seen = [], set()
    for i in np.argsort(-scores):
        c = meta["chunks"][i]
        if c["path"] in seen:
            continue
        seen.add(c["path"])
        snippet = re.sub(r"\s+", " ", c["text"]).strip()
        results.append(
            {
                "path": c["path"],
                "heading": c["heading"],
                "score": round(float(scores[i]), 4),
                "snippet": snippet[:200],
            }
        )
        if len(results) >= k:
            break
    return results


def find(query: str, k: int = 10) -> list[dict]:
    """파일명 검색. glob 문자가 있으면 fnmatch, 아니면 부분일치 우선 + 퍼지 매칭."""
    q = _nfc(query.strip()).lower()
    if not q:
        return []
    scored = []
    for rel in _all_files():
        low = rel.lower()
        stem = Path(rel).stem.lower()
        if any(ch in q for ch in "*?["):
            if fnmatch.fnmatch(low, q) or fnmatch.fnmatch(low, q + ".md"):
                scored.append((1.0, rel))
            continue
        if q in low:
            scored.append((1.0 + (q == stem) + len(q) / len(low), rel))
            continue
        ratio = max(
            SequenceMatcher(None, q, stem).ratio(), SequenceMatcher(None, q, low).ratio()
        )
        if ratio >= FIND_MIN_RATIO:
            scored.append((ratio, rel))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [{"path": rel, "score": round(s, 3)} for s, rel in scored[:k]]


# ---------- CRUD ----------


def list_pages() -> list[str]:
    return list(_all_files())


def read(path: str) -> str:
    rel, full = _resolve(path)
    if not full.is_file():
        raise WikiError(f"없는 문서입니다: {rel}")
    return full.read_text(encoding="utf-8")


def create(path: str, content: str) -> str:
    rel, full = _resolve(path)
    if full.exists():
        raise WikiError(f"이미 있는 문서입니다: {rel}")
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    sync()
    return rel


def update(path: str, content: str, append: bool = False) -> str:
    rel, full = _resolve(path)
    if not full.is_file():
        raise WikiError(f"없는 문서입니다: {rel}")
    if append:
        old = full.read_text(encoding="utf-8")
        content = old.rstrip("\n") + "\n\n" + content
    full.write_text(content, encoding="utf-8")
    sync()
    return rel


def delete(path: str) -> str:
    rel, full = _resolve(path)
    if not full.is_file():
        raise WikiError(f"없는 문서입니다: {rel}")
    full.unlink()
    # 빈 하위 폴더는 치운다 (위키 루트는 남김).
    parent = full.parent
    while parent != WIKI_ROOT.resolve() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent
    sync()
    return rel


# ---------- 에이전트 도구 ----------


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_STR = {"type": "string"}
WIKI_TOOLS = [
    _tool("wiki_search", "위키를 의미로 검색한다 (임베딩)", {"query": _STR, "k": {"type": "integer"}}, ["query"]),
    _tool("wiki_find", "위키 파일명을 검색한다 (부분일치, glob, 오타 허용)", {"query": _STR, "k": {"type": "integer"}}, ["query"]),
    _tool("wiki_read", "위키 문서를 읽는다", {"path": _STR}, ["path"]),
    _tool("wiki_create", "위키 문서를 새로 만든다", {"path": _STR, "content": _STR}, ["path", "content"]),
    _tool(
        "wiki_update",
        "위키 문서를 고친다. append가 true면 끝에 덧붙인다",
        {"path": _STR, "content": _STR, "append": {"type": "boolean"}},
        ["path", "content"],
    ),
    _tool("wiki_delete", "위키 문서를 지운다", {"path": _STR}, ["path"]),
    _tool("wiki_list", "위키 문서 목록을 본다", {}, []),
]


READ_TOOL_NAMES = {"wiki_search", "wiki_find", "wiki_read"}
WIKI_READ_TOOLS = [t for t in WIKI_TOOLS if t["function"]["name"] in READ_TOOL_NAMES]


def run_read_tool(name: str, arguments: dict | str) -> str:
    """읽기 전용 도구만 실행한다. 통화 중 상담원에게 쓰기 권한은 주지 않는다."""
    if name not in READ_TOOL_NAMES:
        return f"오류: 쓸 수 없는 도구: {name}"
    return run_tool(name, arguments)


def run_tool(name: str, arguments: dict | str) -> str:
    """도구 호출을 실행하고 모델에게 돌려줄 짧은 텍스트를 만든다. 오류도 텍스트로 돌려준다."""
    try:
        a = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
        if name == "wiki_search":
            hits = search(a["query"], a.get("k", 5))
            return "\n".join(
                f"{h['path']}" + (f" # {h['heading']}" if h["heading"] else "")
                + f" ({h['score']}): {h['snippet']}"
                for h in hits
            ) or "결과 없음"
        if name == "wiki_find":
            hits = find(a["query"], a.get("k", 10))
            return "\n".join(h["path"] for h in hits) or "결과 없음"
        if name == "wiki_read":
            return read(a["path"])
        if name == "wiki_create":
            return f"만듦: {create(a['path'], a['content'])}"
        if name == "wiki_update":
            return f"고침: {update(a['path'], a['content'], bool(a.get('append')))}"
        if name == "wiki_delete":
            return f"지움: {delete(a['path'])}"
        if name == "wiki_list":
            return "\n".join(list_pages()) or "문서 없음"
        return f"알 수 없는 도구: {name}"
    except WikiError as e:
        return f"오류: {e}"
    except (KeyError, ValueError, TypeError) as e:
        return f"오류: 잘못된 인자 ({type(e).__name__}: {e})"
    except Exception as e:  # 모델 로드·다운로드·MLX 오류가 통화를 끊지 않게 한다
        return f"오류: {type(e).__name__}: {e}"


# ---------- CLI ----------


def _content(args) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    return sys.stdin.read()


def main() -> None:
    p = argparse.ArgumentParser(description="상담원 위키 (./wiki) 관리와 검색")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="문서 목록")
    sub.add_parser("reindex", help="임베딩 인덱스를 처음부터 다시 만든다")
    for name in ("read", "delete"):
        sub.add_parser(name).add_argument("path")
    for name in ("create", "update"):
        sp = sub.add_parser(name)
        sp.add_argument("path")
        sp.add_argument("--file", help="내용을 읽을 파일 (없으면 stdin)")
        if name == "update":
            sp.add_argument("--append", action="store_true")
    for name in ("search", "find"):
        sp = sub.add_parser(name)
        sp.add_argument("query")
        sp.add_argument("-k", type=int, default=5 if name == "search" else 10)
    args = p.parse_args()

    try:
        if args.cmd == "list":
            print("\n".join(list_pages()))
        elif args.cmd == "reindex":
            print(f"{sync(full=True)}개 문서 인덱싱")
        elif args.cmd == "read":
            print(read(args.path))
        elif args.cmd == "delete":
            print(f"지움: {delete(args.path)}")
        elif args.cmd == "create":
            print(f"만듦: {create(args.path, _content(args))}")
        elif args.cmd == "update":
            print(f"고침: {update(args.path, _content(args), args.append)}")
        elif args.cmd == "search":
            for h in search(args.query, args.k):
                head = f" # {h['heading']}" if h["heading"] else ""
                print(f"{h['score']:.3f}  {h['path']}{head}\n       {h['snippet']}")
        elif args.cmd == "find":
            for h in find(args.query, args.k):
                print(f"{h['score']:.2f}  {h['path']}")
    except WikiError as e:
        raise SystemExit(f"오류: {e}")


if __name__ == "__main__":
    main()
