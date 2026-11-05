"""GEPA 후보: system.md와 편집 가능한 위키 문서의 줄 목록, 그리고 계보 사이의 병합. 시나리오를 모른다.

후보 = {문서 키: [(줄 ID, 글)]}. 문서 키는 "system.md" 또는 "wiki/<위키 안 경로>".
줄 ID는 실행 전체에서 유일하고, 고치지 않은 줄은 자식이 부모의 ID를 그대로 물려받는다(learner.PromptLines).
그래서 두 계보가 공통 조상에서 갈라진 뒤 각자 고친 줄을 ID로 찾아 합칠 수 있다.
"""
import shutil
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from learner import PROMPT_MIN_CHARS, PromptLines

SYSTEM = "system.md"
WIKI = "wiki/"
Rows = list[tuple[str, str]]


def num(cid: str) -> int:
    return int(cid[1:])


@dataclass
class Candidate:
    cid: str
    docs: dict[str, Rows]
    parents: list[str] = field(default_factory=list)
    kind: str = "seed"            # seed | reflect | merge
    summary: str = ""
    target: str | None = None     # 이 후보를 만든 반성이 고치도록 한 문서(순환 선택)
    iteration: int = 0

    def text(self, doc: str) -> str:
        return "\n".join(l for _, l in self.docs.get(doc, [])).strip()

    def wiki_keys(self) -> list[str]:
        return sorted(k for k in self.docs if k.startswith(WIKI))

    def wiki_chars(self) -> int:
        return sum(len(self.text(k)) for k in self.wiki_keys())

    def ids(self) -> set[str]:
        return {i for rows in self.docs.values() for i, _ in rows}

    def to_json(self) -> dict:
        return {"cid": self.cid, "docs": {k: [list(r) for r in v] for k, v in self.docs.items()}, "parents": self.parents,
                "kind": self.kind, "summary": self.summary, "target": self.target, "iteration": self.iteration}

    @classmethod
    def from_json(cls, d: dict) -> "Candidate":
        return cls(d["cid"], {k: [(i, t) for i, t in v] for k, v in d["docs"].items()}, list(d.get("parents", [])),
                   d.get("kind", "seed"), d.get("summary", ""), d.get("target"), d.get("iteration", 0))


def size_error(c: Candidate, system_max: int, wiki_max: int) -> str | None:
    """크기 상한(사실 검사가 아니다). 문제가 없으면 None."""
    n = len(c.text(SYSTEM))
    if n < PROMPT_MIN_CHARS:
        return f"system.md가 {n}자입니다. 최소 {PROMPT_MIN_CHARS}자여야 합니다"
    if n > system_max:
        return f"system.md가 {n}자입니다. {system_max}자 이하로 줄이세요"
    if (w := c.wiki_chars()) > wiki_max:
        return f"편집 가능한 위키 문서 전체가 {w}자입니다. {wiki_max}자 이하로 줄이세요"
    return None


def seed_candidate(cid: str, system_text: str, wiki_docs: dict[str, str], used: set[str]) -> Candidate:
    """현재 저장소의 system.md와 편집 가능한 위키 문서로 첫 후보를 만든다. 새 줄 ID는 used에 등록된다."""
    docs: dict[str, Rows] = {SYSTEM: PromptLines(system_text, used=used).rows}
    for rel, text in sorted(wiki_docs.items()):
        docs[WIKI + rel] = PromptLines(text, used=used).rows
    return Candidate(cid, docs)


def materialize_wiki(c: Candidate, dest: Path, readonly_docs: dict[str, str], patterns: list[str]) -> Path:
    """후보의 위키 폴더를 만든다: 읽기 전용 문서 복사본 + .readonly + 후보의 편집 문서. 이미 있으면 그대로 둔다."""
    marker = dest / ".ready"
    if marker.exists():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel, text in readonly_docs.items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(text, encoding="utf-8")
    (dest / ".readonly").write_text("".join(p + "\n" for p in patterns), encoding="utf-8")
    for key in c.wiki_keys():
        if c.docs[key]:  # 줄을 모두 지운 문서는 없는 문서
            path = dest / key[len(WIKI):]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(c.text(key) + "\n", encoding="utf-8")
    marker.write_text("", encoding="utf-8")
    return dest


# ───────────── 계보 ─────────────

def ancestors(cands: dict[str, Candidate], cid: str) -> set[str]:
    """cid 자신과 모든 조상."""
    seen: set[str] = set()
    stack = [cid]
    while stack:
        c = stack.pop()
        if c not in seen:
            seen.add(c)
            stack += cands[c].parents
    return seen


def common_ancestor(cands: dict[str, Candidate], a: str, b: str) -> str | None:
    """a·b의 가장 최근 공통 조상. 한쪽이 다른 쪽의 조상이거나 같으면(합칠 것이 없다) None."""
    A, B = ancestors(cands, a), ancestors(cands, b)
    if a in B or b in A:
        return None
    common = A & B
    return max(common, key=num) if common else None


# ───────────── 병합 ─────────────

@dataclass
class Hunk:
    i1: int          # 공통 조상 쪽 구간 [i1, i2)을 rows로 바꾼다. i1 == i2면 삽입
    i2: int
    rows: Rows
    side: str        # "a" | "b"

    @property
    def insert(self) -> bool:
        return self.i1 == self.i2


def hunks(base: Rows, new: Rows, side: str) -> list[Hunk]:
    sm = SequenceMatcher(None, [i for i, _ in base], [i for i, _ in new], autojunk=False)
    return [Hunk(i1, i2, new[j1:j2], side) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def conflict(x: Hunk, y: Hunk) -> bool:
    """두 수정 덩어리가 공통 조상의 같은 줄(구간)을 건드리는가. 서로 다른 위치의 삽입, 맞닿기만 한 구간은 충돌이 아니다."""
    if x.insert and y.insert:
        return False
    if x.insert:
        return y.i1 < x.i1 < y.i2
    if y.insert:
        return x.i1 < y.i1 < x.i2
    return x.i1 < y.i2 and y.i1 < x.i2


def three_way(base: Rows, a: Rows, b: Rows, prefer: str) -> Rows:
    """공통 조상 base에서 a·b가 각자 고친 줄을 합친다. 겹치는 수정은 prefer("a"|"b") 쪽 것만 쓴다.
    조상에 없던(또는 비어 있던) 문서를 양쪽이 각자 만들었으면 서로 무관한 글이므로 이어 붙이지 않고 prefer 쪽 것을 통째로 쓴다."""
    if not base and a and b and a != b:
        return a if prefer == "a" else b
    ha, hb = hunks(base, a, "a"), hunks(base, b, "b")
    pref, other = (ha, hb) if prefer == "a" else (hb, ha)
    kept = pref + [h for h in other if not any(conflict(h, p) for p in pref)]
    kept.sort(key=lambda h: (h.i1, h.i2, h.side != "a"))
    out: Rows = []
    pos, prev = 0, None
    for h in kept:
        # 같은 자리에 두 계보가 같은 글을 삽입했으면 한 번만 넣는다(줄 ID는 달라도 내용이 같다)
        if prev and h.insert and prev.insert and h.i1 == prev.i1 and [t for _, t in h.rows] == [t for _, t in prev.rows]:
            continue
        out += base[pos:h.i1] + h.rows
        pos = max(pos, h.i2)
        prev = h
    return out + base[pos:]


def module_pick(base: Rows, a: Rows, b: Rows, prefer: str) -> Rows:
    """GEPA의 모듈 병합: 한쪽만 고친 문서는 그쪽 것을, 둘 다 고쳤으면 prefer 쪽 것을 통째로 쓴다."""
    ca, cb = a != base, b != base
    if ca and not cb:
        return a
    if cb and not ca:
        return b
    if ca and cb:
        return a if prefer == "a" else b
    return base


def merge(cands: dict[str, Candidate], a: str, b: str, prefer: str, mode: str, cid: str, iteration: int,
          system_max: int, wiki_max: int) -> Candidate | None:
    """a와 b의 공통 조상을 찾아 합친 새 후보. 합칠 수 없거나 결과가 a·b 중 하나와 같거나 크기 상한을 넘으면 None.

    mode="line"은 문서 안에서 줄 단위로, mode="module"은 문서 통째로 합친다. prefer는 "a"|"b"(겹칠 때 따를 쪽)."""
    g = common_ancestor(cands, a, b)
    if g is None:
        return None
    pick = three_way if mode == "line" else module_pick
    docs: dict[str, Rows] = {}
    for key in sorted(set(cands[g].docs) | set(cands[a].docs) | set(cands[b].docs)):
        base, ra, rb = (cands[x].docs.get(key, []) for x in (g, a, b))
        docs[key] = pick(base, ra, rb, prefer)
    child = Candidate(cid, docs, [a, b], "merge", f"병합({mode}) {a}+{b} (공통 조상 {g})", None, iteration)
    same = lambda x: all([t for _, t in child.docs.get(k, [])] == [t for _, t in x.docs.get(k, [])]
                         for k in set(child.docs) | set(x.docs))
    if same(cands[a]) or same(cands[b]) or size_error(child, system_max, wiki_max):
        return None
    return child
