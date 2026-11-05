"""반성(변이) 연산자: 부모 후보와 그 통화 결과를 보고 학습자 LLM이 문서를 고쳐 자식 후보를 만든다.

learner.py의 방식을 따른다: 줄 ID로 고치는 prompt_edit(PromptLines), 회차당 편집 상한(MAX_EDITS), 도구 호출 기록(_logged),
통화 기록 발췌(_excerpts). 다른 점:
  - 문서가 여러 개다(system.md + 편집 가능한 위키). 줄 ID는 실행 전체에서 유일하므로 line_id만으로 문서를 안다.
  - 이번에 고칠 문서는 정해져 있다(GEPA의 순환 선택). 그 밖의 문서를 고치려 하면 거절한다.
  - 사실 가드(learner._make_guard)는 system.md 편집에만 건다. 위키 편집은 크기 상한만 본다.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import learner
import wiki_store
from agent import LEARN_TOOL_ROUNDS, LLMConfig, PhoneAgent, _strip_think
from prompt_opt.candidates import SYSTEM, WIKI, Candidate, size_error
from prompt_opt.evaluate import SlotPool
from prompts import GEPA_HOW_EDIT, GEPA_HOW_NEW, GEPA_REQUEST

NEW = "(새 문서)"
START = learner.START

_STR = {"type": "string"}
EDIT_TOOL = wiki_store._tool(
    "prompt_edit",
    "문서를 줄 ID([k7x2] 같은 영문 소문자+숫자 4자리 코드)로 고친다. replace는 그 줄을 new_text로 바꾸고, insert_after는 그 줄 뒤에 "
    "new_text를 넣고(line_id가 START면 문서 맨 앞), delete는 그 줄을 지운다. new_text에는 [ID] 표시 없이 내용만 쓴다. "
    "고치지 않은 줄의 ID는 그대로이고 새로 생긴 줄만 새 ID를 받는다",
    {"action": {"type": "string", "enum": ["replace", "insert_after", "delete"]}, "line_id": _STR, "new_text": _STR},
    ["action", "line_id"],
)
CREATE_TOOL = wiki_store._tool(
    "doc_create", "새 위키 문서를 만든다(경로는 위키 안의 상대경로)", {"path": _STR, "content": _STR}, ["path", "content"])

_QUOTE_RE = re.compile(r"['\"‘“][^'\"’”]*['\"’”]")


def targets(c: Candidate, allow_new: bool = True) -> list[str]:
    return [SYSTEM, *c.wiki_keys(), *([NEW] if allow_new else [])]


def pick_target(c: Candidate, iteration: int, allow_new: bool = True) -> str:
    """GEPA의 순환 선택: 반복마다 고칠 문서를 차례로 돌린다."""
    ts = targets(c, allow_new)
    return ts[iteration % len(ts)]


def strip_quotes(text: str) -> str:
    """요약 속 따옴표 문구는 학습자가 그대로 다시 쓰므로 지운다(learner._history와 같다)."""
    return _QUOTE_RE.sub("…", " ".join(text.split()))


def call_lines(calls: list[dict]) -> str:
    out = []
    for i, c in enumerate(calls, 1):
        cell = c["cell"]
        nat = "?" if cell.get("natural") is None else f"{cell['natural']:.2f}"
        stage = "?" if cell.get("stage") is None else cell["stage"]
        line = f"- 통화 {i}: {cell['end_reason']}, 자연스러움 {nat}/1, 단계 {stage}/4" + (", 달성" if cell.get("achieved") else "")
        if cell.get("defects"):
            line += f"\n  결함: {', '.join(cell['defects'])}"
        if cell.get("zero_utts"):  # 따옴표로 감싸지 않는다(대사를 그대로 옮기게 하는 길을 줄인다)
            line += "\n  부자연스럽다고 판정된 발신자 발화: " + " / ".join(u[:70] for u in cell["zero_utts"])
        out.append(line)
    return "\n".join(out) or "(없음)"


@dataclass
class Reflection:
    child: Candidate | None
    summary: str = ""
    edits: int = 0
    log: list[dict] = field(default_factory=list)
    error: str | None = None
    note: str = ""   # child가 None인 이유(변경 없음 등)


def reflect(parent: Candidate, calls: list[dict], *, cid: str, iteration: int, target: str, llm: LLMConfig,
            slots: SlotPool, wiki_dir: Path, objective: str, history: str, used_ids: set[str],
            system_max: int, wiki_max: int, readonly: str, max_edits: int = learner.MAX_EDITS) -> Reflection:
    """calls: [{"path", "end_reason", "achieved", "purpose", "cell"}] 부모가 건 통화들(미니배치).
    used_ids는 실행 전체의 줄 ID 집합(새 줄 ID가 겹치지 않게 공유한다)."""
    docs = {k: learner.PromptLines("", rows=rows, used=used_ids) for k, rows in parent.docs.items()}
    log: list[dict] = []
    state = {"edits": 0, "created": 0}

    def shown() -> str:
        return "\n\n".join(f"[{k}]\n{pl.show() or '(비어 있음)'}" for k, pl in docs.items()) or "(없음)"

    def snapshot() -> Candidate:
        return Candidate(cid, {k: list(pl.rows) for k, pl in docs.items()})

    def too_big(doc: str, before_chars: int) -> str | None:
        """편집한 문서의 크기 상한. 이미 상한을 넘은 시작점에서는 줄이는 편집만 받는다."""
        c = snapshot()
        if doc == SYSTEM:
            n = len(c.text(SYSTEM))
            if n < learner.PROMPT_MIN_CHARS:
                return f"결과가 {n}자입니다. 최소 {learner.PROMPT_MIN_CHARS}자여야 합니다"
            if n > system_max and n >= before_chars:
                return f"결과가 {n}자입니다. {system_max}자 이하로 줄이세요"
        elif (n := c.wiki_chars()) > wiki_max and n >= before_chars:
            return f"편집 가능한 위키 문서 전체가 {n}자입니다. {wiki_max}자 이하로 줄이세요"
        return None

    def chars(doc: str) -> int:
        return len(snapshot().text(SYSTEM)) if doc == SYSTEM else snapshot().wiki_chars()

    def find_doc(line_id: str) -> tuple[str | None, str]:
        """(문서 키, 정규화한 line_id). START[:경로]는 경로가 없으면 이번 대상 문서."""
        head, _, tail = line_id.partition(":")
        if head.strip("[] ").upper() == START:
            tail = tail.strip()
            key = tail if tail in docs else WIKI + tail if (WIKI + tail) in docs else (target if not tail else None)
            return key, START
        lid = line_id.strip("[] ").lower()
        return next((k for k, pl in docs.items() if lid in {i for i, _ in pl.rows}), None), lid

    def run(name: str, arguments: str) -> str:
        if name not in ("prompt_edit", "doc_create"):
            return wiki_store.run_read_tool(name, arguments)
        if state["edits"] >= max_edits:
            return f"오류: 이번 회차에 이미 {max_edits}번 고쳤습니다. 더 고치지 말고 바꾼 내용을 한두 줄로 요약하며 끝내세요"
        try:
            a = json.loads(arguments)
            if not isinstance(a, dict):
                raise TypeError
        except (ValueError, TypeError):
            return "오류: 인자는 JSON 객체여야 합니다"
        if name == "doc_create":
            if target != NEW:
                return f"오류: 이번에는 {target}만 고칠 수 있습니다(doc_create는 쓸 수 없음)"
            if state["created"]:
                return "오류: 새 문서는 하나만 만듭니다. 바꾼 내용을 한두 줄로 요약하며 끝내세요"
            try:
                rel, full = wiki_store._resolve(str(a.get("path", "")))
            except wiki_store.WikiError as e:
                return f"오류: {e}"
            content = str(a.get("content") or "").strip()
            if wiki_store.is_readonly(rel):
                return f"오류: {rel}은 고칠 수 없는 문서입니다"
            if WIKI + rel in docs or full.exists():
                return f"오류: 이미 있는 문서입니다: {rel}"
            if not content:
                return "오류: content가 비었습니다"
            before = chars(WIKI)
            pl = learner.PromptLines(content, used=used_ids)
            docs[WIKI + rel] = pl
            if err := too_big(WIKI + rel, before):
                del docs[WIKI + rel]
                return f"오류: {err}"
            state["edits"] += 1
            state["created"] += 1
            return f"위키 문서 {rel}을 만들었습니다:\n{pl.show()}"

        # prompt_edit
        if target == NEW:
            return "오류: 이번에는 새 문서를 만드는 차례입니다. doc_create를 쓰세요"
        if "line_id" not in a:  # 줄 ID를 키로 보내는 실수를 받아 준다(learner와 같다)
            id_keys = [k for k in a if re.fullmatch(r"\[?[a-z0-9]{4}\]?", str(k))]
            if len(id_keys) == 1:
                a["line_id"] = id_keys[0]
        try:
            action, line_id, new = a["action"], str(a["line_id"]), str(a.get("new_text") or "")
        except KeyError:
            return ('오류: 잘못된 인자입니다. 키는 action, line_id, new_text여야 합니다. '
                    '예: {"action": "replace", "line_id": "k7x2", "new_text": "새 내용"}')
        doc, lid = find_doc(line_id)
        if doc is None:
            return f"오류: 없는 line_id입니다: {line_id}\n현재 문서:\n{shown()}"
        if doc != target:
            return f"오류: 이번에는 {target}만 고칠 수 있습니다({doc}의 줄입니다)"
        pl = docs[doc]
        rows_before, size_before = list(pl.rows), chars(doc)
        err = pl.edit(action, lid, new)
        if err is None and doc == SYSTEM and action != "delete" and guard and (g := guard(new)):
            err = f"오류: {g}"
        if err is None and (e := too_big(doc, size_before)):
            err = f"오류: {e}"
        if err:
            pl.rows = rows_before
            return f"{err}\n현재 {doc}:\n{pl.show()}"
        state["edits"] += 1
        return f"{doc}를 고쳤습니다. 최신 줄 ID:\n{pl.show()}"

    review_rows = [{"purpose": c["purpose"], "achieved": bool(c["achieved"]), "path": c["path"], "end_reason": c["end_reason"]}
                   for c in calls]
    learn_rows = [{"path": c["path"], "end_reason": c["end_reason"], "achieved": bool(c["achieved"])} for c in calls]
    limits = (f"system.md는 전체 {system_max}자 이내여야 한다." if target == SYSTEM
              else f"편집 가능한 위키 문서 전체는 {wiki_max}자 이내여야 한다.")
    target_desc = ("새 위키 문서 하나 만들기" if target == NEW else target)
    request = GEPA_REQUEST.format(
        n=len(calls), objective=objective, reviews=call_lines(calls), transcripts=learner._excerpts(learn_rows),
        history=history or "(없음)", docs=shown(), readonly=readonly or "(없음)", target_desc=target_desc,
        how=(GEPA_HOW_NEW if target == NEW else GEPA_HOW_EDIT).format(limits=limits), max_edits=max_edits)

    refl = Reflection(None)
    guard = None
    try:
        # 반성은 학습자 요청과 사실 검사(2개 병렬)를 보내므로 서버 슬롯을 2개 쓴다
        with slots.hold(llm.api_base, 2), wiki_store.using_root(wiki_dir):
            guard = learner._make_guard(review_rows, docs[SYSTEM].text(), llm)
            agent = PhoneAgent("learner", None, "학습자", "전화 발신자의 프롬프트와 위키를 개선한다.", llm,
                               tools=[*wiki_store.WIKI_READ_TOOLS, CREATE_TOOL if target == NEW else EDIT_TOOL],
                               tool_runner=learner._logged(run, log))
            raw, _, meta, _ = agent._generate(
                messages=[{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": request}],
                max_tokens=llm.plan_max_tokens, tools=agent.tools, final_tools=[], max_rounds=LEARN_TOOL_ROUNDS)
        refl.summary = _strip_think(raw).strip()
    except Exception as e:  # 끝까지 못 간 반성의 반쯤 적용된 편집은 남기지 않는다(learner.learn과 같다)
        refl.error = f"{type(e).__name__}: {e}"
        refl.log, refl.edits = log, state["edits"]
        refl.note = "error"
        return refl
    refl.log, refl.edits = log, state["edits"]
    child = Candidate(cid, {k: list(pl.rows) for k, pl in docs.items()}, [parent.cid], "reflect", refl.summary, target, iteration)
    if not state["edits"]:
        refl.note = "no_edit"
    elif all(child.text(k) == parent.text(k) for k in set(child.docs) | set(parent.docs)):
        refl.note = "no_change"
    elif err := size_error(child, system_max, wiki_max):
        refl.note = f"size: {err}"
    else:
        refl.child = child
    return refl
