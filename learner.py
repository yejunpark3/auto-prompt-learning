import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import wiki_store
from agent import LEARN_TOOL_ROUNDS, LLMConfig, PhoneAgent, _strip_think
from prompts import (
    LEARN_REQUEST,
    SYSTEM_PROMPT_PATH,
    load_caller_profile,
)

LEARN_DIR = Path("agent_workspace/learning")
PENDING = LEARN_DIR / "pending.jsonl"
PROMPT_MAX_CHARS = 300
PROMPT_MIN_CHARS = 20
WRITE_TOOLS = {"wiki_create", "wiki_update", "wiki_delete"}

PROMPT_WRITE_TOOL = wiki_store._tool(
    "prompt_write",
    f"발신자 system.md 전체를 새로 쓴다 (최대 {PROMPT_MAX_CHARS}자)",
    {"content": {"type": "string"}},
    ["content"],
)


def enqueue(record: dict, path: Path) -> int:
    LEARN_DIR.mkdir(parents=True, exist_ok=True)
    review = record.get("review") or {}
    row = {"call_id": record["call_id"], "path": str(path), "end_reason": record["end_reason"], "purpose": record.get("purpose", ""),
           "spam": review.get("spam"), "achieved": review.get("achieved"), "notes": review.get("notes", "")}
    with PENDING.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(_pending())


def _pending() -> list[dict]:
    if not PENDING.exists():
        return []
    rows = [json.loads(l) for l in PENDING.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [r for r in rows if "achieved" in r and "spam" in r]
    if len(kept) < len(rows):  # 점수(score) 형식의 옛 행은 신고·달성 여부를 알 수 없다
        print(f"[경고] 옛 형식 대기열 {len(rows) - len(kept)}건을 버립니다", flush=True)
        PENDING.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    return kept


def _rate(rows: list[dict], key: str) -> float | None:
    return round(sum(bool(r.get(key)) for r in rows) / len(rows), 2) if rows else None


def _copy_wiki(dst: Path) -> None:
    shutil.copytree(wiki_store.WIKI_ROOT, dst, ignore=shutil.ignore_patterns(".*"))


def _tool_runner(prompt_path: str):
    def run(name: str, arguments: str) -> str:
        if name == "prompt_write":
            try:
                content = json.loads(arguments)["content"].strip()
            except (ValueError, KeyError, TypeError, AttributeError):
                return "오류: 잘못된 인자"
            if len(content) < PROMPT_MIN_CHARS:
                return f"오류: {len(content)}자입니다. 최소 {PROMPT_MIN_CHARS}자의 프롬프트를 쓰세요"
            if len(content) > PROMPT_MAX_CHARS:
                return f"오류: {len(content)}자입니다. {PROMPT_MAX_CHARS}자 이하로 줄여 다시 쓰세요"
            Path(prompt_path).write_text(content + "\n", encoding="utf-8")
            return "system.md를 고쳤습니다"
        if name in WRITE_TOOLS:
            try:
                a = json.loads(arguments)
                rel, _ = wiki_store._resolve(a.get("path", ""))
            except (ValueError, TypeError, AttributeError, wiki_store.WikiError) as e:
                return f"오류: {e}"
            if wiki_store.is_readonly(rel):
                return f"오류: {rel}은 고칠 수 없는 문서입니다"
        return wiki_store.run_tool(name, arguments)
    return run


def learn(llm: LLMConfig, prompt_path: str = SYSTEM_PROMPT_PATH) -> Path:
    rows = _pending()
    if not wiki_store.READONLY_FILE.exists():
        print(f"[경고] {wiki_store.READONLY_FILE}이 없어 모든 위키 문서를 고칠 수 있습니다", flush=True)
    n = len(list(LEARN_DIR.glob("round_*"))) + 1
    out = LEARN_DIR / f"round_{n:03d}"
    result: dict = {}
    try:
        out.mkdir(parents=True)
        shutil.copy(prompt_path, out / "system_before.md")
        _copy_wiki(out / "wiki_before")
        rates = {"achieved_rate": _rate(rows, "achieved"), "spam_rate": _rate(rows, "spam")}
        (out / "reviews.json").write_text(
            json.dumps({**rates, "reviews": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")

        reviews = "\n\n".join(f"- (목적: {r.get('purpose', '')}, {r['end_reason']}, 달성 {'예' if r['achieved'] else '아니오'}, 신고 {'예' if r['spam'] else '아니오'}) {r['notes']}" for r in rows)
        request = LEARN_REQUEST.format(
            n=len(rows), achieved_rate=rates["achieved_rate"], spam_rate=rates["spam_rate"], reviews=reviews, prompt=load_caller_profile(prompt_path),
            pages="\n".join(wiki_store.list_pages()) or "(없음)", max_chars=PROMPT_MAX_CHARS,
            readonly="\n".join(wiki_store.readonly_patterns()) or "(없음)")
        learner = PhoneAgent(
            "learner", None, "학습자", "전화 발신자의 프롬프트와 위키를 개선한다.", llm,
            tools=[*wiki_store.WIKI_TOOLS, PROMPT_WRITE_TOOL], tool_runner=_tool_runner(prompt_path))
        raw, _, meta, lookups = learner._generate(
            messages=[{"role": "system", "content": learner.system_prompt},
                      {"role": "user", "content": request}],
            max_tokens=llm.plan_max_tokens, tools=learner.tools, final_tools=[],
            max_rounds=LEARN_TOOL_ROUNDS)
        result = {"summary": _strip_think(raw).strip(), **meta, "tool_calls": lookups}
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}
        print(f"[경고] 학습 실패: {result['error']}", flush=True)
    finally:
        PENDING.unlink(missing_ok=True)  # 어떤 실패에도 비운다 (매 통화 재시도 방지)

    try:
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(prompt_path, out / "system_after.md")
        _copy_wiki(out / "wiki_after")
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        (out / "learner.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[경고] 학습 결과 저장 실패: {e}", flush=True)
    return out


def maybe_learn(llm: LLMConfig, prompt_path: str = SYSTEM_PROMPT_PATH, every: int = 20) -> Path | None:
    if every > 0 and len(_pending()) >= every:
        return learn(llm, prompt_path)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="대기열 상태를 보거나 학습을 바로 실행")
    p.add_argument("--now", action="store_true", help="개수와 무관하게 지금 학습")
    p.add_argument("--caller-prompt", default=SYSTEM_PROMPT_PATH)
    args = p.parse_args()
    rows = _pending()
    print(f"대기열 {len(rows)}건")
    if args.now and rows:
        print(f"→ {learn(LLMConfig(), args.caller_prompt)}")


if __name__ == "__main__":
    main()
