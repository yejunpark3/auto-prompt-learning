"""배치 → 오류 재실행 → 지표 → 자동 게이트(롤백) → 종료 판정을 자동 반복한다.
사용: uv run scripts/auto_loop.py --rounds 5 [--baseline] [--seed 1] [--learn-every 20]
코드·프롬프트 직접 변경과 정독(자연스러움 판단)은 하지 않는다. 회차마다 tmp/loop_log.md에 지표를 남긴다."""
import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PURPOSE = "월 9,900원 실손보험 가입 권유"
HISTORY = Path("agent_workspace/conversation_history")
ERRORS = Path("agent_workspace/errors")
LEARN_DIR = Path("agent_workspace/learning")
PENDING = LEARN_DIR / "pending.jsonl"
LOG = Path("tmp/loop_log.md")
STATE = Path("tmp/auto_state.json")
LOOPY = {"repetition_loop", "max_turns", "empty_response"}

# 게이트: 직전 배치 대비 이만큼 나빠지면 롤백한다.
GATE_ACHIEVED_DROP = 0.20   # 달성률 하락(20통화 중 4건)
GATE_LOOPY_RISE = 3         # 루프성 종료 증가 건수
GATE_LONG_RISE = 0.20       # 발신자 3문장 이상 발화 비율 증가
# 종료: 루프성 종료 0이 이만큼 연속이고 달성률이 더 오르지 않으면 멈춘다.
STOP_STREAK = 3


def run(*cmd: str, out: Path | None = None) -> str:
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(["uv", "run", *cmd], capture_output=True, text=True)
    text = r.stdout + r.stderr
    if out:
        out.write_text(text, encoding="utf-8")
    if r.returncode:
        print(text[-2000:])
        sys.exit(f"실패: {' '.join(cmd)}")
    return r.stdout


def records(last: int) -> list[dict]:
    files = sorted(HISTORY.glob("*.json"))[-last:]
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def metrics(n: int) -> dict:
    recs = records(n)
    reviewed = [r for r in recs if r.get("review")]
    caller = [t["text"] for r in recs for t in r["turns"] if t["speaker"] == "caller" and t["text"]]
    long_ = sum(len([s for s in re.split(r"[.!?]+\s*", t) if s.strip()]) >= 3 for t in caller)
    return {
        "calls": len(recs),
        "achieved": sum(bool(r["review"]["achieved"]) for r in reviewed) / len(reviewed) if reviewed else 0.0,
        "loopy": sum(r["end_reason"] in LOOPY for r in recs),
        "long_ratio": long_ / len(caller) if caller else 0.0,
    }


def error_records(last: int) -> list[tuple[Path, str]]:
    found = []
    for f in sorted(HISTORY.glob("*.json"))[-last:]:
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["end_reason"] == "error":
            found.append((f, r["callee"]["uuid"]))
    return found


def batch(label: str, n: int, learn_every: int, seed: int) -> None:
    Path("tmp").mkdir(exist_ok=True)
    run("main.py", "--purpose", PURPOSE, "--n", str(n), "--learn-every", str(learn_every),
        "--seed", str(seed), out=Path(f"tmp/batch_{label}.log"))
    # 서버 오류 통화는 옮기고 같은 수신자로 다시 돌린다(최대 2번).
    for _ in range(2):
        errs = error_records(n)
        if not errs:
            break
        if len(errs) >= n / 2:
            sys.exit(f"오류 {len(errs)}/{n}건: 서버를 점검하세요. 기록은 그대로 둡니다.")
        ERRORS.mkdir(parents=True, exist_ok=True)
        for f, _ in errs:
            f.rename(ERRORS / f.name)
        for _, uuid in errs:
            run("main.py", "--purpose", PURPOSE, "--n", "1", "--learn-every", str(learn_every),
                "--callee-uuid", uuid, out=Path(f"tmp/batch_{label}_retry.log"))


def gate(prev: dict, cur: dict) -> list[str]:
    """직전 배치보다 나빠진 이유 목록(비어 있으면 통과)."""
    why = []
    if prev["achieved"] - cur["achieved"] >= GATE_ACHIEVED_DROP:
        why.append(f"달성률 {prev['achieved']:.0%}→{cur['achieved']:.0%}")
    if cur["loopy"] - prev["loopy"] >= GATE_LOOPY_RISE:
        why.append(f"루프성 종료 {prev['loopy']}→{cur['loopy']}건")
    if cur["long_ratio"] - prev["long_ratio"] >= GATE_LONG_RISE:
        why.append(f"3문장↑ 발화 {prev['long_ratio']:.0%}→{cur['long_ratio']:.0%}")
    return why


def rollback() -> str:
    """최신 round N이 아니라 N-1의 *_before로 프롬프트·위키를 되돌린다(계획서 3절 5번)."""
    rounds = sorted(LEARN_DIR.glob("round_*"))
    if len(rounds) < 2:
        return "되돌릴 round 없음"
    target = rounds[-2]
    shutil.copy(target / "system_before.md", "prompt/system.md")
    run_rsync = subprocess.run(
        ["rsync", "-a", "--delete", "--exclude=.*", f"{target / 'wiki_before'}/", "wiki/"])
    if run_rsync.returncode:
        sys.exit("rsync 실패")
    return f"{target.name}의 *_before로 복원(코드 변경은 그대로)"


def fmt(m: dict) -> str:
    return (f"통화 {m['calls']}건, 달성률 {m['achieved']:.0%}, 루프성 종료 {m['loopy']}건, "
            f"발신자 3문장↑ 발화 {m['long_ratio']:.0%}")


def log(title: str, body: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n## {title} ({datetime.now():%Y-%m-%d %H:%M})\n{body}\n")


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"prev": None, "streak": 0}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(s), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=1, help="일반 회차 수")
    p.add_argument("--baseline", action="store_true", help="학습 없는 기준선 배치만 돈다")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--learn-every", type=int, help="기본값은 n. n보다 크면 학습하지 않는다")
    p.add_argument("--no-gate", action="store_true", help="자동 롤백 끄기")
    args = p.parse_args()
    learn_every = args.learn_every or args.n
    learns = learn_every <= args.n

    if args.baseline:
        batch("0", args.n, args.n + 1, args.seed)
        m = metrics(args.n)
        save_state({"prev": m, "streak": 0})
        log("배치 0 (기준선, 학습 없음)", fmt(m))
        print(fmt(m))
        print("기준선 끝. review_check.py로 리뷰어를 점검하고 `uv run learner.py --now`로 첫 학습을 하세요.")
        return

    state = load_state()
    for k in range(1, args.rounds + 1):
        batch(str(k), args.n, learn_every, args.seed)
        cur = metrics(args.n)
        note = [fmt(cur)]
        prev = state["prev"]

        if learns and PENDING.exists() and PENDING.read_text(encoding="utf-8").strip():
            run("learner.py", "--now", out=Path(f"tmp/learn_after_{k}.log"))  # 리뷰 실패 등으로 남은 대기열

        bad = gate(prev, cur) if prev and learns and not args.no_gate else []
        if bad:
            note.append(f"**롤백**: {', '.join(bad)} → {rollback()}")
            state["streak"] = 0
        else:
            # 종료 판정: 루프성 종료 0 연속 + 달성률 정체
            clean = cur["loopy"] == 0
            plateau = prev is not None and cur["achieved"] <= prev["achieved"]
            state["streak"] = state["streak"] + 1 if clean and plateau else (1 if clean and prev is None else 0)
        if not bad:  # 롤백했으면 나빠진 배치가 아니라 직전 정상 배치를 기준으로 계속 비교한다
            state["prev"] = cur
        save_state(state)
        log(f"배치 {k}", "\n".join(note))
        print("\n".join(note), flush=True)

        if state["streak"] >= STOP_STREAK:
            print("종료 조건 충족: 다른 시드(2)로 학습 없이 검증합니다.")
            batch("seed2", args.n, args.n + 1, 2)
            v = metrics(args.n)
            ok = v["loopy"] == 0
            log("시드 2 검증", f"{fmt(v)} → {'통과' if ok else '미통과'}")
            PENDING.unlink(missing_ok=True)
            print(f"{fmt(v)} → {'통과: 멈추고 보고합니다' if ok else '미통과: 정독 후 직접 판단하세요'}")
            return


if __name__ == "__main__":
    main()
