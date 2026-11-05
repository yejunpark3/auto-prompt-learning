"""리뷰어를 시나리오가 다른 정답 사례(scripts/review_bench_cases.json)로 채점한다.
리뷰어(`reviewer.py`)나 리뷰 프롬프트(`prompts.py`의 REVIEW_*·SUBMIT_REVIEW_*)를 고칠 때마다 돌려, 점수가 같거나 올라야 채택한다.
한 시나리오에 맞춰 고치지 않도록 사례는 여러 시나리오(판매·설문·예약 확인·일정 협의)와 유형(수락·유보·관심만·거절·조기 종료)을 섞었다.
사용: uv run scripts/review_bench.py [--out 결과.json] [--workers 2] [--scenario 이름] [--kind 유형]"""
import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reviewer  # noqa: E402
from agent import LLMConfig  # noqa: E402

CASES = Path(__file__).with_name("review_bench_cases.json")


def run_case(case: dict, llm: LLMConfig) -> dict:
    record = {"purpose": case["purpose"], "end_reason": case["end"],
              "turns": [{"speaker": s, "text": t} for s, t in case["turns"]]}
    t = time.time()
    try:
        got = reviewer.review_call(record, llm)
    except Exception as e:  # 서버 오류는 채점에서 따로 센다
        return {**case, "got": None, "error": f"{type(e).__name__}: {e}", "sec": time.time() - t}
    return {**case, "got": None if got is None else {k: got[k] for k in ("achieved", "stage")}, "sec": time.time() - t}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--scenario")
    p.add_argument("--kind")
    p.add_argument("--plan-max-tokens", type=int, default=3000)
    args = p.parse_args()
    cases = [c for c in json.loads(CASES.read_text(encoding="utf-8"))
             if (not args.scenario or c["scenario"] == args.scenario) and (not args.kind or c["kind"] == args.kind)]
    llm = LLMConfig(plan_max_tokens=args.plan_max_tokens)
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda c: run_case(c, llm), cases))

    ok = [r for r in results if r["got"]]
    fail = len(results) - len(ok)
    ach_ok = sum(r["got"]["achieved"] == r["expect"]["achieved"] for r in ok)
    st_exact = sum(r["got"]["stage"] == r["expect"]["stage"] for r in ok)
    st_near = sum(abs(r["got"]["stage"] - r["expect"]["stage"]) <= 1 for r in ok)
    accepts = [r for r in ok if r["expect"]["achieved"]]
    false_neg = sum(not r["got"]["achieved"] for r in accepts)
    false_pos = sum(r["got"]["achieved"] and not r["expect"]["achieved"] for r in ok)
    for r in results:
        g, e = r["got"], r["expect"]
        mark = "  " if g and g["achieved"] == e["achieved"] and g["stage"] == e["stage"] else "✗ "
        got = "리뷰 실패" if not g else f"단계 {g['stage']}/달성 {g['achieved']}"
        print(f"{mark}{r['scenario']} · {r['kind']}: 정답 단계 {e['stage']}/달성 {e['achieved']} → {got} ({r['sec']:.0f}s)")
    n = len(ok) or 1
    print(f"\n사례 {len(results)}건, 리뷰 실패 {fail}건")
    print(f"달성 판정 정확도 {ach_ok}/{len(ok)} ({ach_ok / n:.0%}), 수락 놓침 {false_neg}/{len(accepts)}, 거짓 달성 {false_pos}")
    print(f"단계 정확 {st_exact}/{len(ok)} ({st_exact / n:.0%}), ±1 이내 {st_near}/{len(ok)} ({st_near / n:.0%})")
    by = defaultdict(list)
    for r in ok:
        by[r["scenario"]].append(r["got"]["achieved"] == r["expect"]["achieved"] and r["got"]["stage"] == r["expect"]["stage"])
    print("시나리오별 완전 일치:", {k: f"{sum(v)}/{len(v)}" for k, v in by.items()})
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
