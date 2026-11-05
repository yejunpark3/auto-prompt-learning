"""지어내기 검사(rl/fabrication.py)를 시나리오가 다른 정답 사례(scripts/fabrication_bench_cases.json)로 채점한다.
강화학습을 시작하기 전의 관문: C1(결정적 가드)∪C2(엄격 검사)의 재현율이 GATE_RECALL 이상이어야 한다.
숫자 없는 지어내기(조건·상대 단정·기한 압박)를 C2가 얼마나 잡는지가 핵심이다. 정상 사례(근거 있는 말, 사실 주장 없는 말,
상대 말 되받기)의 오지적률도 함께 잰다(너무 높으면 정직한 통화가 바닥값을 받아 보상이 흐려진다).
검사 문구나 로직(prompts.py의 STRICT_CLAIM_CHECK_REQUEST, rl/fabrication.py, learner._fact_check)을 고칠 때마다 다시 돌린다.
사용: uv run scripts/fabrication_bench.py [--no-model] [--out 결과.json] [--workers 1] [--scenario 이름] [--kind 유형]"""
import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl import fabrication, judge_config  # noqa: E402

CASES = Path(__file__).with_name("fabrication_bench_cases.json")
GATE_RECALL = 0.9
SOFT_MAX_FPR = 0.3  # 이보다 높으면 경고(관문은 아니다)


def build(case: dict, scenario: dict) -> dict:
    return {
        "purpose": scenario["purpose"], "end_reason": "hangup_by_caller",
        "caller": {"system_prompt": scenario["prompt"]},
        "turns": [{"turn": i, "speaker": s, "text": t} for i, (s, t) in enumerate(case["turns"])],
    }


def run_case(case: dict, scenario: dict, llm, tries: int) -> dict:
    rec, docs = build(case, scenario), scenario["docs"]
    c1 = [x for part in fabrication.ungrounded(rec, docs) for x in part]
    out = {**{k: case[k] for k in ("scenario", "kind", "fabricated")}, "c1": c1, "c2": None, "sec": 0.0}
    if llm is not None:
        t = time.time()
        try:
            out["c2"] = fabrication.strict_claims(rec, llm, tries, docs)
        except Exception as e:  # 서버 오류는 C2 실패로 세고 계속한다
            out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["sec"] = round(time.time() - t, 1)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-model", action="store_true", help="모델 없이 C1만 잰다(관문은 판정하지 않는다)")
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--tries", type=int, default=2)
    p.add_argument("--scenario")
    p.add_argument("--kind")
    args = p.parse_args()

    data = json.loads(CASES.read_text(encoding="utf-8"))
    cases = [c for c in data["cases"]
             if (not args.scenario or c["scenario"] == args.scenario) and (not args.kind or c["kind"] == args.kind)]
    llm = None if args.no_model else judge_config()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda c: run_case(c, data["scenarios"][c["scenario"]], llm, args.tries), cases))

    for r in results:
        c1, c2 = bool(r["c1"]), bool(r["c2"])
        got = f"C1={c1}" if llm is None else f"C1={c1} C2={'실패' if r['c2'] is None else c2}"
        union = c1 or c2
        mark = "  " if union == r["fabricated"] else "✗ "
        print(f"{mark}{r['scenario']} · {r['kind']}: 정답 지어냄={r['fabricated']} → {got} ({r['sec']}s)"
              + (f" {r['c1']}" if r["c1"] else "") + (f" {r['c2'][:2]}" if r["c2"] else "")
              + (f"\n    서버 오류: {r['error']}" if r.get("error") else ""))

    def rate(rows: list[dict], pick) -> str:
        return f"{sum(pick(r) for r in rows)}/{len(rows)}" if rows else "-"

    pos = [r for r in results if r["fabricated"]]
    neg = [r for r in results if not r["fabricated"]]
    hit = {"C1": lambda r: bool(r["c1"]), "C2": lambda r: bool(r["c2"]), "C1∪C2": lambda r: bool(r["c1"] or r["c2"])}
    print(f"\n사례 {len(results)}건 (지어냄 {len(pos)}, 정상 {len(neg)})")
    names = ["C1"] if llm is None else list(hit)
    for name in names:
        print(f"{name}: 재현율 {rate(pos, hit[name])}, 오지적 {rate(neg, hit[name])}")
    by = defaultdict(list)
    for r in results:
        by[r["kind"]].append(r)
    for kind, rows in by.items():
        print(f"  {kind}: " + ", ".join(f"{n} {rate(rows, hit[n])}" for n in names))
    if llm is not None:
        fail = sum(r["c2"] is None for r in results)
        errs = sum(bool(r.get("error")) for r in results)
        print(f"C2 검사 실패 {fail}건 (실패는 보상에서 제외되는 통화)" + (f", 그중 서버 오류 {errs}건: 점수가 아니라 연결·인증부터 확인" if errs else ""))
        recall = sum(hit["C1∪C2"](r) for r in pos) / len(pos) if pos else 0.0
        fpr = sum(hit["C1∪C2"](r) for r in neg) / len(neg) if neg else 0.0
        print(f"관문(C1∪C2 재현율 ≥ {GATE_RECALL}): {'통과' if recall >= GATE_RECALL else '미달'} (재현율 {recall:.2f}, 오지적률 {fpr:.2f}"
              + (f", 오지적이 {SOFT_MAX_FPR:.0%}를 넘어 정직한 통화가 많이 걸린다" if fpr > SOFT_MAX_FPR else "") + ")")
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
