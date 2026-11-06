"""자연스러움 심판(rl/reward.py의 judge_call)을 정답이 붙은 통화(scripts/natural_bench_cases.json)로 채점한다.
강화학습을 시작하기 전의 관문: 심판의 0점 재현율이 GATE_ZERO_RECALL 이상이어야 한다(부자연스러운 발화를 놓치면 그 말투가 보상을 받는다).
사례: 실제 통화 10건(종료 사유가 섞이게 뽑음)과 시나리오가 다른 합성 통화 6건. 정답은 사람이 아니라 Claude가 심판 출력을 보지 않고
매겼고, 사용자가 일부를 검수한다(검수 결과 어긋난 발화는 사례의 "reviewed"에 표시하고 기준 문구를 고친다).
심판 문구(prompts.py의 NATURAL_JUDGE_*, SUBMIT_NATURALNESS_*)나 모델을 바꿀 때마다 돌려, 지표가 같거나 올라야 채택한다.
모델 후보 비교: --model openai/... --api-base http://.../v1 (학습 심판은 기본 14B, 감사 심판은 다른 모델).
사용: uv run scripts/natural_bench.py [--out 결과.json] [--repeat 1] [--workers 1] [--model M --api-base URL]"""
import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl import judge_config, reward  # noqa: E402

CASES = Path(__file__).with_name("natural_bench_cases.json")
GATE_ZERO_RECALL = 0.8  # 가중치 RL 관문(발화 단위). RL은 정책이 심판의 빈틈을 파고들 수 있어 높은 재현율이 필요하다.
# GEPA 관문(통화 단위): GEPA는 후보 몇 개를 통화 단위로 비교하고 최종 후보는 사람이 읽는다. 필요한 것은 "통화의 자연스러움 순위가 맞는가"다.
GATE_PAIRWISE = 0.75    # 정답 순위가 다른 통화 쌍 중 심판 순서가 맞는 비율
GATE_SPEARMAN = 0.60
LOOPY = reward.LOOPY_ENDS  # 이 종료 사유는 코드가 끝맺음 부자연으로 정하므로 심판의 ending_ok는 채점하지 않는다


def _ranks(xs: list[float]) -> list[float]:
    """평균 순위(동점은 평균)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out, i = [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def _pearson(a: list[float], b: list[float]) -> float | None:
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    da, db = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return None if da == 0 or db == 0 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (da * db) ** 0.5


def call_level_agreement(results: list[dict], cases: dict[str, dict], speaker: str = "caller") -> dict:
    """통화마다 그 화자의 발화 점수 평균(정답 대 심판)을 구해, 통화 사이 순위가 맞는지 잰다.
    pairwise: 정답 순위가 다른 통화 쌍 중 심판 순서가 맞는 비율(심판이 동점인 쌍은 틀린 것으로 센다. 심판이 순위를 못 매긴 것이므로
    분모에서 빼거나 반만 맞았다고 하면 점수가 부풀어 오른다), pairwise_excl_ties: 동점 쌍을 뺀 값(참고용), spearman: 평균 순위 상관.
    results는 run_case의 결과. 심판 평균은 같은 통화의 여러 실행(runs)의 평균이다."""
    gold, judge, ids = [], [], []
    for r in results:
        turns = cases[r["id"]]["turns"]
        idx = [k for k, (i, (s, t)) in enumerate([(i, st) for i, st in enumerate(turns) if i > 0 and st[1]]) if s == speaker]
        runs = [run for run in r["runs"] if run]
        if not idx or not runs:
            continue
        gold.append(statistics.fmean(r["gold"][k] for k in idx) / 2)
        judge.append(statistics.fmean(statistics.fmean(run["scores"][k] for k in idx) / 2 for run in runs))
        ids.append(r["id"])
    pairs = [(i, j) for i in range(len(gold)) for j in range(i) if gold[i] != gold[j]]
    right = sum((judge[i] > judge[j]) == (gold[i] > gold[j]) and judge[i] != judge[j] for i, j in pairs)
    ties = sum(judge[i] == judge[j] for i, j in pairs)
    sp = _pearson(_ranks(gold), _ranks(judge))   # 동점은 평균 순위(표준 정의). 동점을 순서대로 풀면 점수가 부풀어 오른다
    return {"calls": len(gold), "pairs": len(pairs), "pairwise": right / len(pairs) if pairs else None, "ties": ties,
            "pairwise_excl_ties": right / (len(pairs) - ties) if len(pairs) > ties else None, "spearman": sp,
            "pearson": _pearson(gold, judge), "ids": ids, "gold": gold, "judge": judge}


def build(case: dict) -> dict:
    turns = []
    for i, (speaker, text) in enumerate(case["turns"]):
        t = {"turn": i, "speaker": speaker, "text": text}
        if i > 0 and not text:
            t["hangup"] = True
        turns.append(t)
    return {"end_reason": case["end_reason"], "turns": turns}


def run_case(case: dict, llm, repeat: int) -> dict:
    rec = build(case)
    runs, t0, err = [], time.time(), None
    for _ in range(repeat):
        try:
            jr = reward.judge_call(rec, llm)
        except Exception as e:  # 서버 오류는 실패로 세되 원인을 남긴다(심판이 형식을 못 맞춘 실패와 구분한다)
            jr, err = None, f"{type(e).__name__}: {str(e)[:200]}"
        runs.append(None if jr is None else {"scores": [jr["scores"][n] for n in sorted(jr["scores"])],
                                             "ending_ok": jr["ending_ok"], "notes": jr["notes"]})
    return {"id": case["id"], "source": case["source"], "end_reason": case["end_reason"],
            "gold": case["scores"], "gold_ending": case["ending_ok"], "runs": runs, "error": err, "sec": round(time.time() - t0, 1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1, help="같은 통화를 심판에게 여러 번 물어 변동을 본다")
    p.add_argument("--model")
    p.add_argument("--api-base")
    p.add_argument("--source", choices=["real", "synthetic"])
    args = p.parse_args()

    cases = [c for c in json.loads(CASES.read_text(encoding="utf-8")) if not args.source or c["source"] == args.source]
    over = {}
    if args.model:
        over["model"] = args.model
    if args.api_base is not None:
        over["api_base"] = args.api_base or None
    llm = judge_config(**over)
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda c: run_case(c, llm, args.repeat), cases))

    pairs, fails, end_pairs = [], 0, []  # (정답, 심판)
    for r in results:
        for run in r["runs"]:
            if run is None:
                fails += 1
                continue
            pairs += list(zip(r["gold"], run["scores"]))
            if r["end_reason"] not in LOOPY:
                end_pairs.append((r["gold_ending"], run["ending_ok"]))
        shown = " ".join("".join(str(x) for x in run["scores"]) if run else "실패" for run in r["runs"])
        gold = "".join(str(x) for x in r["gold"])
        print(f"{r['id']:12} {r['end_reason']:17} 정답 {gold:14} 심판 {shown}  ({r['sec']}s)" + (f"\n    서버 오류: {r['error']}" if r["error"] else ""))

    errs = sum(bool(r["error"]) for r in results)
    print(f"\n통화 {len(results)}건 × {args.repeat}회, 심판 실패 {fails}회" + (f"(서버 오류가 난 통화 {errs}건: 점수가 아니라 연결·인증부터 확인)" if errs else "")
          + f", 발화 판정 {len(pairs)}개")
    if not pairs:
        return
    exact = sum(g == j for g, j in pairs)
    near = sum(abs(g - j) <= 1 for g, j in pairs)
    zeros = [(g, j) for g, j in pairs if g == 0]
    flagged = [(g, j) for g, j in pairs if j == 0]
    recall = sum(j == 0 for _, j in zeros) / len(zeros) if zeros else 0.0
    prec = sum(g == 0 for g, _ in flagged) / len(flagged) if flagged else 0.0
    twos = [(g, j) for g, j in pairs if g == 2]
    print(f"정확 일치 {exact}/{len(pairs)} ({exact / len(pairs):.0%}), ±1 이내 {near}/{len(pairs)} ({near / len(pairs):.0%}), "
          f"평균 절대오차 {statistics.fmean(abs(g - j) for g, j in pairs):.2f}")
    print(f"0점 재현율 {sum(j == 0 for _, j in zeros)}/{len(zeros)} ({recall:.0%}), 0점 정밀도 {sum(g == 0 for g, _ in flagged)}/{len(flagged)} ({prec:.0%}), "
          f"자연스러운 발화(정답 2점)를 0점으로 오판 {sum(j == 0 for _, j in twos)}/{len(twos)}")
    print("혼동행렬(행=정답, 열=심판 0/1/2): " + " | ".join(
        f"{g}: " + "/".join(str(sum(1 for gg, jj in pairs if gg == g and jj == j)) for j in (0, 1, 2)) for g in (0, 1, 2)))
    if end_pairs:
        ok = sum(g == j for g, j in end_pairs)
        bad = [(g, j) for g, j in end_pairs if not g]
        print(f"끝맺음(종료 사유가 루프·max_turns가 아닌 통화) 일치 {ok}/{len(end_pairs)}, 부자연스러운 끝맺음 재현율 "
              f"{sum(not j for _, j in bad)}/{len(bad)}")
    print(f"[가중치 RL 관문] 발화 단위 0점 재현율 ≥ {GATE_ZERO_RECALL}: {'통과' if recall >= GATE_ZERO_RECALL else '미달'} (재현율 {recall:.2f}, 정밀도 {prec:.2f})")
    cl = call_level_agreement(results, {c["id"]: c for c in cases})
    if cl["pairs"]:
        sp = cl["spearman"]
        ok = cl["pairwise"] >= GATE_PAIRWISE and sp is not None and sp >= GATE_SPEARMAN
        print(f"통화 단위 순위 일치(발신자 기준, {cl['calls']}통화): 쌍 순서 일치 {cl['pairwise']:.0%}({cl['pairs']}쌍 중 심판 동점 {cl['ties']}쌍"
              + (f", 동점을 뺀 값 {cl['pairwise_excl_ties']:.0%}" if cl["pairwise_excl_ties"] is not None else "") + "), 스피어먼 "
              + (f"{sp:+.2f}" if sp is not None else "계산 불가") + (f", 피어슨 {cl['pearson']:+.2f}" if cl["pearson"] is not None else ""))
        print(f"[GEPA 관문] 쌍 순서 일치 ≥ {GATE_PAIRWISE:.0%} 그리고 스피어먼 ≥ {GATE_SPEARMAN}: {'통과' if ok else '미달'} "
              f"(통화 {cl['calls']}개라 쌍 순서 일치의 오차 범위는 ±10%p 안팎이다. 정답이 사람 라벨이 아니면 잠정)")
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
