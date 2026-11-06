"""GEPA의 선택 로직: 파레토 후보 풀, 부모 뽑기, 짝지은 순차 검정. 모델도 서버도 모른다(순수 계산)."""
import math
import statistics

EPS = 1e-9


def _num(cid: str) -> int:
    return int(cid[1:])


def pareto_front(table: dict[str, dict[str, float | None]], keys: list[str]) -> dict[str, int]:
    """table[후보][키] = 점수(없으면 None·누락). 키마다 최고 점수를 낸 후보(동점 모두)를 모으고,
    다른 살아 있는 후보가 같은 키들을 모두 이기고 있는 후보(지배당한 후보)를 뺀다.

    돌려주는 값은 {후보: 이긴 키 수}이고 부모를 뽑는 가중치로 쓴다. 지배 제거는 이긴 키가 가장 적은 후보부터,
    같으면 나중에 만든 후보부터 지운다(같은 일을 하는 후보가 둘이면 먼저 만든 쪽을 남긴다)."""
    cover: dict[str, set[str]] = {c: set() for c in table}
    for k in keys:
        vals = {c: t[k] for c, t in table.items() if t.get(k) is not None}
        if vals:
            best = max(vals.values())
            for c, v in vals.items():
                if v >= best - EPS:
                    cover[c].add(k)
    live = {c for c in table if cover[c]}
    while True:
        for c in sorted(live, key=lambda c: (len(cover[c]), -_num(c))):
            others = live - {c}
            if all(any(k in cover[o] for o in others) for k in cover[c]):
                live.remove(c)
                break
        else:
            return {c: len(cover[c]) for c in sorted(live, key=_num)}


def sample_parent(rng, weights: dict[str, int]) -> str:
    cids = sorted(weights, key=_num)
    return rng.choices(cids, weights=[weights[c] for c in cids])[0]


def paired_deltas(child: dict[str, float | None], parent: dict[str, float | None], order: list[str]) -> list[float]:
    """order(수신자 순서)에서 자식·부모 둘 다 점수가 있는 수신자의 차이(자식 − 부모)."""
    return [child[u] - parent[u] for u in order if child.get(u) is not None and parent.get(u) is not None]


def mean_se(xs: list[float]) -> tuple[float, float, int]:
    """(평균, 표준오차, n). n < 2이면 표준오차는 무한대(아직 아무것도 말할 수 없다)."""
    n = len(xs)
    if n == 0:
        return 0.0, math.inf, 0
    m = statistics.fmean(xs)
    return m, (statistics.stdev(xs) / math.sqrt(n) if n >= 2 else math.inf), n


def verdict(deltas: dict[str, list[float]]) -> str:
    """짝지은 순차 검정. deltas[목표] = 수신자별 (자식 − 부모).
    "accept": 어떤 목표에서든 평균 > 표준오차(n ≥ 2). 자식이 어느 한 목표에서라도 뚜렷이 나아졌다.
              (다른 목표가 나빠져도 받는다: 두 목표의 균형은 다목적 파레토 풀이 맡는다)
    "reject": 점수가 있는 모든 목표에서 평균 + 표준오차 ≤ 0(n ≥ 2). 더 돌려도 이길 가능성이 낮다.
    "more": 어느 쪽도 아니다(수신자를 더 돌려야 한다). 표준편차가 0이고 평균 > 0이면 accept, 평균 = 0이면 reject다."""
    stats = {o: mean_se(d) for o, d in deltas.items() if d}
    if any(n >= 2 and m > se for m, se, n in stats.values()):
        return "accept"
    if stats and all(n >= 2 and m + se <= 0 for m, se, n in stats.values()):
        return "reject"
    return "more"


def lexi(means: dict[str, float | None], objectives: list[str]) -> tuple:
    """사전식 비교 키(없는 값은 가장 낮게): 첫 목표가 높은 후보가 앞서고 같으면 다음 목표."""
    return tuple(-math.inf if means.get(o) is None else means[o] for o in objectives)


# ───────────── 확인 단계(ours) ─────────────
# 선별(verdict)은 2~6명으로 느슨하게 거른다. 그 선별에 쓰지 않은 나머지 수신자는 채택된 후보를 어차피 평가하므로(풀에 넣으려면 D 전체가
# 필요하다), 그 독립 표본으로 선별 결과를 확인한다. 시뮬레이션(수신자별 Δ의 SD 0.58, 수신자 10명): 선별만 하면 아무 개선이 없는 후보의
# 55%가 통과하지만, 확인(α=0.20)을 더하면 13%로 줄고 +0.29 개선의 검출력은 80% → 49%, +0.58은 96% → 88%다.

# 한쪽 검정 t 임계값 {α: {df: 값}}. 표에 없는 df는 그보다 작은 가장 가까운 표의 df 값(약간 보수적)을 쓰고 30을 넘으면 정규분포 값이다.
_T = {
    0.20: {1: 1.376, 2: 1.061, 3: 0.978, 4: 0.941, 5: 0.920, 6: 0.906, 7: 0.896, 8: 0.889, 9: 0.883, 10: 0.879, 11: 0.876, 12: 0.873,
           13: 0.870, 14: 0.868, 15: 0.866, 16: 0.865, 17: 0.863, 18: 0.862, 19: 0.861, 20: 0.860, 25: 0.856, 30: 0.854, "inf": 0.842},
    0.10: {1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476, 6: 1.440, 7: 1.415, 8: 1.397, 9: 1.383, 10: 1.372, 11: 1.363, 12: 1.356,
           13: 1.350, 14: 1.345, 15: 1.341, 16: 1.337, 17: 1.333, 18: 1.330, 19: 1.328, 20: 1.325, 25: 1.316, 30: 1.310, "inf": 1.282},
    0.05: {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782,
           13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725, 25: 1.708, 30: 1.697, "inf": 1.645},
    0.025: {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
            13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042, "inf": 1.960},
}
CONFIRM_ALPHAS = tuple(sorted(_T, reverse=True))   # 0.20, 0.10, 0.05, 0.025


def t_crit(alpha: float, df: int) -> float:
    """한쪽 검정 t 임계값(표). df < 1이면 확인할 수 없다."""
    if alpha not in _T:
        raise ValueError(f"alpha는 {CONFIRM_ALPHAS} 중 하나여야 합니다: {alpha}")
    if df < 1:
        raise ValueError("df는 1 이상이어야 합니다")
    row = _T[alpha]
    if df > 30:
        return row["inf"]
    return row[max(k for k in row if k != "inf" and k <= df)]


def passed_objectives(deltas: dict[str, list[float]]) -> list[str]:
    """verdict의 accept 조건(n ≥ 2이고 평균 > 표준오차)을 채운 목표들. 확인은 이 목표들만 본다."""
    out = []
    for o, d in deltas.items():
        m, se, n = mean_se(d)
        if n >= 2 and m > se:
            out.append(o)
    return out


def confirm(fresh: dict[str, list[float]], alpha: float) -> tuple[bool | None, dict]:
    """선별을 통과한 목표별로, 선별에 쓰지 않은 새 수신자들의 (자식 − 부모) 차이에 한쪽 t 검정을 한다.
    한 목표라도 통과하면 확인된 것(True). 확인할 수 있는 목표가 없으면(새 표본 2명 미만) None: 판정하지 않는다.
    돌려주는 info는 목표별 {n, mean, t, crit, ok}."""
    info: dict = {}
    for o, d in fresh.items():
        m, se, n = mean_se(d)
        if n < 2:
            info[o] = {"n": n, "ok": None}
            continue
        t = (math.inf if m > 0 else 0.0 if m == 0 else -math.inf) if se == 0 else m / se
        crit = t_crit(alpha, n - 1)
        info[o] = {"n": n, "mean": round(m, 4), "t": round(t, 3) if math.isfinite(t) else str(t), "crit": crit, "ok": bool(t > crit)}
    judged = [v["ok"] for v in info.values() if v["ok"] is not None]
    return (any(judged) if judged else None), info
