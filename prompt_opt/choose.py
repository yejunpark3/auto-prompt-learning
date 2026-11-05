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
