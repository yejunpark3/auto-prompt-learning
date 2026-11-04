import ast
import os
import re
from functools import cache

import pyarrow.compute as pc
import pyarrow.dataset as ds
import yaml
from huggingface_hub import snapshot_download

DATASET_ID = "nvidia/Nemotron-Personas-Korea"

_NAME_RE = re.compile(r"^(\S+?)\s*씨")


@cache
def open_dataset() -> ds.Dataset:
    path = snapshot_download(
        DATASET_ID, repo_type="dataset", allow_patterns=["data/*.parquet"]
    )
    return ds.dataset(f"{path}/data", format="parquet")


@cache
def _num_rows() -> int:
    return open_dataset().count_rows()


def parse_value(value):
    """`"['a', 'b']"` 형태의 문자열 열을 list로 바꾼다. 실패하면 원문을 돌려준다."""
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _row(table) -> dict:
    row = table.to_pylist()[0]
    for key in ("skills_and_expertise_list", "hobbies_and_interests_list"):
        row[key] = parse_value(row[key])
    return row


def sample_persona(rng, exclude_uuid: str | None = None) -> dict:
    dataset = open_dataset()
    while True:
        row = _row(dataset.take([rng.randrange(_num_rows())]))
        if row["uuid"] != exclude_uuid:
            return row


def get_persona(uuid: str) -> dict:
    table = open_dataset().to_table(filter=pc.field("uuid") == uuid)
    if table.num_rows == 0:
        raise KeyError(f"persona not found: {uuid}")
    return _row(table)


def persona_name(persona: dict) -> str:
    m = _NAME_RE.match(persona["persona"])
    return m.group(1) if m else persona["uuid"][:6]


def summarize(persona: dict) -> str:
    return (
        f"{persona_name(persona)} ({persona['age']}세 {persona['sex']}, "
        f"{persona['district']}, {persona['occupation']})"
    )


# 수신자 프롬프트에는 데이터셋의 페르소나 필드를 uuid(식별자)만 빼고 전부 넣는다.
# 정체성 필드(_LEADING_FIELDS)는 [Persona]에, 나머지 서술 필드는 [Background]에 나눠 넣는다.
# 이력: 예전 4B 모델은 직업·특기를 길게 서술한 필드까지 넣으면 그 어휘(예: "데이터 분석")를 모든 답에
# 되풀이해서 정체성 필드만 넣었다. 14B로 바꾼 뒤 한 턴 블라인드 벤치(맥락 9개×2샘플, 0~2점)에서
# 전체를 한 블록에 넣으면 1.28→0.61, 서술 필드를 [Style] 뒤 [Background]로 나누면 1.00(표준오차 이내)이었다.
# 되풀이나 비문이 다시 늘면 필드를 빼기 전에 이 이력을 먼저 볼 것.
_LEADING_FIELDS = (
    "persona", "sex", "age", "marital_status", "family_type", "housing_type",
    "education_level", "occupation", "district",
)
_EXCLUDED_FIELDS = ("uuid",)


# 발신자가 통화 전에 아는 수신자 정보.
CALLER_KNOWN_FIELDS = ("sex",)


def render_known(persona: dict) -> str:
    data = {k: persona[k] for k in CALLER_KNOWN_FIELDS}
    return yaml.dump(data, allow_unicode=True, sort_keys=False)


def render_profile(persona: dict) -> str:
    """정체성 필드. 수신자 프롬프트의 [Persona]에 들어간다."""
    data = {k: persona[k] for k in _LEADING_FIELDS}
    return yaml.dump(data, allow_unicode=True, sort_keys=False)


def render_background(persona: dict) -> str:
    """[Background]에 넣을 서술 필드. 수신자 프롬프트의 [Style] 뒤에 들어간다.

    기본은 정체성·uuid를 뺀 서술 필드 전부다(사용자 요청: 페르소나 내용을 되도록 모두).
    .env의 PERSONA_BACKGROUND=0이면 비워서 정체성 필드 9개만 넣고,
    PERSONA_BACKGROUND_FIELDS=family_persona,hobbies_and_interests처럼 필드를 골라 넣을 수도 있다(우선).
    이력(14B, 시드 31, 같은 수신자 8명): 힌트 없이 전체를 넣은 팔이 9필드보다 수신자 발화 채점 1.24→0.82,
    repetition_loop 2→4였으나, 같은 설정을 다시 돌려도 이만큼 달라지는 실행 간 변동이 있었고
    (3개 추가 팔 0.62 대 3개 추가+힌트 팔 1.25), 힌트를 켠 전체 팔은 9필드와 같은 수준(1.29 대 1.31)이었다."""
    picked = [f.strip() for f in os.environ.get("PERSONA_BACKGROUND_FIELDS", "").split(",") if f.strip()]
    if picked:
        data = {k: persona[k] for k in persona if k in picked and k not in _LEADING_FIELDS and k not in _EXCLUDED_FIELDS}
    elif os.environ.get("PERSONA_BACKGROUND", "1") != "0":
        data = {k: v for k, v in persona.items() if k not in _LEADING_FIELDS and k not in _EXCLUDED_FIELDS}
    else:
        return ""
    return yaml.dump(data, allow_unicode=True, sort_keys=False) if data else ""
