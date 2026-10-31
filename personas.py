import ast
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


# 직업·특기를 길게 서술한 필드까지 다 넣으면 모델이 그 어휘(예: "데이터 분석")를 모든 답에
# 되풀이한다. 일상 대화에 필요한 정체성 필드만 프롬프트에 넣는다.
_PROFILE_FIELDS = (
    "persona", "sex", "age", "marital_status", "family_type", "housing_type",
    "education_level", "occupation", "district",
)


# 발신자(상담원)가 통화 전에 아는 수신자 정보.
CALLER_KNOWN_FIELDS = ("sex",)


def render_known(persona: dict) -> str:
    data = {k: persona[k] for k in CALLER_KNOWN_FIELDS}
    return yaml.dump(data, allow_unicode=True, sort_keys=False)


def render_profile(persona: dict) -> str:
    data = {k: persona[k] for k in _PROFILE_FIELDS}
    return yaml.dump(data, allow_unicode=True, sort_keys=False)
