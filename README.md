# auto-prompt-learning

[Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea)에서 페르소나 한 명을 뽑아 수신자로 만들고, 발신자 에이전트가 전화를 거는 상황을 litellm으로 텍스트 시뮬레이션한다. 환경(텔레마케팅, 설문, 안내 전화 등)은 `prompt/system.md`와 `wiki/`로 정하며, 기본 예시는 텔레마케팅이다. 통화 기록은 JSON으로 저장된다.

- 발신자는 페르소나 없이 `prompt/system.md`를 시스템 프롬프트로 쓰고, `--purpose`로 준 통화 목적을 안다. 수신자는 모르는 번호로 전화를 받는다.
- 통화 전에 발신자가 계획을 한 번 세우고(`--plan-max-tokens`, 기본 8192), 그 계획이 시스템 프롬프트의 `[Plan]`으로 붙어 통화 내내 참고된다. 수신자 정보 중 `personas.CALLER_KNOWN_FIELDS`(지금은 성별)만 시스템 프롬프트의 `[Callee]`로 받는다.
- 수신자가 먼저 고정 대사 "여보세요?"로 받고(모델 호출 없음), 이후 번갈아 말한다. 한쪽이 `end_call` 도구(tool calling)를 호출하면 통화가 끝난다.

## 환경 구성

통화 환경은 아래 네 가지로만 정해진다. 코드에는 환경별 문구가 없다.

| 요소 | 역할 |
|---|---|
| `prompt/system.md` | 발신자의 역할과 행동 원칙 |
| `wiki/` | 발신자가 참고하는 지식(상품 정보, 응대 요령, 설문 지침 등) |
| `wiki/.readonly` | 학습자가 고칠 수 없는 문서 glob |
| `--purpose` | 이번 통화의 목적 |

예를 들어 구청 설문 조사로 바꾸려면 `system.md`를 "구청 설문 조사원. 처음에 소속과 이름을 밝히고 [Goal]의 설문을 진행한다…"로 고치고, `wiki/`에 설문 문항 문서를 두고, `--purpose "주민 만족도 설문"`으로 실행한다. 변하지 않는 것은 전화 통화, 발신자가 페르소나에게 모르는 번호로 거는 방향, 수신자의 "여보세요?", 한국어 음성 채널이다.

## 준비

```
uv sync
```

데이터셋(`nvidia/Nemotron-Personas-Korea`)은 Hugging Face 캐시에 없으면 처음 실행할 때 내려받는다(약 4GB).

## 로컬 서버

모델 설정은 `.env`에서 읽는다(`cp .env.example .env`로 만든다. `LLM_MODEL`, `LLM_API_BASE`, `API_KEY`). `.env.example`의 기본값은 `optiq serve`로 띄운 로컬 서버(`http://127.0.0.1:8080/v1`)의 `mlx-community/Qwen3.5-9B-OptiQ-4bit:think`(thinking 켠 변형)이다. thinking 토큰도 `max_tokens`에 포함되므로 기본값을 2048로 두어 think 길이의 상한으로 쓴다(넘으면 대사가 비어 한 번 재시도한다. `--max-tokens`로 조절). `optiq serve`는 `sk-optiq-` 접두사의 키를 요구해서 `API_KEY`의 자리표시 키(`sk-optiq-local`)를 보낸다.

로컬 서버(`api_base`를 줄 때)에는 요청마다 `top_k: 20`, `presence_penalty: 0.5`도 함께 보낸다(기본 `temperature 1.0`, `top_p 0.95`는 Qwen3.5 thinking 권장값. 권장값 1.5에서는 외국어 단어가 섞여 0.5로 낮췄다. temperature는 `.env`의 `LLM_TEMPERATURE`나 `--temperature`로 바꾼다. DLM-2.0-14B에서는 1.0일 때 비문과 같은 말 되풀이가 잦아 0.6이 나았다). `optiq serve`는 이 필드를 받아들이지만 실제로 반영하는지는 확인하지 않았다.

그 밖의 선택 설정은 모두 `.env`에서 바꾼다(자세한 설명과 근거는 `.env.example`): `LLM_EXTRA_BODY`(서버 전용 샘플링 필드 덮어쓰기), `CALLER_MAX_SENTENCES`·`CALLEE_MAX_SENTENCES`(발화 문장 수 제한, 기본 2), `PERSONA_BACKGROUND`·`PERSONA_BACKGROUND_FIELDS`(수신자 프롬프트에 넣을 페르소나 서술 필드, 기본은 전부), `LOOP_NUDGE`(같은 말이 되풀이될 때 한 번 끝내라는 힌트를 주는 기능, 기본 켜짐), `CALLEE_PARTNER_LABEL`(수신자에게 가는 상대 말 앞에 붙이는 라벨, 기본 꺼짐). 서버가 빈 응답(완성 토큰 0)을 돌려주면 통화는 `error`로 끝나고, 서버 오류 통화가 연속 3개면 배치를 멈춘다.

통화 종료에 tool calling을 쓴다. `mlx_lm.server`는 `tools`를 `tool_calls`로 파싱해 준다. 서버가 파싱하지 않고 content에 `<tool_call>` 블록을 남기면 그 블록에서 `end_call`을 찾고 발화에서는 지운다.

`mlx_lm.server`로 더 작은 모델을 돌릴 수도 있다. 이 서버는 모델의 `generation_config.json`을 읽지 않아서 `presence_penalty`를 빼면 같은 말을 되풀이하는 루프에 빠지기 쉽다(4B에서 확인).

```
uv run mlx_lm.server --model mlx-community/Qwen3.5-4B-OptiQ-4bit --port 8081 \
  --chat-template-args '{"enable_thinking": false}'

uv run main.py --purpose "..." --model openai/mlx-community/Qwen3.5-4B-OptiQ-4bit \
  --api-base http://127.0.0.1:8081/v1
```

## 실행

```
uv run main.py --purpose "월 9,900원 실손보험 가입 권유" --seed 1
```

| 옵션 | 설명 |
|---|---|
| `--purpose` | (필수) 발신자의 통화 목적 |
| `--caller-prompt` | 발신자 시스템 프롬프트 파일 (기본: 저장소의 `prompt/system.md`) |
| `--callee-uuid` | 수신자 페르소나를 지정. 없으면 무작위 |
| `--seed` | 페르소나 샘플링 시드 |
| `--n` | 페르소나를 새로 뽑아 반복할 통화 수 |
| `--max-turns` | 최대 턴 수 (기본 30) |
| `--model`, `--api-base` | litellm 모델 문자열과 엔드포인트 |
| `--temperature`, `--max-tokens` | 샘플링 설정 |
| `--plan-max-tokens` | 통화 전 계획 생성의 max_tokens (기본 8192, 0이면 계획 생략) |
| `--no-wiki` | 발신자의 위키 도구 사용을 끈다 |
| `--out` | 기록 저장 폴더 (기본 `agent_workspace/conversation_history`) |

호스팅 모델을 쓰려면 `--api-base ""`로 `api_base`와 MLX 전용 옵션을 끈다.

```
uv run main.py --purpose "..." --model gemini/gemini-2.5-flash --api-base ""
```

## 출력

통화마다 `{시작시각}_agent-{수신자uuid6}_{call_id}.json`이 저장된다.

```
call_id, started_at, ended_at, purpose, end_reason, num_turns,
llm: {model, api_base, temperature, top_p, max_tokens, plan_max_tokens, extra_body},
caller: {uuid: null, name, system_prompt([Plan] 포함), persona: null, plan: {text, latency_s, finish_reason, usage, reasoning} | null (비었거나 max_tokens에서 잘리면 null)}
callee: {uuid, name, system_prompt, persona(원본 26개 필드)},
turns: [{turn, speaker, name, text, latency_s, finish_reason, usage, reasoning, [ignored_end_call]}]
```

`end_reason`: `hangup_by_caller`, `hangup_by_callee`, `spam_reported_by_callee`(수신자가 `report_spam`으로 신고하고 끊음), `max_turns`, `repetition_loop`(같은 화자가 직전 발화를 되풀이), `empty_response`, `error`.

작은 모델이 `end_call`을 엉뚱하게 부르는 경우를 `PhoneAgent.respond`가 다듬는다. 각 화자의 첫 발화와 질문(`?`)으로 끝나는 발화에서 부른 `end_call`은 끊김으로 치지 않고 무시한다(턴 기록의 `ignored_end_call`에 횟수). 그 밖에는 말없이 `end_call`만 불러도 그대로 끊긴다(대사가 빈 `hangup` 턴). 도구가 하나도 없는 요청은 보내지 않는다(4B think 모델이 장문 사고로 흘러 `max_tokens`에서 잘려 대사가 빈다).

## 구조

| 파일 | 역할 |
|---|---|
| `personas.py` | 데이터셋 로드, 샘플링, 프로필 렌더링 |
| `prompts.py` | 전화 규칙과 프롬프트 조립 (발신자 본문은 `prompt/system.md`) |
| `agent.py` | `LLMConfig`, `PhoneAgent` (litellm 호출, 출력 정리) |
| `phone_call.py` | 통화 루프, 종료 판정, 기록 저장 |
| `reviewer.py` | 통화 평가 (스팸 신고 여부, 목적 달성 여부) |
| `wiki_store.py` | 발신자 위키(`wiki/`) CRUD, 임베딩·파일명 검색, 에이전트 도구 스키마, CLI |
| `main.py` | CLI |

## 위키

`wiki/`에 마크다운 문서를 두면 발신자가 참고할 지식 베이스가 된다(하위 폴더 가능). 통화 중 발신자는 `wiki_read`, `wiki_search`, `wiki_find` 도구를 쓸 수 있다(수신자는 제외). 위키에 문서가 있으면 통화 전에 인덱스를 맞추고 모델을 미리 올리며, 통화 전 계획 단계에서도 같은 도구로 조회할 수 있다. 계획 한 번, 발화 한 번마다 도구는 최대 3번 부른다. `--no-wiki`로 끈다. 쓰기 도구(`wiki_create/update/delete`)는 `WIKI_TOOLS`와 `run_tool`에 있지만 발신자에게는 주지 않는다. 조회 내역은 턴 기록과 `caller.plan`의 `wiki`에 남는다.

- 임베딩 검색은 `mlx-community/embeddinggemma-2-bf16`(첫 사용 시 약 1.5GB 다운로드)을 쓴다. 문서를 헤딩 단위 청크로 나눠 `wiki/.index/`(git 제외)에 저장하고, 검색 때마다 바뀐 파일만 다시 임베딩한다. 에디터로 직접 고쳐도 반영된다.
- 파일명 검색은 모델 없이 부분일치, glob(`상품/*`), 오타 허용 매칭을 지원한다.
- 경로는 `wiki/` 안으로 제한되고 숨김 경로(`.index` 등)는 거부한다.

```
uv run wiki_store.py create 상품/실손보험 --file draft.md   # 또는 stdin
uv run wiki_store.py search "보험료가 얼마예요" -k 3
uv run wiki_store.py find 실손
uv run wiki_store.py update 상품/실손보험 --append < note.md
uv run wiki_store.py read|delete|list|reindex
```

## 학습

통화 `--learn-every`(기본 20, 0이면 끔)건마다 발신자의 `prompt/system.md`와 `wiki/`를 자동으로 고친다. 같은 모델이 두 단계로 일한다.

1. 통화 직후 `reviewer.review_call`이 평가(스팸 신고 여부는 `end_reason`으로, 목적 달성 여부와 근거 메모는 모델이)를 만들어 기록의 `review`에 저장하고 `agent_workspace/learning/pending.jsonl`에 쌓는다. 스팸 신고된 통화는 모델 응답과 무관하게 항상 대기열에 들어간다. 여러 번에 나눠 실행해도 이어서 센다(옛 형식 행은 경고를 내고 버린다).
2. 쌓인 리뷰가 N건이 되면 `learner.learn`이 리뷰·현재 프롬프트·위키 목록을 보고 위키 도구와 `prompt_edit`(줄 ID로 replace·insert_after·delete, 학습자에게는 줄마다 영문 소문자+숫자 무작위 4자리 ID `[k7x2]`를 붙여 보여준다. 고쳐도 다른 줄의 ID는 바뀌지 않는다)로 고친다. 다음 통화부터 새 프롬프트를 쓴다.

system.md는 편집 결과가 20자 미만이거나 600자를 넘거나 없는 줄 ID를 쓰면 거절된다. 신고가 아닌 통화는 리뷰 형식이 틀리면 한 번 다시 묻고, 그래도 못 읽거나 비었거나 max_tokens에서 잘리면 대기열에 넣지 않는다. `wiki/.readonly`(한 줄에 glob 하나, `#` 주석)에 적힌 문서는 학습자가 고칠 수 없다(예: `상품/*`). 회차마다 `agent_workspace/learning/round_NNN/`에 `system_before/after.md`, `wiki_before/after/`, `reviews.json`(달성률·신고율), `learner.json`을 남기므로 `*_before`를 복사해 되돌릴 수 있다. `uv run learner.py [--now]`로 대기열을 보거나 바로 학습한다. 기록에는 `review: {spam, achieved, notes, latency_s}`가 추가된다.

### 러너는 시나리오를 모른다

`learner.py`·`reviewer.py`·`prompts.py`·`scripts/`에는 특정 시나리오(상품, 설문 등)의 문구가 없다. 시나리오는 `prompt/system.md`, `wiki/`, `--purpose`로만 정한다. 러너가 따르는 규약은 두 가지다.

- **사실의 근거는 읽기 전용 문서(`wiki/.readonly`), `--purpose`, 현재 `system.md`뿐이다.** 시나리오의 사실(금액, 기간, 조건 등)은 읽기 전용 문서에 둔다. 러너는 이 밖의 숫자와 사실 주장을 학습 편집에서 막는다.
- **성공은 `achieved`(상대가 목적의 요청에 명확히 응함)이고, 진행 단계 `stage`(0~4)는 달성이 없어도 개선을 재는 보조 신호다.** 0 용건 전에 끝남, 1 용건 전달, 2 상대 관심·질문, 3 발신자가 목적에 필요한 요청을 직접 함, 4 상대가 응함.

러너의 안전장치(모두 시나리오와 무관):

| 장치 | 동작 |
|---|---|
| 수락 근거 검증 | 리뷰어가 `achieved`와 함께 상대 발화를 인용해야 하고, 인용이 기록에 없으면 달성을 취소 |
| 숫자 가드 | 읽기 전용 문서·목적·현재 문서에 없는 숫자가 든 편집을 거절 |
| 복사 가드 | 달성 못 한 통화에서 발신자가 한 말을 12자 이상 그대로 옮기는 편집을 거절(비문을 규칙으로 굳히지 않게) |
| 따옴표 문구 가드 | 따옴표로 묶은 새 문구는 발신자가 통화에서 그대로 말하는 대사라서, 사실 검사를 믿을 수 있게 하지 못하면 거절(이미 문서에 있던 문구는 제외). 이력 요약의 따옴표 문구는 학습자가 그대로 다시 쓰므로 모든 회차에서 지워서 보여 줌 |
| 주소·연락처 가드 | 근거 문서에 없는 URL·이메일이 든 편집을 거절 |
| 위키 쓰기 끔(기본) | 학습자는 `system.md`만 고친다. 위키에 쓴 내용은 발신자가 통화에서 그대로 따라 말하는데, 14B 학습자가 쓴 내용은 5회 실측에서 매번 지어낸 사실·중복이었고 사실 검사기도 다 막지 못했다. `.env`에 `LEARN_WIKI_WRITES=1`이면 켠다(그때는 검사기를 신뢰할 수 없는 쓰기를 거절) |
| 편집 상한 | 한 회차에 고칠 수 있는 횟수(`MAX_EDITS`, 기본 3)를 두고, 위키 쓰기 결과에 현재 문서를 보여 줘 같은 내용을 되풀이해 덧붙이지 않게 함 |
| 사실 주장 검사 | 근거 없는 사실 주장이 든 편집을 거절. 근거가 확실한 문장을 섞어 검사기의 과잉 지적을 가려내고, 두 번 모두 지적한 줄만 막는다(신뢰할 수 없으면 막지 않음) |
| 위키 감사 | 학습 시작 전 고칠 수 있는 위키 문서에서 근거 없는 사실 주장 줄을 정리 |
| 학습자 입력 | 통화 기록 발췌에 마지막의 말없는 끊김을 `(말없이 끊음)`으로 보여 줘, 질문에 답하지 않고 끊은 통화를 학습자가 볼 수 있게 함. 리뷰 노트(14B가 쓴 요약)는 틀린 진단이 섞여 보여 주지 않음 |
| 발언 검사 | 통화별 발신자의 근거 없는 발언을 찾아 학습자에게 보여 주고 `invented_rate`로 집계 |
| 자동 롤백 | 직전 회차 변경 뒤 묶음이 달성률 −0.2 이상, 또는 달성이 늘지 않은 채 평균 단계 −0.6 이상 나빠지면 학습 전에 되돌림 |
| 실패 회차 되돌림 | 학습자 호출이 끝까지 못 가면(서버 300초 제한 등) 반쯤 적용된 편집을 되돌리고 도구 호출 기록을 남김 |

**리뷰어도 시나리오와 무관해야 한다.** 리뷰어(`reviewer.py`와 `prompts.py`의 리뷰 문구)를 고칠 때는 `uv run scripts/review_bench.py`로 서로 다른 시나리오(판매·설문·예약 확인·일정 협의) × 유형(수락·유보·관심만·거절·조기 종료) 20건에 채점해 달성 정확도·수락 놓침·거짓 달성·단계 정확도가 같거나 올라야 채택한다. 사례는 `scripts/review_bench_cases.json`이고, 새 시나리오 사례를 더해 한 시나리오에 맞춰지는 것을 막는다.
현재 리뷰 문구(14B, 20건): 달성 정확도 100%, 수락 놓침 0/4, 거짓 달성 0, 단계 정확 85%(±1 이내 100%). 판정 기준을 고치기 전(수락 놓침 1/3, 단계 정확 50%)보다 모든 지표가 올랐다. 리뷰어를 바꾸면 단계·달성 수치의 척도가 달라지므로, 학습 회차 사이의 비교(자동 롤백)는 같은 리뷰어로 잰 묶음끼리 해야 한다.

`uv run learner.py --audit [--fix]`(위키 감사), `--rollback N [--only system|wiki]`(그 회차 변경을 복원, 일부만 고를 수 있음), `--requeue N`(그 회차 리뷰를 대기열에 다시 넣기), `--plan-max-tokens`(학습자 응답 길이; 서버에 요청당 시간 제한이 있으면 줄인다). `--now`는 학습자 호출이 실패해도 대기열을 남기므로 같은 명령으로 다시 돌릴 수 있다.

## 알려진 한계

- 4B 4비트 모델은 맥락을 놓치거나 상대의 말투와 이름을 따라 하는 일이 잦다. 9B가 훨씬 낫다.
- 9B 서버는 턴당 지연이 길고 들쭉날쭉하다(관측: 7~74초, 프롬프트 약 800~1100토큰). 한 통화에 몇 분이 걸린다. `--max-concurrent 1` 서버라 다른 요청과 겹치면 더 느려질 수 있다.
- 리뷰어(14B)의 진행 단계는 수신자가 용건을 먼저 짐작해 거절·종료한 통화를 한 단계 높게(0 대신 1) 매기는 경향이 있다(`scripts/review_bench.py`의 '추측 거절' 사례 0/4). 판정 문구를 조이거나(후보 D: 이 4건은 고쳤지만 기존 사례 17→12), 좁은 예/아니오 질문 세 개로 쪼개 코드가 단계를 계산해도(후보 E: 14/24) 나아지지 않았다. 달성 판정(100%)과 ±1 이내(100%)는 정확하므로, 같은 리뷰어로 잰 수치끼리만 비교하고 단계의 절대값은 믿지 않는다.
- 페르소나 두 명은 서로 무관하게 무작위로 뽑힌다. 통화 목적은 어떤 사람에게나 걸 수 있는 내용(안부, 설문, 영업 등)이어야 자연스럽다. 예를 들어 "치과에 예약 문의"처럼 수신자의 직업을 전제하는 목적은 수신자가 건물 청소원일 수 있어 어긋난다.
