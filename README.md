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

기본 설정은 `optiq serve`로 띄운 로컬 서버(`http://127.0.0.1:8080/v1`)의 `mlx-community/Qwen3.5-9B-OptiQ-4bit:think`(thinking 켠 변형)를 쓴다. thinking 토큰도 `max_tokens`에 포함되므로 기본값을 2048로 두어 think 길이의 상한으로 쓴다(넘으면 대사가 비어 한 번 재시도한다. `--max-tokens`로 조절). `optiq serve`는 `sk-optiq-` 접두사의 키를 요구해서 자리표시 키 `sk-optiq-local`을 보낸다.

로컬 서버(`api_base`를 줄 때)에는 요청마다 `top_k: 20`, `presence_penalty: 0.5`도 함께 보낸다(기본 `temperature 1.0`, `top_p 0.95`는 Qwen3.5 thinking 권장값. 권장값 1.5에서는 외국어 단어가 섞여 0.5로 낮췄다). `optiq serve`는 이 필드를 받아들이지만 실제로 반영하는지는 확인하지 않았다.

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
turns: [{turn, speaker, name, text, latency_s, finish_reason, usage, reasoning}]
```

`end_reason`: `hangup_by_caller`, `hangup_by_callee`, `spam_reported_by_callee`(수신자가 `report_spam`으로 신고하고 끊음), `max_turns`, `repetition_loop`(같은 화자가 직전 발화를 되풀이), `empty_response`, `error`.

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
2. 쌓인 리뷰가 N건이 되면 `learner.learn`이 리뷰·현재 프롬프트·위키 목록을 보고 위키 도구와 `prompt_edit`(old_string→new_string 치환, Edit 도구처럼 한 곳씩)로 고친다. 다음 통화부터 새 프롬프트를 쓴다.

system.md는 편집 결과가 20자 미만이거나 600자를 넘거나 old_string이 정확히 한 번 나오지 않으면 거절된다. 신고가 아닌 통화는 리뷰 형식이 틀리면 한 번 다시 묻고, 그래도 못 읽거나 비었거나 max_tokens에서 잘리면 대기열에 넣지 않는다. `wiki/.readonly`(한 줄에 glob 하나, `#` 주석)에 적힌 문서는 학습자가 고칠 수 없다(예: `상품/*`). 회차마다 `agent_workspace/learning/round_NNN/`에 `system_before/after.md`, `wiki_before/after/`, `reviews.json`(달성률·신고율), `learner.json`을 남기므로 `*_before`를 복사해 되돌릴 수 있다. `uv run learner.py [--now]`로 대기열을 보거나 바로 학습한다. 기록에는 `review: {spam, achieved, notes, latency_s}`가 추가된다.

## 알려진 한계

- 4B 4비트 모델은 맥락을 놓치거나 상대의 말투와 이름을 따라 하는 일이 잦다. 9B가 훨씬 낫다.
- 9B 서버는 턴당 지연이 길고 들쭉날쭉하다(관측: 7~74초, 프롬프트 약 800~1100토큰). 한 통화에 몇 분이 걸린다. `--max-concurrent 1` 서버라 다른 요청과 겹치면 더 느려질 수 있다.
- 페르소나 두 명은 서로 무관하게 무작위로 뽑힌다. 통화 목적은 어떤 사람에게나 걸 수 있는 내용(안부, 설문, 영업 등)이어야 자연스럽다. 예를 들어 "치과에 예약 문의"처럼 수신자의 직업을 전제하는 목적은 수신자가 건물 청소원일 수 있어 어긋난다.
