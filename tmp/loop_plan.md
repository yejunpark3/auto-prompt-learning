# Loop 계획: 실손보험 텔레마케팅 자기개선 루프

목적: 월 9,900원 실손보험 가입자 최대화.
우선순위: ① 대화 자연스러움 > ② 스팸 신고 최소화 > ③ 목적 달성(가입) 최대화.
상위 지표를 해치는 하위 지표 개선은 채택하지 않는다.

## 0. 측정과 역할 분담
- 리뷰어(`reviewer.py`)는 통화마다 두 값만 낸다: `spam`(end_reason이 spam_reported_by_callee) / `achieved`(목적 달성 t/f, 신고 통화는 항상 f) + 근거 메모.
- 자연스러움은 리뷰어가 재지 않는다. 매 회차 사람(=loop를 돌리는 Claude)이 transcript를 직접 읽고 판단한다.
- 회차 지표: `spam_rate`, `achieved_rate`(`reviews.json`에도 기록), 종료 사유 분포, 평균 턴, 발신자 평균 발화 길이.

## 1. 원칙 (매 회차 지킨다)
- 프롬프트 최소화: 자연스러움 문제는 모델 설정·코드로 먼저 고친다. 프롬프트 추가는 한 줄 이내, 규칙 목록 금지.
  - Claude가 넣는 한 줄은 `prompts.py`의 `caller_prompt`(발신자 전용)에 넣는다. `prompt/system.md`는 학습자가 `prompt_write`로 통째로 다시 써서 지워질 수 있고 롤백 때도 되돌아간다. `CHANNEL`은 수신자 프롬프트에도 들어가므로 건드리지 않는다.
- 공정성: 지표를 위해 수신자(callee_prompt, report_spam 도구 설명), 리뷰어 기준, 페르소나 샘플링을 느슨하게 바꾸지 않는다. 수신자·리뷰어 쪽 수정은 현실성 버그(역할 혼동, 형식 오류)일 때만.
  - 공통 경로를 고칠 때는 발신자에게만 적용되게 한다. `main.py`는 하나의 `llm`을 발신자·수신자·리뷰어·학습자에게 같이 넘기고, `clean_utterance`·`respond`, `WRAP_UP_HINT`(다음 화자 누구에게나 붙는다), `CHANNEL`도 수신자와 공통이다. 샘플링은 `phone_call.build_agents`에서 발신자에게만 `dataclasses.replace(llm, ...)`로 주고, 발화 정리·마무리는 `role == "caller"`일 때만 바꾼다.
- 사실 날조 금지: 위키(`상품/*` 읽기 전용)와 목적에 없는 숫자·연락처·URL·회사명 지어내기, 수신자 정보 유출(알 수 없는 이름·직업 언급)은 자연스러움 결함으로 센다.
- 회차당 Claude의 변경은 1개(코드 1건 또는 learner 1건). 원인과 효과를 구분하기 위해.
  - 단, 매 배치 끝에 자동 학습이 프롬프트·위키를 한 번 더 바꾸므로 한 회차의 실제 변경은 2개(Claude 변경 + 자동 학습)다. 지표 변화는 둘의 합으로 해석하고, 로그에는 두 변경을 각각 적는다.
  - `learner.py` 변경은 한 배치 늦게 보인다. 배치 k를 분석할 때는 round_{k+1}이 이미 돌았으므로, 바뀐 learner는 배치 k+1 끝의 round_{k+2}부터 쓰이고 효과는 배치 k+2에서 보인다. 그래서 learner를 바꾼 다음 회차(배치 k+1 분석)에는 Claude가 아무것도 바꾸지 않고 관찰만 하며, 로그에 learner 변경이 처음 쓰인 round 번호를 적는다.
  - 롤백은 프롬프트·위키만 되돌린다. 코드 변경(`caller_prompt`에 넣은 한 줄 포함)은 남으므로, 롤백할 때 그 회차의 코드 변경이 악화에 기여했는지 따로 판단해 로그에 "코드 유지/되돌림"과 이유를 적는다(코드는 사용자가 커밋하지 않았다면 `git diff`로 확인해 되돌린다).
- 통화는 Claude가 직접 실행한다(백그라운드 Bash, 출력은 파일로). 한 번에 한 배치만 돌리고, 배치 도중에는 학습·리뷰어 점검을 겹쳐 돌리지 않는다. 커밋은 사용자가 요청할 때만.
- 변경과 관찰은 `tmp/loop_log.md`에 기록.

## 2. 준비 (루프 시작 전 1회)
1. [완료] 리뷰어 구현: `reviewer.py`, `learner.py`(`spam`·`achieved` 저장, `reviews.json` 비율), `prompts.py`(`REVIEW_REQUEST`, `SUBMIT_REVIEW_TOOL`, `LEARN_REQUEST` 우선순위 한 줄), `main.py` 연결.
2. 지표 스크립트 `tmp/loop_stats.py`: `--last N`으로 최근 N개 기록의 end_reason 분포, spam_rate, achieved_rate, 평균 턴, 발신자 평균 발화 길이(문장 수·글자 수), 루프성 종료(repetition_loop·max_turns·empty_response) 수를 출력. 서버 오류(`error`)는 따로 센다. `review`가 없는 기록은 건너뛰고 개수를 표시.
3. 리뷰어 점검(기준선 배치 직후, 첫 학습 전 1회): 기준선 20통화를 `reviewer.review_call`로 다시 돌려 `achieved`가 사람이 읽은 판정과 맞는지, 같은 기록을 2번 돌려도 같은지 확인. 불일치가 크면 `SUBMIT_REVIEW_TOOL`의 `achieved` 설명(달성 기준)을 고친다.
   - 도구: `uv run tmp/review_check.py --last 20` (기록당 2번 판정, 불일치 표시). 리뷰 40회(회당 최대 8192토큰)라 Bash 10분 제한을 넘기기 쉬우므로 Claude가 백그라운드로 돌리고 출력을 파일로 받는다. 같은 서버를 나눠 쓰므로 통화 배치가 도는 동안에는 돌리지 않는다.
   - 리뷰어는 `submit_review` 도구 호출(`achieved`: boolean, `notes`)로 판정을 낸다. 도구를 안 부르거나 인자 타입이 틀리면 재시도 후 리뷰 실패(None)로 처리하므로, 점검 때 "(리뷰 실패)"가 잦으면(서버가 도구 호출을 못 하는 것일 수 있다) `REVIEW_REQUEST`·`SUBMIT_REVIEW_TOOL` 설명을 손본다.
   - 기준이 맞으면 `uv run learner.py --now`로 대기열 20건을 학습해 round_001을 만든다.
   - 기준을 고쳤으면 옛 기준의 리뷰로 학습하지 않도록 `agent_workspace/learning/pending.jsonl`을 지우고 기준선 배치를 다시 돌린다(`loop_stats --last 20`은 새 배치만 본다).
4. 기준선: 재작성한 프롬프트·위키(한빛손해보험, 읽기 전용 `상품/*`·`응대/거절대응.md`)로 Claude가 돌린 첫 20통화만 쓴다. 이전 기록은 삭제했다. 리뷰어 점검 전에 학습이 돌지 않도록 이 배치만 `--learn-every 21`로 돌린다(리뷰는 대기열에 쌓이고 학습은 3번 점검 뒤로 미룬다).
   - `uv run main.py --purpose "월 9,900원 실손보험 가입 권유" --n 20 --learn-every 21 --seed 1`
   - 지표는 `tmp/loop_log.md` 0회차로 기록.
   - 이전 기록에서 이미 보인 문제(발화 5문장 안팎, 역할 혼동, 회사명·사실 날조)는 1회차 진단표 첫 줄에 해당하므로 첫 변경 후보다.

## 3. 회차 절차
Claude가 실행한 통화 묶음마다 아래를 수행한다(배치는 `run_in_background`로 돌리고, 종료 알림이 오면 분석한다). 묶음은 20통화이고, 배치마다 같은 수신자 20명으로 비교하도록 시드를 고정한다(기준선과 같은 `--seed 1`).
- `uv run main.py --purpose "월 9,900원 실손보험 가입 권유" --n 20 --learn-every 20 --seed 1`
- 회차 번호: 배치 k(k≥1)는 round_k의 학습 결과로 돌고, 배치 k의 20번째 통화에서 round_{k+1}이 자동으로 돈다. 롤백하면 이 대응이 어긋나므로 로그에 배치마다 "어느 프롬프트로 돌았는지"(round 번호 또는 롤백 지점)를 적는다.
1. 측정: `uv run tmp/loop_stats.py --last 20`. 직전 회차와 비교. `error` 종료(서버·연결 오류)는 루프성 종료와 따로 표시되며, 진단에 쓰지 않고 먼저 처리한다. `main.py`는 이런 통화를 리뷰·대기열에 넣지 않으므로 그 배치는 대기열이 20건에 못 미쳐 학습이 돌지 않은 상태다.
   - 서버를 점검하고, `loop_stats`가 보여 준 오류 기록을 옮긴다: `mkdir -p agent_workspace/errors && mv agent_workspace/conversation_history/<파일> agent_workspace/errors/`. 그래야 `--last 20`이 배치의 정상 통화만 센다.
   - Claude가 오류 통화마다 같은 수신자로 다시 돌린다: `uv run main.py --purpose "월 9,900원 실손보험 가입 권유" --n 1 --learn-every 20 --callee-uuid <uuid>`(기준선 배치면 `--learn-every 21`). 마지막 재실행이 대기열 20번째를 채우면 학습이 그때 돈다.
   - 오류가 절반 이상이면(서버 다운) 그 배치 기록을 모두 옮기고 `pending.jsonl`을 지운 뒤 배치를 처음부터 다시 돌린다.
   - 재실행이 끝나면 `loop_stats --last 20`으로 다시 측정한다.
2. 정독(자연스러움 판단): 가장 길거나 이상한 통화 2~3건, 신고 1건(있으면), 달성 1건(있으면)의 transcript와 review를 읽는다. 확인 항목: 한 발화 길이, 같은 말 반복, 역할 혼동(자기 이름 부르기 등), 날조, 수신자 정보 유출, 끊을 때를 못 맞춤.
3. 학습 결과 확인: `round_NNN/learner.json`, `system_before/after.md`, 위키 diff에서 날조 사실 기입, 빈 학습, 300자 거절 반복, 읽기 전용 침범 여부.
4. 진단 → 변경 1개 (위에서부터 처음 걸리는 것)
   | 조건 | 다룰 곳 |
   |---|---|
   | 루프성 종료, 역할 혼동, 발화 3문장↑, 날조, 정보 유출 | 코드(발신자에게만, 1절): `agent.py`(clean_utterance), `phone_call.py`(발신자 샘플링, 종료·마무리), `prompts.py` `caller_prompt` 한 줄 |
   | spam_rate > 10%(20통화 중 3건↑) 또는 직전 대비 4건↑ 상승 | `learner.py` (후보 A·B) |
   | 위 둘 정상 | `learner.py`/위키로 달성률 (후보 C·D) |
5. 게이트: 학습 직후 묶음에서 spam_rate가 직전 배치보다 +20%p(20통화 중 4건) 이상 오르거나, 자연스러움이 눈에 띄게 나빠졌으면 되돌리고 로그에 "롤백"과 되돌린 round 번호를 기록.
   - 되돌릴 곳은 최신 round(N)가 아니라 그 바로 전 round(N-1)의 `*_before`다. 평가 중인 배치는 round N-1이 만든 프롬프트로 돌았고, round N은 그 나쁜 배치의 끝에서 이미 돌아 그 위에 학습한 것이라(N의 `system_before.md`가 곧 나빠진 프롬프트) 함께 버린다.
   - 명령(NNN = N-1):
     ```
     cp agent_workspace/learning/round_NNN/system_before.md prompt/system.md
     rsync -a --delete --exclude='.*' agent_workspace/learning/round_NNN/wiki_before/ wiki/
     ```
     `wiki_before/`에는 숨김 파일이 없으므로 `wiki/`를 통째로 바꾸지 않는다(`.readonly`가 사라지면 학습자가 `상품/*`도 고친다). 위 rsync는 숨김 파일(`.readonly`, `.index`)을 남기고, 학습자가 새로 만든 문서는 지운다. 읽기 전용 문서(`상품/*`, `응대/거절대응.md`)는 학습자가 바꿀 수 없어 `wiki_before/`와 같으므로 그대로 남는다. 인덱스는 다음 통화 때 자동으로 다시 맞춰진다.
6. 기록: `tmp/loop_log.md`에 회차 번호, 지표 표, 관찰 2~3줄, 변경 내용과 가설, 롤백 여부.
7. 다음 배치를 Claude가 바로 실행한다(위 명령, 백그라운드). 그 전에 이전 배치의 `error` 통화를 1단계대로 처리(재실행)하고, `uv run learner.py`로 대기열을 확인한다. 0건이 아니면(리뷰 실패로 20건이 안 찬 경우) `uv run learner.py --now`를 백그라운드로 돌려 학습을 마친 뒤 다음 배치를 시작한다. 그렇지 않으면 학습이 다음 배치 중간에 실행되어 회차 비교가 깨진다.

## 4. learner.py 개선 후보 (필요할 때 하나씩)
- A. 신고·미달성 통화 근거 강화: 리뷰 메모뿐 아니라 마지막 몇 턴 발췌를 LEARN_REQUEST에 넣는다(신고 직전 무엇이 거슬렸는지).
- B. 자동 게이트: 다음 learn 때 직전 round 이후 통화 지표를 그 전 round와 비교, 악화면 `*_before`로 복원 후 학습. 결과를 learner.json에 기록.
- C. 학습 이력: 이전 2~3회차 변경 요약·지표를 LEARN_REQUEST에 넣어 같은 수정을 되풀이하거나 진동하지 않게.
- D. 위키 성장: 달성 통화에서 통한 말(반론 처리, 혜택 설명)을 `응대/` 아래 문서로 쌓도록 지시. 읽기 전용 문서와 날조 금지는 유지.

## 5. 종료 조건
- 3회차 연속 루프성 종료 0, 정독에서 자연스러움 문제 없음, spam_rate ≤ 10%이고 achieved_rate가 더 오르지 않으면, 시드 1의 수신자 20명에 과적합하지 않았는지 다른 시드로 한 배치를 더 돌려(`--seed 2 --learn-every 21`: 리뷰는 하고 학습은 하지 않는다. `--learn-every 0`은 리뷰도 끈다) 같은 기준을 만족하는지 확인한 뒤 멈추고 사용자에게 보고. 남은 대기열(`pending.jsonl`)은 지운다.
- 같은 문제가 3회차 연속 해결되지 않으면 멈추고 사용자에게 방향을 묻는다.
