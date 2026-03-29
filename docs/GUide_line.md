# Research Guideline

## 1. Overview
- 커뮤니티는 보통 **A**를 믿지만, 실제로는 **조건 D**에서 이 전제가 깨진다.
- 원인은 **B1/B2/B3**로 분해할 수 있고, 이를 **M**으로 측정한다.
- 이후 **I**(targeted intervention)와 **J**(negative control)로 가설을 검증하고, 최종적으로 **R**(설계 원칙)을 제시한다.

예시 문장:
- 기존 방법은 similarity score가 evidence를 잘 반영한다고 가정하지만, rare localized evidence regime에서는 이 가정이 깨지며, 원인은 margin squashing + semantic misalignment이다. 이를 diagnostic metric으로 측정하고 targeted readout intervention으로 검증해 설계 원칙을 제시한다.

## 2. 암묵적 전제 A
- “이 분야가 기본값으로 믿는 가정”을 명시한다.
- 질문:
  - baseline들이 공통으로 깔고 있는 전제는 무엇인가?
  - reviewer가 “당연히 그렇다”고 생각하는 것은 무엇인가?
- 좋은 예:
  - softmax confidence가 evidence quality를 반영한다.
  - stronger generator context는 항상 solver를 개선한다.
- 나쁜 예:
  - 우리 문제는 중요하다.
  - 기존 방법은 부족하다.

## 3. 파손 조건 D
- A가 “언제, 어떤 조건에서” 깨지는지 명시한다.
- 형식:
  - A는 **C에서는 맞지만**, **D에서는 구조적으로 깨진다**.
- 질문:
  - 어떤 regime에서만 문제가 터지는가?
  - 반대로 언제는 A가 잘 맞는가?

## 4. 핵심 메커니즘 B1/B2/B3
- 병목을 단일 원인으로 끝내지 말고 계층적으로 분해한다.
- 형식:
  - **B1**: 직접 원인
  - **B2**: B1이 유발하는 중간 현상
  - **B3**: 관측 가능한 실패 패턴

## 5. 측정 도구 M
- “분석했다”가 아니라 **무엇을, 어떻게 측정했는지**를 쓴다.
- 좋은 M 조건:
  - headline metric과 다르다.
  - failure-sensitive하다.
  - intervention 없이도 standalone 분석 가치가 있다.

핵심:
- 방법이 실패해도 metric 자체는 남아야 한다.

## 6. 반증 가능한 예측 P1/P2/P3
- 가설을 검증 가능한 형태로 사전 명시한다.
- 형식:
  - **P1**: 가설이 맞다면 먼저 무너져야 하는 현상
  - **P2**: targeted intervention만 복구시켜야 하는 현상
  - **P3**: unrelated intervention은 같은 recovery를 만들지 못해야 하는 현상

## 7. Targeted Intervention I
- I는 “성능 개선 모듈”이 아니라 “B1 검증 도구”로 설계한다.
- 문장 템플릿:
  - We design I to selectively target B1 while minimally affecting unrelated components.

## 8. Negative Control J
- J는 I와 비슷해 보여도 B1을 직접 건드리지 않아야 한다.
- 기대 결과:
  - headline metric이 잠깐 오를 수 있어도, M/failure-sensitive metric은 회복시키지 못해야 한다.

## 9. 최종 설계 원칙 R
- 마지막은 방법이 아니라 **rule**로 끝낸다.
- 형식:
  - 따라서 이런 조건의 문제에서는 설계 시 **R**을 따라야 한다.

## 10. Execution Loop (실행 프로토콜)
- 앞으로 모든 실험은 아래 순서로 진행한다.
1. A/D/B1/B2/B3를 먼저 문장으로 고정한다.
2. M을 먼저 측정하고, P1/P2/P3를 사전 등록한다.
3. I와 J를 동일 budget에서 비교한다.
4. 결과는 `gain/harm`, failure-sensitive metric, control gap을 함께 보고 판단한다.
5. 마지막에 R을 업데이트한다.

## 11. Experiment Card Template
- 실험마다 아래 카드를 남긴다.
- `Date`:
- `Dataset/Split`:
- `A`:
- `D`:
- `B1/B2/B3`:
- `M (primary)`:
- `P1/P2/P3`:
- `I`:
- `J`:
- `Result (headline + M + gain/harm)`:
- `Decision (keep/drop/revise)`:

## 12. Current Working Mapping (2026-03-10)
- 현재 프로젝트의 임시 매핑:
  - `A`: V-PMI 기반 스칼라 요약이 hallucination/correct를 안정적으로 분리한다.
  - `D`: 질문 타입 혼합 + 생성 길이 변동이 큰 regime.
  - `B1`: 길이/질문 타입 confound로 분포 스케일이 흔들림.
  - `B2`: 토큰 궤적의 초반 spike와 후반 collapse가 평균 요약에서 상쇄됨.
  - `B3`: 전체 집합에서는 KS/AUC가 낮고, 특정 failure subtype에서만 신호가 강화됨.
  - `M`: token-wise V-PMI trajectory, collapse gap, rank dynamics, subgroup failure lift.

운영 원칙:
- 전체 failure를 한 묶음으로 보지 않고 subtype 분석을 우선한다.
- 약한 신호(KS≈0.1)는 바로 정책화하지 않고, control 실험으로만 해석한다.

## 13. Registered Hypotheses (H1/H2/H3)
- 아래 3가설은 사전 등록(Pre-registered) 상태로 취급한다.

### H1: 형태 가설 (Morphology over Scalar)
- 정의: 정오답 분리는 절대값 평균보다 토큰 궤적의 형태(초반 spike + 후반 collapse)로 설명된다.
- 핵심 피처군:
  - 형태: slope / inversion ratio / rank drop / post-peak drop
  - 대조군(negative control): 단순 평균형 스칼라(`vpmi_logit_mean` 등)
- 판정:
  - 형태 피처의 KS/AUC가 대조군보다 안정적으로 높아야 한다.

### H2: 길이 불변 가설 (Length-Invariant Robustness)
- 정의: 길이 정규화/비모수 피처(순위, 부호 비율, 낙폭)는 길이 분포 변화에서도 분리력 저하가 작다.
- 검증 분할:
  - 1000 subset (dev-like) + 12k holdout
- 판정:
  - 두 split에서 지표 방향(개선/악화)이 동일해야 한다.
  - 길이 상관(`pearson_with_len`) 절대값이 낮고, 분리력이 유지되어야 한다.

### H3: 개입 가설 (Conditional Intervention)
- 정의: 무조건 개입(always-on)보다 조건부 개입(gated)이 `gain-harm`를 안정적으로 양수로 유지한다.
- 비교축:
  - Always-on intervention vs Conditioned intervention vs No-intervention
- 판정:
  - accuracy 단독이 아니라 `gain`, `harm`, `switch_precision`, `gain-harm`로 최종 판단.

## 14. Acceptance Criteria (고정 판정 기준)
- 모든 신규 주장/모듈은 아래 기준을 동시에 통과해야 “채택”한다.
1. Direction Consistency:
   - 1000 subset과 12k holdout에서 같은 방향(개선 또는 악화)이어야 한다.
2. Statistical Confidence:
   - KS bootstrap CI: 하한(`ks_ci_lo`) > 0.
   - AUC bootstrap CI: 0.5를 포함하지 않아야 한다(`auc_ci_lo > 0.5` 또는 `auc_ci_hi < 0.5`).
3. Operational Utility:
   - `gain-harm > 0`를 기본 조건으로 한다.
   - `switch_precision`이 baseline random switch보다 유의하게 높아야 한다.
4. Reproducibility:
   - seed 변경(최소 3개)에서도 결론 부호가 유지되어야 한다.

## 15. Decision Matrix (Keep / Revise / Drop)
- Keep:
  - H1/H2/H3 기준을 모두 충족.
- Revise:
  - 통계는 유의하지만(`AUC CI` 비중첩), 운영 지표(`gain-harm`)가 불안정.
- Drop:
  - holdout에서 방향 역전, 또는 `AUC CI`가 0.5를 반복 포함.
