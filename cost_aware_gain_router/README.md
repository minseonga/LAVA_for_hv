# Cost-Aware Gain Router

## 한 줄 요약

이 방법론은 `baseline`과 `method`를 둘 다 끝까지 실행한 뒤 더 좋은 쪽을 고르는 controller가 아니다.  
대신 cheap probe만 먼저 보고, **"이 샘플에서 method를 쓰는 것이 baseline 대비 실제로 이득인가?"** 를 예측한 뒤, **오직 한 branch만 실행**하는 budget-aware router이다.

현재 이 폴더의 결과는 **VGA on POPE-9000** 기준으로 정리되어 있다.  
하지만 문제 설정 자체는 VGA뿐 아니라 **VISTA, EAZY 같은 training-free inference method 전반**에 그대로 적용할 수 있다.

서버에서 바로 artifact를 만들고 actual online inference까지 돌리려면 [SERVER_ONLINE_EXPERIMENT.md](./SERVER_ONLINE_EXPERIMENT.md)를 먼저 보면 된다.

---

## 1. 왜 이 방법이 필요한가

우리가 풀고 싶은 문제는 단순하다.

- `baseline`은 비교적 안전하다.
- `method`는 어떤 샘플에서는 성능을 올려 준다.
- 하지만 다른 샘플에서는 오히려 **harm**를 낸다.

즉, 모든 샘플에 무조건 `method`를 적용하는 것은 비효율적이고 위험하다.

원래는 다음과 같은 방식으로 문제를 풀려고 했다.

1. cheap probe에서 `FRG`를 계산한다.
2. 고정된 cutoff `tau`를 둔다.
3. `FRG >= tau`면 baseline, 아니면 method로 보낸다.

이 접근은 offline에서는 꽤 잘 되었지만, online runtime으로 옮기면 성능이 무너졌다.  
핵심 원인은 `tau` 숫자 자체가 아니라 **ordering mismatch**였다.

즉,

- offline에서 `tau` 근처에 있던 샘플들이
- online에서는 같은 순서로 정렬되지 않았고
- 그래서 단일 threshold 규칙이 더 이상 안정적으로 작동하지 않았다.

이 때문에 문제를 다시 정의할 필요가 생겼다.

---

## 2. 핵심 아이디어: route imitation이 아니라 utility prediction

이 방법론의 핵심 전환은 다음과 같다.

- 기존 접근: `offline controller가 어떤 route를 탔는지`를 복원하려고 함
- 새로운 접근: `이 샘플에서 method를 쓰는 것이 baseline 대비 실제로 이득인가`를 직접 예측함

즉, 목표를 "route label 복원"에서 "branch utility 예측"으로 바꾼다.

이 차이는 매우 중요하다.

`route label`은 중간 의사결정이지만, 우리가 진짜 원하는 것은 최종 answer accuracy다.  
어떤 route가 선택되었는지를 모사하는 것보다, **어느 branch가 더 맞을 가능성이 높은지**를 직접 예측하는 편이 배포 목적에 더 가깝다.

---

## 3. 문제 설정

각 샘플 `i`에 대해 두 개의 expensive branch가 있다고 하자.

- `baseline`
- `method` (예: VGA)

각 branch의 예측은 다음과 같다.

- `pred_baseline`
- `pred_method`
- `gt` = 정답

우리가 원하는 것은 모든 샘플에 대해 두 branch를 모두 실행하지 않고, cheap probe만 보고 하나를 선택하는 것이다.

최종 예측은 다음과 같다.

```text
if route = method:
    final_pred = pred_method
else:
    final_pred = pred_baseline
```

그리고 전체적으로 `method`를 보내는 비율은 예산 `beta`를 넘지 않게 한다.

즉, 문제는 다음과 같이 볼 수 있다.

```text
maximize final accuracy
subject to method_rate <= beta
```

---

## 4. 감독 신호: utility label

이 방법의 감독 신호는 route label이 아니다.  
각 샘플에 대해 `method`가 `baseline`보다 실제로 도움이 되었는지를 직접 라벨로 만든다.

정의는 다음과 같다.

```text
utility = +1  if method correct and baseline wrong
utility = -1  if baseline correct and method wrong
utility =  0  otherwise
```

해석:

- `+1`: method를 보내야 실제로 이득인 샘플
- `-1`: method를 보내면 오히려 손해인 샘플
- `0`: 어느 쪽을 보내도 성능 차이가 없는 샘플

이 정의가 중요한 이유는, 네가 원래 타겟으로 삼은 목표와 정확히 일치하기 때문이다.

즉 이 방법은:

- method가 harm를 낼 수 있는 샘플에서는 baseline으로 남기고
- method가 진짜 도움이 되는 샘플에서만 method를 쓰도록

직접 학습한다.

---

## 5. FRG를 쓰는가?

쓴다. 많이 쓴다.  
하지만 예전처럼 **FRG 하나 + tau 하나**로 최종 route를 정하지는 않는다.

현재 router 입력 feature는 5개다.

1. `frg_off`
   offline probe에서 계산된 FRG

2. `g_top5_mass`
   상위 토큰 질량 기반 confidence 계열 feature

3. `probe_anchor_yes`
   anchor 유무를 이진화한 feature

4. `abs_frg_to_tau`
   기존 offline tau와의 거리

5. `frg_x_mass`
   `frg_off * g_top5_mass`

즉 FRG 관련 신호가 중심이다.

- `frg_off`
- `abs_frg_to_tau`
- `frg_x_mass`

이 세 개는 모두 FRG를 직접 또는 간접으로 사용한다.

따라서 이 방법은 **FRG를 버린 것**이 아니라,

> "FRG를 핵심 신호로 유지하되, threshold 하나로는 해결되지 않는 ordering mismatch를 다른 cheap probe feature와 작은 nonlinearity로 보정하는 방법"

이라고 이해하면 된다.

---

## 6. 왜 FRG 단일 threshold 대신 learned router가 필요한가

실제 비교 결과는 이 점을 잘 보여준다.

기준 정책:

- Baseline only: `0.8522`
- VGA only: `0.8661`
- Offline controller: `0.8764`
- Strict gain oracle: `0.9046`

cheap probe 기반 budgeted router:

- `frg_rank @ 30% budget`: `0.8628`
- `tree_utility @ 30% budget`: `0.8662`
- `hgb_utility @ 30% budget`: `0.8726`

이 결과는 두 가지를 말한다.

1. FRG만으로 rank를 매기는 것보다
2. FRG + 보조 feature + learned utility score가
3. 최종 accuracy를 더 잘 최적화한다.

즉, **FRG는 중요하지만 충분조건은 아니다.**

---

## 7. 실제 학습 데이터는 어떻게 만들었는가

현재 결과는 두 종류의 artifact를 `id` 기준으로 조인해서 만들었다.

1. cheap probe / route 관련 로그
2. per-sample 정답 및 branch 예측 reference

이 폴더 기준으로 핵심 입력은:

- `oof_scores.csv`
- `reference_policies.csv`
- `budget_sweep.csv`

실험 생성에 사용된 원본 경로 정보는 [run_info.json](./run_info.json)에 기록되어 있다.

현재 설정:

- 데이터셋: `POPE-9000`
- branch: `baseline` vs `VGA`
- 샘플 수: `9000`
- utility 분포:
  - gain: `471`
  - harm: `346`
  - neutral: `8183`

이 분포는 현실적인 이유를 설명해 준다.

- 정말 method가 꼭 필요한 샘플은 일부다.
- harm 샘플도 적지 않다.
- 대부분 샘플은 neutral이다.

따라서 무조건 method를 쓰는 정책은 낭비가 크고,  
utility-aware router가 실용적이다.

---

## 8. 어떤 모델을 비교했는가

현재 비교한 router는 세 가지다.

### 8.1 `frg_rank`

가장 단순한 baseline이다.

- 사실상 `frg_off` 기반 단일 ranking
- threshold/순위 기반 rule의 약한 버전

장점:

- 구현이 쉽다
- 해석이 단순하다

단점:

- ordering mismatch나 feature interaction을 반영하지 못한다

### 8.2 `tree_utility`

작은 depth-3 decision tree regressor다.

장점:

- 해석 가능
- 사람이 규칙을 읽을 수 있음

단점:

- 표현력이 제한됨
- 중간 구간의 미세한 ranking이 약함

규칙은 [tree_utility_rules.txt](./tree_utility_rules.txt) 에 정리돼 있다.

### 8.3 `hgb_utility`

작은 HistGradientBoostingRegressor다.

장점:

- feature 수는 적지만 비선형 관계를 잘 잡음
- budget sweep에서 가장 좋은 성능

현재 최종 deploy 후보는 이 모델이다.

---

## 9. 학습/평가 프로토콜

평가는 5-fold stratified OOF로 했다.

중요한 점:

- train split으로 router를 학습
- held-out split에 대해 utility score 예측
- 이 out-of-fold score를 모아서 전체 budget sweep 계산

즉, `budget_sweep.csv`의 결과는 단순 train-set 재적합 점수가 아니라,  
조금 더 일반화에 가까운 추정치다.

stratification 기준은 `utility in {-1, 0, +1}`이다.  
이렇게 해야 gain / harm / neutral 비율이 fold마다 너무 무너지지 않는다.

OOF score는 [oof_scores.csv](./oof_scores.csv)에 저장되어 있다.

---

## 10. budgeted routing은 어떻게 동작하는가

budget `beta`가 주어지면,

1. 각 샘플에 대해 utility score를 계산한다.
2. 점수가 높은 순으로 정렬한다.
3. 상위 `k = round(beta * N)`개만 method로 보낸다.
4. 나머지는 baseline으로 보낸다.

즉, 평가용 policy는 본질적으로 **top-k selection**이다.

예를 들어 `beta = 0.30`이면:

- 전체 9000개 중 상위 약 2700개만 method
- 나머지는 baseline

이 방식은 단순하다.

- method 사용량을 정확히 제어할 수 있고
- accuracy-efficiency trade-off curve를 명확하게 그릴 수 있다.

---

## 11. 왜 30% budget을 선택했는가

[budget_sweep.csv](./budget_sweep.csv)를 보면 HGB router는 다음과 같다.

- 20% budget: `0.8691`
- 30% budget: `0.8726`
- 40% budget: `0.8738`
- 50% budget: `0.8751`

30%를 선택한 이유:

- `20% -> 30%`는 accuracy 상승폭이 꽤 의미 있다.
- 하지만 `30% -> 40% -> 50%`로 갈수록 증가폭은 줄어든다.
- 반면 method 사용량은 계속 증가한다.

즉, 30%는 accuracy와 cost 사이의 실용적인 knee point다.

또한 기존 offline controller는 method rate가 약 `59.4%`였다.  
30% budget은 method 사용량을 거의 절반으로 줄이면서도 accuracy를 꽤 유지한다.

---

## 12. 현재 결과를 어떻게 해석해야 하나

핵심 성능 비교:

| Policy | Method Rate | Accuracy |
|---|---:|---:|
| Baseline only | 0.0000 | 0.8522 |
| VGA only | 1.0000 | 0.8661 |
| Offline controller | 0.5942 | 0.8764 |
| HGB utility @ 30% | 0.3000 | 0.8726 |

이 결과의 의미는 분명하다.

- baseline only보다 확실히 좋다.
- VGA only보다도 좋다.
- method 호출 비율은 매우 크게 줄었다.
- offline controller에는 조금 못 미치지만, cost를 많이 줄인 상태에서 꽤 근접한다.

즉 이 router는

- "offline controller를 정확히 모사하는 방법"은 아니지만
- "single-branch deploy under budget"이라는 실제 목적에는 더 맞는 방법이다.

---

## 13. 실제 배포에서는 어떻게 쓰는가

배포 시에는 baseline과 method를 둘 다 돌리지 않는다.

runtime flow:

1. cheap probe 실행
2. probe feature 5개 생성
3. trained utility router가 점수 계산
4. 점수 기준으로 branch 선택
5. 오직 선택된 branch만 실행

즉, 실제 runtime cost는 대략 다음과 같이 생각할 수 있다.

```text
cost ~= probe_cost + router_cost + one_branch_cost
```

둘 다 실행하는 controller와 달리:

```text
dual_run_cost ~= probe_cost + baseline_cost + method_cost
```

가 아니다.

이 차이가 deploy 관점에서 매우 중요하다.

---

## 14. 이 방법은 "training-free"인가?

엄밀히 말하면:

- `VGA / VISTA / EAZY` 자체는 training-free inference method일 수 있다.
- 하지만 이 router는 **supervised trained selector**다.

즉 이 방법은:

- `end-to-end training-free`는 아니다.
- 대신 `training-free method를 더 안전하게 배포하기 위한 lightweight trained router`다.

이 framing은 오히려 솔직하고 좋다.

논문에서 이렇게 쓰는 것이 맞다.

> "We do not train the underlying method. We train a lightweight router that decides when a training-free method should be used."

---

## 15. VGA/VISTA/EAZY로 일반화 가능한가

문제 설정상으로는 충분히 가능하다.

왜냐면 이 방법은 특정 method의 내부 로직보다 다음 사실을 이용하기 때문이다.

- baseline과 method가 샘플별로 다르게 행동한다
- method가 일부 샘플에서 gain, 일부 샘플에서 harm를 낸다
- cheap probe에서 얻은 신호가 이 utility와 상관이 있다

이 조건이 맞으면, VGA/VISTA/EAZY 모두 같은 framing에 넣을 수 있다.

즉 일반화 버전은 이렇게 쓸 수 있다.

```text
baseline vs method_M
cheap probe -> utility score for method_M
budgeted single-branch routing
```

method `M`만 바꾸면 된다.

다만 현재 이 폴더의 결과는 **VGA on POPE-9000** 기준이므로,  
VISTA/EAZY까지 주장하려면 각각에 대해 같은 파이프라인을 다시 돌려야 한다.

---

## 16. 이 접근의 장점

### 16.1 실제 배포 목적과 잘 맞는다

목표가 "method를 무조건 쓰지 말고, 필요한 샘플에서만 써라"라면 이 framing이 정답에 가깝다.

### 16.2 ordering mismatch 문제를 우회한다

FRG 단일 tau rule은 offline/online ordering mismatch에 취약했다.  
여기서는 FRG를 쓰되, 그 하나에 전부 걸지 않는다.

### 16.3 cost-accuracy trade-off를 명시적으로 제어한다

budget `beta`가 명확한 knob 역할을 한다.

### 16.4 single-run deploy가 가능하다

실제 inference에서는 한 branch만 돈다.

---

## 17. 현재 한계

이 폴더의 결과를 해석할 때 주의할 점도 있다.

### 17.1 아직 VGA 중심 결과다

현재 수치는 VGA on POPE-9000 중심이다.  
VISTA/EAZY로 바로 일반화했다고 말할 수는 없다.

### 17.2 offline supervision이 필요하다

utility label을 만들려면 학습 시점에는 baseline과 method를 둘 다 알아야 한다.

### 17.3 distribution shift는 여전히 문제일 수 있다

새 데이터 분포에서는 score threshold나 budget 최적점이 바뀔 수 있다.

### 17.4 cheap probe feature의 runtime reproducibility는 명확히 써야 한다

논문에서는 어떤 feature가 offline-only artifact가 아니라 실제 runtime probe에서 계산 가능한지 분명히 적어야 한다.

---

## 18. 이 폴더의 파일을 어떻게 읽으면 되는가

처음 보는 사람이면 아래 순서가 가장 좋다.

1. [summary.md](./summary.md)
   빠른 개요

2. 이 파일 `README.md`
   방법론 전체 설명

3. [deployment_rule_30_budget.md](./deployment_rule_30_budget.md)
   현재 30% budget deploy 정책 상세

4. [paper_methodology.md](./paper_methodology.md)
   논문 서술용 포맷

5. [reference_policies.csv](./reference_policies.csv)
   기준 정책 성능

6. [budget_sweep.csv](./budget_sweep.csv)
   budget별 전체 성능 곡선

7. [deployment_score_bands_30_budget.csv](./deployment_score_bands_30_budget.csv)
   30% cutoff 주변 score band 해석

8. [deployment_tree_leaf_table.csv](./deployment_tree_leaf_table.csv)
   해석용 tree leaf 요약

9. [oof_scores.csv](./oof_scores.csv)
   샘플별 OOF score

10. [run_info.json](./run_info.json)
    원본 실험 설정 정보

---

## 19. 이 방법론을 한 문장으로 다시 요약하면

이 접근의 본질은 다음과 같다.

> `offline FRG + fixed tau` 기반 controller 대신, cheap probe feature를 이용해 method의 sample-wise utility를 예측하고, budget 제약 하에서 method를 선택적으로 호출하는 single-branch router를 학습한다.

---

## 20. 지금 네 연구 목표와의 연결

네가 진짜 하고 싶은 것은 이것이다.

- VGA/VISTA/EAZY 같은 training-free method는 유용하지만
- 모든 샘플에 무조건 적용하면 harm가 생긴다
- baseline이 이미 잘하는 샘플은 baseline으로 두고
- method가 진짜 이득인 샘플만 method로 보내고 싶다

이 폴더의 방법론은 정확히 그 목표를 겨냥하고 있다.

즉, 현재 방향은:

- 단순 threshold controller보다 낫고
- deploy 목적과 맞고
- 논문화도 충분히 가능한 방향이다.

다만 top-tier main 수준으로 가려면:

- VGA뿐 아니라 VISTA/EAZY까지 확장
- 추가 benchmark
- 실제 online deploy 검증

이 붙어야 한다.

현재 이 폴더는 그 중 **핵심 방법론과 첫 번째 강한 실험 근거**를 담고 있다고 보면 된다.
