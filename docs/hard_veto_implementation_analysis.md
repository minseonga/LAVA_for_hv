# Hard Veto Implementation Analysis

## TL;DR

현재 실제 메인 결과로 연결된 `hard veto`는 **FRG/GMI 해석의 `C/E` feature를 쓰는 오프라인 sample-level controller**다.

- 입력: `C = faithful_minus_global_attn`, `E = guidance_mismatch_score`
- 결정 규칙: `veto = (C >= tau_c) OR (E >= tau_e)`
- veto이면 `baseline`을 쓰고, 아니면 `VGA/VISTA` 출력을 유지한다.

즉 현재 hard-veto 스토리의 중심은:

- `FRG` 축에 해당하는 `C`
- `GMI` 축에 해당하는 `E`
- 이 둘로 baseline/guidance route를 고르는 offline veto

다.

중요한 점은, 이 controller가 기본적으로 쓰는 `C/E`는 지금 기준으로 **token-level FRG/GMI 최종형이 아니라 sample-level proxy**라는 것이다.

반면 레포 안에는 별도로:

- `RFHAR`: token-level `C/A/D/B`를 써서 late self-attention logits를 직접 수정하는 online intervention
- `FRGG`: token-level `A/C/E`를 써서 faithful head에만 late-layer guidance bias를 주는 online intervention

도 같이 들어 있다.

다만 이 online 계열은 내가 확인한 실험 산출물 기준으로는:

- 성능 개선이 거의 없거나
- 일부 약한 개선만 있고
- 메인 hard-veto 성과 경로로 채택되지는 않은

**실험 분기 / 사실상 유기된 가지**에 더 가깝다.

즉, 현재 코드 기준으로는:

1. `FRG/GMI 기반 hard veto controller`
2. `RFHAR/FRGG online intervention branch`

을 같은 것으로 보면 안 된다.


## 1. 관련 파일과 역할

### 먼저 결론

연구 스토리 기준으로 우선순위는 아래처럼 보는 게 가장 자연스럽다.

1. 메인: `C/E = FRG/GMI 해석 feature` 기반 offline hard veto
2. 부차: `RFHAR/FRGG` online intervention 실험

즉 구현이 존재한다는 사실과, 실제로 메인 결과를 만든 경로라는 사실은 분리해야 한다.

### A. 오프라인 hard veto controller

- `scripts/run_vga_hard_veto_controller.py`
  - 실제 veto rule 적용
  - threshold calibration
  - baseline vs VGA/VISTA decision switching
- `scripts/run_vga_hard_veto_controller_9000.sh`
  - VGA용 wrapper
- `scripts/run_vista_hard_veto_controller_9000.sh`
  - VISTA용 wrapper

### B. hard veto가 쓰는 sample-level feature table

- `scripts/build_pope_feature_screen_v1.py`
  - `features_unified_table.csv` 생성
  - 여기서 현재 hard veto가 쓰는 `faithful_minus_global_attn`, `guidance_mismatch_score`가 나온다
- `scripts/extract_vga_vsc_features.py`
  - grounding/VSC 계열 `A/G` feature 추출

### C. online late-layer intervention

- `llava/model/ais_gating.py`
  - generation 중 late-layer attention interception
  - `RFHAR`, `FRGG`, `FRRS`를 실제 model forward에 연결
- `llava/model/rfhar.py`
  - token-level `C/A/D/B` 기반 logits intervention
- `llava/model/frgg.py`
  - token-level `A/C/E` 기반 faithful-head guidance gating
- `llava/eval/model_vqa_loader.py`
  - feature json/jsonl 로딩, generate kwargs 주입
- `llava/model/language_model/llava_llama.py`
  - generation 시작/종료 시 runtime hook 연결

### D. token-level RFHAR feature builder

- `scripts/build_rfhar_feats_from_role_and_trace.py`
  - tokenwise `C/A/D/B` 생성


## 2. 현재 hard veto controller는 정확히 무엇을 하나

`scripts/run_vga_hard_veto_controller.py`는 baseline 출력과 VGA/VISTA 출력이 둘 다 있을 때, 샘플마다 어느 쪽을 채택할지 결정하는 **training-free decision controller**다.

현재 hard-veto 연구 스토리에서 실제 controller라고 부를 수 있는 것은 이쪽이다.

흐름은 아래와 같다.

1. `per_case_csv`를 읽는다.
   - 보통 taxonomy 결과물
   - `gt`, `pred_baseline`, `pred_vga`, `case_type`를 포함해야 한다
2. `features_csv`를 읽는다.
   - 기본값은 `features_unified_table.csv`
   - 여기서 `C`와 `E`를 가져온다
3. 두 테이블을 `id`로 merge한다.
4. calibration split에서 `tau_c`, `tau_e`를 고른다.
5. 전체 샘플에 veto rule을 적용한다.
6. veto면 baseline, 아니면 VGA/VISTA를 채택한다.

### 현재 decision rule

현재 구현은 아주 단순하다.

```text
veto = (C >= tau_c) OR (E >= tau_e)
pred_controller = baseline if veto else target_method
```

즉:

- `C`가 높아도 veto
- `E`가 높아도 veto
- 둘 다 낮을 때만 guidance 쪽 출력을 유지

이 구조를 해석하면:

- `C`가 높다 = faithful routing이 충분하다고 봄
- `E`가 높다 = guidance mismatch 위험이 크다고 봄
- 따라서 둘 중 하나만 높아도 guidance를 끄고 baseline으로 되돌린다

### calibration objective

threshold는 고정값이 아니라 calibration split에서 고른다.

정의:

- `D1 = vga_improvement`
  - VGA가 baseline보다 좋아진 샘플
  - 여기서 veto는 손해
- `D2 = vga_regression`
  - VGA가 baseline보다 나빠진 샘플
  - 여기서 veto는 이득

탐색 objective:

```text
objective = D2_correct_veto - lambda_d1 * D1_wrong_veto
```

제약:

```text
D1_wrong_veto_rate <= max_d1_wrong_rate
```

즉 calibration은:

- D2를 많이 막고
- D1을 너무 많이 죽이지 않는

`tau_c`, `tau_e`를 quantile grid 위에서 찾는다.

### missing feature 처리

`C/E`가 없는 샘플은 `fallback_when_missing_feature`로 처리한다.

- `baseline`이면 missing도 veto
- `vga`면 missing은 non-veto

기본값은 `vga`다.

### 구현상 주의할 점

- `run_vista_hard_veto_controller_9000.sh`도 내부적으로 `run_vga_hard_veto_controller.py`를 그대로 쓴다
- 따라서 script 내부 변수명과 route label은 여전히 `vga` 기준이다
- 개념적으로는 generic controller지만, naming은 VGA 중심으로 남아 있다


## 2.1 현재 남아 있는 성능 흔적상 무엇이 메인인가

실험 산출물 기준으로 보면 메인은 offline hard veto다.

### VGA 9k hard veto

- baseline acc: `0.8522`
- VGA acc: `0.8661`
- hard-veto controller acc: `0.8750`

즉 VGA 위에 veto를 얹어 추가 개선이 있다.

### VISTA 9k hard veto

- baseline acc: `0.8522`
- VISTA acc: `0.8314`
- hard-veto controller acc: `0.8434`

즉 VISTA 자체보다는 회복되지만 baseline은 넘지 못했다.

### RFHAR / FRGG / FRRS

내가 확인한 산출물 기준으로는:

- `FRGG`: baseline 대비 `changed_pred = 0`, 사실상 no-op
- `RFHAR` 기본 설정: baseline 대비 `changed_pred = 0`, 사실상 no-op
- `RFHAR` 튜닝 설정: `acc +0.008` 정도의 약한 개선
- `FRRS`: baseline과 동일 metric

따라서 “실제로 채택된 hard-veto 방법”은 online branch가 아니라 offline controller로 보는 것이 맞다.


## 3. 현재 hard veto가 쓰는 C와 E는 어떻게 계산되나

핵심은 여기다.

현재 hard veto의 기본 입력은 `features_unified_table.csv`에서 온다.
즉 현재 controller는 **tokenwise map을 직접 쓰지 않고, 이미 sample-level로 축약된 feature**를 쓴다.


## 3.1 A/G: grounding 또는 visual confidence

`A` 계열은 `build_pope_feature_screen_v1.py`에서 두 경로 중 하나로 들어온다.

### 경로 1. VGA-style VSC 사용

`extract_vga_vsc_features.py`가 다음을 계산한다.

- object token grounding 또는 entropy fallback으로 patch-level `grounding`
- 정규화된 지도 `G`
- sample-level 요약:
  - `obj_token_prob_max`
  - `obj_token_prob_mean`
  - `obj_token_prob_lse`
  - `obj_token_prob_topkmean`
  - `G_entropy`
  - `G_top1_mass`
  - `G_top5_mass`
  - `G_effective_support_size`

여기서 `G_top5_mass`가 뒤에서 `E` proxy 계산에 직접 쓰인다.

### 경로 2. trace proxy fallback

VSC가 없으면 `per_layer_trace_csv`에서:

- `yes_sim_objpatch_max`
- `yes_sim_objpatch_topk`
- `yes_sim_local_max`
- `yes_z_local_max`

등으로 proxy를 만든다.


## 3.2 C: faithful routing proxy

현재 hard veto가 쓰는 기본 `C`는:

```text
faithful_minus_global_attn
```

이며 `build_pope_feature_screen_v1.py`의 `routing_features(...)`에서 계산된다.

입력은:

- `per_head_trace_csv`
- faithful headset json
- late window (`late_start..late_end`, 기본 16..24)

수식 수준으로 쓰면:

- late window 안에서 faithful headset에 속하는 `(layer, head)`들의 `head_attn_vis_ratio`를 모은다
- 그 평균을 `m`
- late window 전체 head들의 `head_attn_vis_ratio` 평균을 `g`
- 최종 C proxy는:

```text
C_sample = m - g
```

함께 저장되는 값:

- `faithful_head_attn_mean = m`
- `faithful_head_attn_topkmean`
- `faithful_head_coverage`
- `faithful_minus_global_attn = m - g`
- `faithful_n_points`

### 중요한 해석

이건 지금 사용자가 정리한 “projected visual token마다 faithful-over-global score를 만들고 TopKMean 하는 FRF/FRG”와는 다르다.

현재 구현은:

- unit이 visual token이 아니라
- late window 안의 `(layer, head)` trace point들을 먼저 모아서
- 바로 sample-level 평균 차이로 축약한다

즉 hard veto controller가 쓰는 현재 `C`는:

- **token-level FRF/FRG 원형이 아니라**
- **late faithful headset consumption의 sample-level proxy**

라고 보는 게 정확하다.


## 3.3 D: harmful routing proxy

`D`도 같은 방식이다.

- harmful headset에 속하는 `(layer, head)`들의 `head_attn_vis_ratio` 평균
- late 전체 평균과의 차이

저장되는 주요 컬럼:

- `harmful_head_attn_mean`
- `harmful_head_attn_topkmean`
- `harmful_head_coverage`
- `harmful_minus_global_attn`
- `harmful_minus_faithful`


## 3.4 E: guidance mismatch / GMI proxy

현재 hard veto가 쓰는 `E`는:

```text
guidance_mismatch_score
```

이며 `build_pope_feature_screen_v1.py`에서 **A/C/D를 조합한 sample-level proxy**로 계산된다.

코드상 계산은:

```text
g5 = G_top5_mass
fmean = faithful_head_attn_mean
hmean = harmful_head_attn_mean

faithful_on_G     = fmean * g5
faithful_on_nonG  = fmean * (1 - g5)
harmful_on_G      = hmean * g5
harmful_on_nonG   = hmean * (1 - g5)
```

그 다음:

```text
supportive_outside_G   = faithful_on_nonG / (fmean + eps)
harmful_inside_G       = harmful_on_G / (hmean + eps)
guidance_mismatch_score = harmful_on_G - faithful_on_G
context_need_score      = faithful_on_nonG - faithful_on_G
```

### 중요한 해석

이것도 현재 사용자가 정리한 “token마다 guided-but-harmful / needed-but-unguided mismatch를 계산하고 TopKMean 하는 GMI”와는 다르다.

현재 구현의 `E`는:

- tokenwise mismatch map이 아니라
- `G_top5_mass`와 sample-level `faithful/harmful` 평균을 조합한
- **sample-level mismatch proxy**

다.

즉 지금 hard veto controller의 `E`는 GMI의 최종 논문 정의라기보다:

- “guidance mismatch를 sample level에서 요약한 근사치”

로 보는 것이 정확하다.


## 4. 따라서 현재 hard veto는 무엇을 측정한다고 보는 게 맞나

현재 hard veto controller가 실제로 사용하는 축은 아래처럼 정리하는 게 가장 정확하다.

### 현재 C

`C = faithful_minus_global_attn`

- late self-attention trace 기반
- faithful headset 평균과 late global 평균의 차이
- sample-level proxy

이건 현재 이름상 `FRG` 해석에 더 가깝다.
즉 hard-veto 스토리에서 `C`는 사실상 FRG 축이라고 보는 것이 자연스럽다.

`FRF`로 쓰고 싶다면, 현재 수학은 그대로 두되 “faithful consumption quality proxy”라고 해석을 바꾸는 정도가 안전하다.

### 현재 E

`E = guidance_mismatch_score`

- grounding mass `G_top5_mass`
- faithful/harmful head attention 평균
- 이 셋을 조합한 sample-level mismatch proxy

즉 현재 코드의 `E`는 GMI의 token-level 정식 버전이라기보다 **GMI surrogate**다.
그래도 hard-veto 스토리 차원에서는 현재 `E`를 GMI 축으로 읽는 것이 가장 자연스럽다.


## 5. online intervention 쪽은 어떻게 다른가

여기서부터는 메인 hard-veto 경로가 아니라, generation 중 attention logits를 직접 건드려 보려던 **online 실험 분기**다.


## 5.1 RFHAR: token-level C/A/D/B intervention

`RFHAR`는 `llava/model/rfhar.py`에 구현되어 있고, `ais_gating.py`를 통해 generation late layer에 들어간다.

### RFHAR가 기대하는 입력

`rfhar_feats={"C":..., "A":..., "D":..., "B":...}`

shape는 모두 `[B, K_img]`다.
즉 여기는 실제로 **visual token 단위 feature**를 받는다.

### RFHAR feature builder

`scripts/build_rfhar_feats_from_role_and_trace.py`가 tokenwise `C/A/D/B`를 만든다.

#### C_map

`role_csv`에서 `supportive` label patch들에 대해 누적:

```text
rank_w = 1 / (1 + rank)^rank_decay
delta_abs = max(|delta_gt_margin|, |delta_yes_minus_no|)
score = rank_w * (sim + delta_scale * delta_abs)
```

supportive면 이 score가 `C_map[p]`에 더해진다.

#### D_map

같은 구조로 `harmful` 또는 `assertive` label patch에 score를 더한다.

#### A

두 source의 max를 취한다.

1. `trace_csv`
   - `a_layer`에서 `yes_sim_local_topk_weight_json`
2. `role_csv`
   - `max(sim, 0.5 * delta_abs)`

즉:

```text
A[p] = max(A_trace[p], A_role[p])
```

#### B

early/late attention top-k 등장 빈도 차이로 instability를 잡는다.

```text
e_freq = early_topk_freq[p]
l_freq = late_topk_freq[p]
B[p] = max(0, e_freq - l_freq)
```

그리고 같은 patch가 supportive와 harmful 양쪽에 모두 걸리면 `conflict_bonus`를 더한다.

### RFHAR runtime 동작

`RFHAR`는:

1. `C/A/D/B`를 z-score 기반으로 정규화
2. token utility `rf`를 계산

```text
C_tilde = ReLU(z(C))
A_tilde = sigmoid(z(A))
D_tilde = sigmoid(z(D))
B_tilde = sigmoid(z(B))

rf = C_tilde * A_tilde / (1 + lambda_penalty * (D_tilde + B_tilde))
```

3. 현재 last-query attention에서 image-only normalized attention을 구함
4. `rf`를 많이 읽는 head를 positive role
5. `1-rf`를 많이 읽는 head를 negative role
6. late layer image-token logits에 additive delta를 넣는다

즉 RFHAR는:

- hard veto처럼 model-level route switch를 하는 게 아니라
- late self-attention logits를 직접 수정하는 **soft online reweighting**

이다.

또한 RFHAR는 faithful headset을 직접 쓰지 않고, 현재 step attention과 `rf`를 조합해 **dynamic head role**을 만든다.


## 5.2 FRGG: token-level A/C/E intervention

`FRGG`는 `llava/model/frgg.py`에 구현되어 있고, 이것도 `ais_gating.py`를 통해 late self-attention에 들어간다.

### FRGG가 기대하는 입력

`frgg_feats={"A":..., "C":..., "E":...}`

shape는 `[B, K_img]`다.

즉 FRGG는 설계상:

- visual token 단위 `A/C/E`
- faithful head mask
- late self-attention image columns

을 모두 사용한다.

설계만 놓고 보면 사용자가 정리한 FRF/GMI 서술과 가장 가깝다.

추가로, 내가 이번에 확인한 범위 안에서는 `RFHAR`처럼 명시적인 `FRGG feature builder` 스크립트는 바로 보이지 않았다.
즉 현재 레포는:

- `FRGG` runtime 자체는 구현되어 있고
- `model_vqa_loader.py`도 `frgg_feats_json`을 로드할 수 있게 되어 있지만
- token-level `A/C/E`를 어떤 canonical script로 만드는지는 코드상 바로 드러나지 않는다

는 점을 같이 기억하는 게 좋다.

### FRGG 내부 수식

#### token prior

```text
S_i = ReLU(z(C_i)) * sigmoid(z(A_i))
P_i = S_i / sum_j S_j
```

즉 `A`와 `C`로 token prior를 만든다.

#### sample gate

먼저 `C`, `E` 각각에 대해 top-k mean을 만든다.

```text
C_bar = TopKMean(C)
E_bar = TopKMean(E)
```

그 다음:

```text
g_c = sigmoid(k_c * (tau_c - C_bar))
g_e = sigmoid(k_e * (tau_e - E_bar))
g   = g_c * g_e
```

즉:

- `C_bar`가 높아질수록 gate는 닫히고
- `E_bar`가 높아질수록 gate도 닫힌다

이 구조는 해석상 hard veto와 매우 비슷하지만, hard cutoff 대신 sigmoid gate를 쓴다.

#### late-layer intervention

최종 delta는:

```text
delta = gamma * g * faithful_head_mask * P
```

이고,

- last query row만
- image-token columns만
- faithful head rows만

수정한다.

### 중요한 해석

FRGG는 개념적으로 hard veto의 “if C high or E high then guidance off”를

- sample switch가 아니라
- faithful-head late attention bias

로 옮긴 soft version이라고 볼 수 있다.

다만 현재 남아 있는 결과물 기준으로는, 이 아이디어가 메인 성과 경로가 되지는 못했다.


## 6. hard veto controller와 FRGG/RFHAR의 관계

정리하면:

### hard veto controller

- decision level
- baseline vs VGA/VISTA를 샘플 단위로 스위치
- 기본 입력은 sample-level `C/E` proxy
- 연구 스토리상 메인 경로

### FRGG

- token level
- faithful heads만 late self-attention에서 bias
- 설계상 FRF/GMI 논문 서술과 가장 가까움
- 하지만 현재 산출물 기준으론 메인 성공 경로는 아님

### RFHAR

- token level
- `C/A/D/B` 기반 latent utility로 head-aware reweighting
- FRF/GMI보다는 broader한 late-layer routing intervention
- 역시 메인 hard-veto 결과로 이어진 branch는 아님


## 7. 지금 코드가 사용자가 정리한 논문 정의와 어디가 맞고 어디가 다른가

### 맞는 부분

- “late self-attention image columns”를 intervention 위치로 본다
- faithful head subset 개념이 존재한다
- FRGG는 `TopKMean(C)`, `TopKMean(E)` 식의 sample gate를 가진다
- RFHAR/FRGG/FRRS 모두 visual token ordering이 맞아야 동작한다

### 다른 부분

#### 1. hard veto controller의 C는 token-level FRF가 아니다

현재 기본 `C`는:

- projected visual token별 점수의 TopKMean이 아니라
- late faithful headset 평균 minus late 전체 평균

이다.

#### 2. hard veto controller의 E는 token-level GMI가 아니다

현재 기본 `E`는:

- token별 mismatch score의 TopKMean이 아니라
- `G_top5_mass`, `faithful_mean`, `harmful_mean`으로 만든 sample-level proxy

다.

#### 3. token alignment sanity check가 완전하진 않다

online module들은 `image_mask`의 image column 수와 feature length `K_img`가 같은지만 검사한다.
즉:

- 길이 불일치
- 범위 오류

는 어느 정도 잡지만,

- 서로 다른 permutation인데 길이만 같은 경우
- role/trace/G가 같은 patch ordering을 쓰는지

는 코드가 직접 검증하지 않는다.

#### 4. late window가 스크립트마다 다르다

현재 기본값 예시:

- feature screen C/D: `16..24`
- RFHAR late: `16..31`
- FRGG late: `16..30`
- FRRS late: `18..21`

즉 “late-consumption window”를 본문에 쓸 때는 어떤 파이프라인을 말하는지 구체화해야 한다.

#### 5. FRGG 재현 경로는 RFHAR보다 덜 명시적이다

`RFHAR`는 `build_rfhar_feats_from_role_and_trace.py`가 분명히 있지만, `FRGG`는 loader/runtime는 있으나 token-level `A/C/E` 생성 경로가 이번 inspection 범위에선 명시적으로 연결되지 않았다.
즉 논문/문서에서 “현재 구현되어 있다”와 “재현 경로가 정리되어 있다”는 분리해서 쓰는 게 좋다.


## 8. 지금 상태에서 가장 정직한 서술

현재 레포 기준으로 가장 안전한 설명은 아래다.

### hard veto controller에 대해서

현재 hard veto controller는 `faithful_minus_global_attn`과 `guidance_mismatch_score`라는 **sample-level routing proxies**를 사용하여 baseline과 VGA/VISTA 사이를 switching하는 offline decision rule이다.
그리고 실제 성능 향상 기록이 확인되는 메인 경로도 이쪽이다.

### FRF/FRG에 대해서

현재 default hard-veto pipeline의 `C`는 논문적으로는 “faithful-over-global late routing proxy”라고 부르는 것이 정확하다.
실제 hard-veto 연구 스토리에서는 이 축을 `FRG`로 읽는 것이 가장 자연스럽다.
FRF라고 부르려면 “faithful consumption fidelity proxy”라는 표현이 더 안전하다.

### GMI에 대해서

현재 default hard-veto pipeline의 `E`는 strict token-level GMI가 아니라, object-centric grounding mass와 faithful/harmful routing averages를 조합한 **sample-level mismatch surrogate**다.
즉 hard-veto 문맥에서의 GMI는 현재 구현상 sample-level surrogate 형태로 쓰이고 있다.

### online method에 대해서

FRGG와 RFHAR는 이보다 더 세밀한 token-level online intervention이며, 특히 FRGG는 `A/C/E` token features, faithful heads, late self-attention image columns, TopKMean gate를 모두 사용한다는 점에서 논문 정의와 더 직접적으로 연결된다.
다만 현재 확인된 결과물 기준으로는 이 둘이 메인 방법으로 채택되었다고 쓰기 어렵고, 실패하거나 유기된 branch로 정리하는 쪽이 더 정직하다.


## 9. 정리 한 줄

현재 실제 메인 결과로 연결된 `hard veto`는 **FRG/GMI 해석의 `C/E` feature를 쓰는 offline sample-level controller**다.
즉 `C = faithful_minus_global_attn`, `E = guidance_mismatch_score`가 핵심 축이고, 이 둘은 현재 구현상 **token-level 최종형이 아니라 sample-level proxy/summary**다.

`FRGG/RFHAR`는 레포 안에 구현과 실험 흔적은 남아 있지만, 내가 확인한 산출물 기준으로는 **메인 hard-veto 성과 경로라기보다 실패하거나 유기된 online branch**에 더 가깝다.
