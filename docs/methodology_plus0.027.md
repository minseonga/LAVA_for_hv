# FAD +0.027 방법론 정리 (Reproducible Note)

이 문서는 `+0.027` 성능 향상을 만든 실험 구성을 처음 보는 사람도 재현할 수 있도록 정리한 메모입니다.

## 1) 무엇이 +0.027인가

- 기준 데이터: `1000`문항 GQA subset
- 기준 베이스라인 정확도: `0.622`
- 최종 정확도: `0.649`
- 정확도 증가: `+0.027`
- 해당 조합:
  - Trigger: `P3`
  - Selector policy: `agree_vminpm_wmin_dfull_le:-0.05`
- 결과 출처:
  - `LLaVA_calibration/experiments/artrap_costaware_gen1/offline_sweep/policy_table.csv`
  - `LLaVA_calibration/experiments/artrap_costaware_gen1/offline_sweep/diagnostics/metrics.csv`


## 2) 입력 아티팩트(재평가 대상)

`+0.027`은 다음 `in_dir`의 후보/점수 CSV를 오프라인 selector-tradeoff로 평가해 얻은 값입니다.

- `in_dir`: `LLaVA_calibration/experiments/artrap_fragility_1000_canonical_v2_tailfeat`
- 생성 설정 요약 (`summary.json` 기준):
  - model: `liuhaotian/llava-v1.5-7b`
  - `num_beams=6`, `num_return_sequences=6`
  - `max_new_tokens=24`
  - `attn_impl=sdpa`
  - eval match mode: `heuristic`

원본 요약 파일:
- `LLaVA_calibration/experiments/artrap_fragility_1000_canonical_v2_tailfeat/summary.json`


## 3) 방법론 개요 (파이프라인)

## Step A. 후보 풀과 점수

각 샘플에 대해 챔피언(`c_full`)은 `S_full` 최대 후보입니다.

- `S_full(c)`: 이미지 조건 전체 토큰 평균 로그확률
- `S_core_img(c)`: core answer span 토큰 평균 로그확률 (image-conditioned)
- `S_q(c)`: core span 토큰 평균 로그확률 (question-only)
- `VPMI(c) = S_core_img(c) - S_q(c)`


## Step B. 핵심 selector (`agree_vminpm_wmin`)

도전자 후보는 2-view agreement로 선택합니다.

1. `vpmi_core_min_prior_masked` top-1 후보 선택
2. `vpmi_word_min` top-1 후보 선택
3. 둘의 후보 인덱스가 같을 때만 도전자 인정

해당 구현 위치:
- `LLaVA_calibration/eval_selector_tradeoff.py:357`


## Step C. 안전 게이트 추가 (`dfull`)

`agree_vminpm_wmin_dfull_le:-0.05`는 Step B 후보에 다음 조건을 추가합니다.

- `S_full(c_safe) - S_full(c_full) <= -0.05`

즉, agreement 후보라도 챔피언 대비 전체 점수 차가 너무 나면 버립니다.

해당 구현 위치:
- `LLaVA_calibration/eval_selector_tradeoff.py:385`


## Step D. Trigger (`P3`)

스위칭 조건:

- `VPMI(c_safe) > VPMI(c_full)`
- `VPMI(c_full) < 0`

둘 다 참일 때만 챔피언에서 도전자로 교체합니다.

해당 구현 위치:
- `LLaVA_calibration/eval_selector_tradeoff.py:669`


## 4) 재현 커맨드

오프라인 평가(+0.027 포함 top 조합 표):

```bash
python /home/kms/LLaVA_calibration/eval_selector_tradeoff.py \
  --in_dir /home/kms/LLaVA_calibration/experiments/artrap_fragility_1000_canonical_v2_tailfeat \
  --out_dir /home/kms/LLaVA_calibration/experiments/artrap_costaware_gen1/offline_sweep \
  --eval_mode heuristic \
  --triggers "P3,P3C_cvlt:-0.5,P3Q_pctl:70,P3Q_pctl:80,P3Q_pctl:90" \
  --policies "max_vpmi;agree_vminpm_wmin_dfull_le:-0.08;agree_vminpm_wmin_dfull_le:-0.05;agree_vminpm_wmin;max_vpmi_core_min_prior_masked_tb_vpmi"
```

진단(CI, McNemar, G/M/U, oracle bottleneck):

```bash
python /home/kms/LLaVA_calibration/eval_policy_diagnostics.py \
  --in_dir /home/kms/LLaVA_calibration/experiments/artrap_fragility_1000_canonical_v2_tailfeat \
  --out_dir /home/kms/LLaVA_calibration/experiments/artrap_costaware_gen1/offline_sweep/diagnostics \
  --eval_mode heuristic \
  --configs "ref_max_vpmi|P3|max_vpmi;holdout_locked|P3C_cvlt:-0.5|agree_vminpm_wmin_dfull_le:-0.08;dynamic_p80|P3Q_pctl:80|agree_vminpm_wmin_dfull_le:-0.08;leaky_best|P3|agree_vminpm_wmin_dfull_le:-0.05" \
  --ref_name ref_max_vpmi \
  --bootstrap_n 5000 \
  --bootstrap_seed 123
```


## 5) +0.027 결과 요약

`leaky_best` (`P3 + agree_vminpm_wmin_dfull_le:-0.05`) 기준:

- `base_acc=0.622`
- `final_acc=0.649`
- `delta_acc=+0.027`
- `gain=41`, `harm=14`
- `switch_rate=0.151`
- `precision_gain=0.745`
- McNemar exact: `p=3.55e-4`
- 95% CI (bootstrap): `delta_acc [0.013, 0.042]`

출처:
- `LLaVA_calibration/experiments/artrap_costaware_gen1/offline_sweep/diagnostics/ci_significance.csv`


## 6) G/M/U 및 병목 해석 (+0.027 기준)

`base_wrong=378`에서:

- `G=41` (복구 성공)
- `M=123` (복구 가능했지만 실패)
- `U=214` (후보풀에 정답 없음)

oracle 분해:

- actual delta: `0.027`
- trigger-constrained oracle delta: `0.063`
- pool oracle delta: `0.164`
- 남은 병목:
  - selector/trigger gap: `0.036`
  - coverage gap(정답 후보 미포함 등): `0.101`

출처:
- `LLaVA_calibration/experiments/artrap_costaware_gen1/offline_sweep/diagnostics/gmu_table.csv`
- `LLaVA_calibration/experiments/artrap_costaware_gen1/offline_sweep/diagnostics/oracle_bottleneck.csv`


## 7) 중요한 해석 주의

- 이 `+0.027`은 같은 test-set 아티팩트 위에서 정책을 고른 값(`leaky best`)입니다.
- 더 보수적인 보고는 holdout-locked 조합(`P3C_cvlt:-0.5 + agree_vminpm_wmin_dfull_le:-0.08`)을 권장합니다.
- holdout-locked는 동일 조건에서 `+0.025`였습니다.


## 8) 코드 참조 포인트

- Selector/Trigger 구현:
  - `LLaVA_calibration/eval_selector_tradeoff.py:357`
  - `LLaVA_calibration/eval_selector_tradeoff.py:385`
  - `LLaVA_calibration/eval_selector_tradeoff.py:669`
- 후보 및 토큰 점수 생성:
  - `LLaVA_calibration/analyze_artrap_pairwise_fragility.py`
- 진단(CI/유의성/GMU/oracle):
  - `LLaVA_calibration/eval_policy_diagnostics.py`

