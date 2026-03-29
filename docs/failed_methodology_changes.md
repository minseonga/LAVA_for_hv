# 실패/미채택 방법론 변경 사항 정리

이 문서는 `+0.027` 기준 조합 대비, 실제로 시도했지만 성능 향상에 실패했거나 채택하지 않은 변경안을 기록합니다.

## 기준(채택)

- 기준 조합: `P3 + agree_vminpm_wmin_dfull_le:-0.05`
- 결과: `base 0.622 -> final 0.649`, `delta +0.027` (`gain 41`, `harm 14`)
- 출처: `LLaVA_calibration/experiments/artrap_costaware_gen1/offline_sweep/policy_table.csv`

---

## 1) Dynamic Percentile Trigger (`P3Q_pctl:*`)

### 시도 의도
- 고정 임계값 대신 샘플별 VPMI 분포 퍼센타일을 사용해 trigger miss를 줄이려는 목적

### 결과
- `P3Q_pctl:70` best: `delta +0.014` (`final 0.636`)
- `P3Q_pctl:80` best: `delta +0.014` (`final 0.636`)
- `P3Q_pctl:90` best: `delta +0.016` (`final 0.638`)
- 모두 기준 `+0.027`보다 낮음

### 판단
- **미채택**

### 근거 파일
- `LLaVA_calibration/experiments/artrap_costaware_gen1/offline_sweep/policy_table.csv`

---

## 2) NTP_min 기반 selector 게이트

### 시도 의도
- token-level 최저 확률(NTP_min)로 위험한 후보를 제거해 harm를 더 줄이기

### 결과
- `agree_vminpm_wmin_ntpmin_ge:*` 계열이 스위칭을 거의 막음
- 예) `P3C_cvlt:-0.5 + ntpmin_ge:-5.0` -> `delta -0.003` (`gain 1`, `harm 4`)
- `-5.0 ~ -3.5` 전부 동일하게 음수 delta

### 판단
- **실패(과도한 보수화)**

### 근거 파일
- `LLaVA_calibration/experiments/artrap_fragility_1000_canonical_v2_tailfeat_ntp_feasibility/policy_table.csv`
- `LLaVA_calibration/experiments/artrap_fragility_1000_canonical_v2_tailfeat_ntpmin_sweep/policy_table.csv`

---

## 3) NTP spread / VPMI instability 게이트

### 시도 의도
- `S_core_img - S_core_img_min` 또는 `vpmi_core_min_mean_gap`으로 불안정 후보 필터링

### 결과
- `agree_vminpm_wmin_ntpdrop_le:*`, `agree_vminpm_wmin_vg_le:*` 모두 기준보다 낮음
- 상위도 대체로 `delta +0.010` 수준

### 판단
- **미채택**

### 근거 파일
- `LLaVA_calibration/experiments/artrap_fragility_1000_canonical_v2_tailfeat_ntp_feasibility/policy_table.csv`

---

## 4) Compact Pool 재생성 (`beam4 + extra sample2`)

### 시도 의도
- coverage를 늘리면서 비용을 줄이기 위해 `beam 6 -> 4` + `sample 2` 조합 실험

### 결과(핵심)
- 재생성셋 baseline 자체 하락: `0.622 -> 0.620`
- holdout_locked 성능 하락:
  - old: `delta +0.025`, `final 0.647`
  - new: `delta +0.018`, `final 0.638`
- leaky_best도 하락:
  - old: `+0.027`
  - new: `+0.015`
- coverage 악화:
  - recoverable `164 -> 155`
  - unrecoverable `214 -> 225`

### 판단
- **실패(coverage 개선 대신 악화)**

### 근거 파일
- `LLaVA_calibration/experiments/artrap_costaware_gen1/gen_beam4_extra2/summary.json`
- `LLaVA_calibration/experiments/artrap_costaware_gen1/gen_beam4_extra2/selector_eval/policy_table.csv`
- `LLaVA_calibration/experiments/artrap_costaware_gen1/gen_beam4_extra2/policy_diagnostics/gmu_table.csv`
- 비교 기준:
  - `LLaVA_calibration/experiments/artrap_costaware_gen1/offline_sweep/diagnostics/metrics.csv`
  - `LLaVA_calibration/experiments/artrap_costaware_gen1/offline_sweep/diagnostics/gmu_table.csv`

---

## 5) DBS (Diverse Beam Search) 전환

### 시도 의도
- 빔 다양성으로 정답 후보 포함률 증가 기대

### 결과
- DBS 런 baseline 자체 하락:
  - `accuracy_heuristic = 0.601`
- non-DBS canonical 계열(`0.622`) 대비 열세

### 판단
- **미채택**

### 근거 파일
- `LLaVA_calibration/experiments/artrap_fragility_1000_dbs/summary.json`
- 비교:
  - `LLaVA_calibration/experiments/artrap_fragility_1000_canonical_v2_tailfeat/summary.json`

---

## 6) Min-VPMI 단독 계열 확대

### 시도 의도
- core-min/word-min 중심 selector로 hallucination 억제

### 결과
- `artrap_fragility_1000_minvpmi_nodbs_selector_eval` best가 `delta +0.014`
- relaxed trigger(`P3C_cvlt:1.0`)는 `+0.011`
- 기준 `+0.027`보다 낮음

### 판단
- **보조 피처로는 유효하나, 단독 주력은 실패**

### 근거 파일
- `LLaVA_calibration/experiments/artrap_fragility_1000_minvpmi_nodbs_selector_eval/policy_table.csv`
- `LLaVA_calibration/experiments/artrap_fragility_1000_minvpmi_nodbs_selector_relax_eval/policy_table.csv`

---

## 요약 결론

1. 지금까지 실패한 공통 패턴은 “스위칭을 너무 막거나(보수화), 후보풀 자체를 악화시킨 경우”입니다.
2. 현재 확정 기준은 여전히:
   - `P3 + agree_vminpm_wmin_dfull_le:-0.05` (leaky best, `+0.027`)
   - 또는 공정 보고용 `P3C_cvlt:-0.5 + agree_vminpm_wmin_dfull_le:-0.08` (holdout-locked)
3. 다음 실험은 `beam 축소 없이 coverage 개선`이 우선이며, 현재 실패한 `beam4+extra2` 조합은 재사용하지 않습니다.

