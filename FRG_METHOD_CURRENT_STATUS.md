# FRG / Cheap Proxy Current Method Status

## 문서 목적

이 문서는 현재 저장소에서 `FRG idea -> Stage-B verifier -> expensive reference B+C -> aligned cheap proxy`로 이어진 방법론을 처음부터 끝까지 한 번에 정리한 최신 상태 문서다.

목표는 네 가지다.

1. 무엇이 이미 실험으로 입증되었는지와 무엇이 아직 미완인지 분리한다.
2. 현재 논문에 넣을 수 있는 결과와 넣으면 안 되는 결과를 분리한다.
3. 현재 메인 method가 무엇인지, reference가 무엇인지, cheap proxy가 무엇인지 혼동하지 않게 한다.
4. 다음 단계인 `sequential replay`와 `latency 측정`이 왜 필요한지 명확히 남긴다.

---

## 1. 원래 문제와 첫 실패

출발 문제는 간단하다.

- `VGA`, `VISTA`, `EAZY` 같은 intervention은 평균적으로 이득이 있어도 모든 샘플에서 항상 안전하지는 않다.
- 일부 샘플에서는 intervention이 baseline보다 오히려 나빠진다.
- 따라서 `intervention을 먼저 쓰되, 위험한 샘플만 다시 baseline으로 rescue`하는 selective routing이 필요했다.

처음에는 이를 `FRG scalar thresholding`으로 풀려고 했다.

- offline trace extractor에서 FRG scalar를 만든다.
- discovery에서 `tau`를 고른다.
- online inference에서도 같은 `tau`를 쓰면 된다고 가정했다.

이 라인은 현재 결론상 실패다.

- offline에서는 signal이 있었다.
- 하지만 offline scalar ordering과 online runtime ordering이 잘 맞지 않았다.
- 즉 `offline FRG scalar -> online tau transfer`는 deployment controller로 서지 않았다.

현재 판정은 다음과 같다.

- `FRG as idea`: KEEP
- `current scalar family`: DEMOTE
- `offline scalar -> online tau transfer controller`: KILL

---

## 2. 무엇이 살아남았는가: Stage-B verifier

FRG-only controller는 실패했지만, path-level verifier signal 자체는 살아남았다.

현재 이해는 이렇다.

- Stage-B score는 `harmful vs helpful clean separator`는 아니다.
- 하지만 `risky intervention outputs`를 low-score tail에 농축시키는 `risk detector`로는 성립한다.

실험상 이미 확인된 것:

- discovery와 held-out POPE 모두에서 `regression vs non-regression` 분리는 유지되었다.
- 다만 `regression vs improvement` 분리는 약했다.
- 따라서 Stage-B의 올바른 해석은 `harm detector`보다 `risky-pool miner`다.

이게 중요했던 이유는:

- Stage-B가 완전히 죽은 게 아니라면,
- expensive verifier를 reference policy로 쓸 수 있고,
- 그 decision을 cheap proxy로 근사하는 다음 단계가 가능해지기 때문이다.

---

## 3. Expensive Reference B+C가 무엇인가

현재 `reference B+C`는 다음을 의미한다.

1. VGA output을 canonical generation path로 먼저 만든다.
2. expensive verifier를 돌린다.
   - B-stage: Stage-B risk signal
   - C-stage: cheap feature family 또는 aligned feature family
3. 위험하다고 판단되면 baseline으로 rescue한다.

중요한 점은 이 reference가 지금의 `deployable method`가 아니라는 것이다.

현재 역할은:

- `scientific reference`
- `same-runtime expensive verifier`
- `upper-quality teacher`

즉 reference는 accuracy 측면에선 강하지만 비용이 너무 크다.

현재 reference의 핵심 held-out 결과:

- canonical VGA 기준 `intervention_acc = 0.866111...`
- fixed policy apply 후 `final_acc = 0.875111...`
- `delta_vs_intervention = +0.009`

이 숫자는 여전히 유효하다.
다만 이 reference는 `3x+`에 가까운 구조라 deployment main method로 밀면 안 된다.

---

## 4. 왜 첫 decode-time proxy는 실패했는가

다음 단계로 expensive reference를 cheap proxy가 근사할 수 있는지 보려고 했다.

처음 시도는 `fresh live decode-time proxy feature extraction`이었다.

문제는 두 가지였다.

### 4.1 Teacher mismatch

- proxy feature는 `freshly generated VGA outputs`에서 뽑혔다.
- 그런데 target label은 기존 reference `decision_rows.csv`에서 왔다.
- 즉 feature가 본 candidate path와 label이 정의된 candidate path가 달랐다.

이 상태에서는 `reference_rescue` imitation도, `actual_rescue` prediction도 제대로 될 수 없었다.

### 4.2 VGA path mismatch

더 큰 문제는 proxy extractor가 canonical VGA path 자체를 바꿔버렸다는 점이다.

- reference VGA는 canonical `VGA_origin` 경로에서 생성된 output이다.
- initial proxy extractor는 다른 adapter 경로로 VGA를 새로 생성했다.
- 그 결과 VGA 자체의 accuracy가 reference와 달라졌다.

즉 이 실험은 원래 질문인

`같은 VGA output 위에서 cheap verifier가 expensive verifier를 근사할 수 있는가?`

를 본 것이 아니라,

`다른 VGA path 위에서 proxy가 먹히는가?`

를 본 셈이 되었다.

그래서 그 결과는 메인 비교에서 폐기했다.

---

## 5. 현재 올바른 cheap proxy 정의

지금 메인 cheap proxy 실험은 반드시 `canonical VGA outputs` 위에서 돌아가야 한다.

즉 현재 올바른 구조는:

1. canonical VGA output은 reference artifact에서 그대로 가져온다.
2. 그 canonical candidate에 대해 cheap feature를 계산한다.
3. discovery에서 `actual_rescue` 기준으로 proxy threshold를 고른다.
4. held-out에 frozen policy를 sample-wise apply한다.

이 alignment를 맞춘 뒤 얻은 결과가 현재 가장 중요한 메인 결과다.

held-out aligned cheap proxy 결과:

- feature: `cheap_lp_content_min`
- rescue rate: `0.022`
- intervention_acc: `0.866111...`
- final_acc: `0.878333...`
- delta_vs_intervention: `+0.012222...`
- actual precision: `0.636`
- actual recall: `0.364`
- actual F1: `0.463`

이 결과의 의미:

- cheap proxy가 canonical VGA path를 해치지 않았다.
- same-runtime cheap signal 하나만으로 held-out에서 실제 이득이 났다.
- current reference `+0.009`보다도 더 좋은 held-out gain이 나왔다.

따라서 이제 메인 candidate는 `aligned cheap proxy`다.

---

## 6. `cheap_lp_content_min`이 정확히 무엇인가

현재 가장 잘 먹는 feature는 `cheap_lp_content_min`이다.

정의:

1. 이미 생성된 VGA 답변을 real image 위에서 teacher-forced로 다시 넣는다.
2. 각 output token 위치에서, 실제 생성된 target token의 `log-prob`를 계산한다.
3. 그중 문자/숫자가 있는 `content token`만 남긴다.
4. 그 content token들의 log-prob 중 최소값을 취한다.

즉 해석은:

- 답변 안의 핵심 token들 중
- 가장 취약하고 가장 불안정한 token 하나의 confidence를 보는 feature다.

이 feature가 잘 먹는 이유는:

- harmful output은 전체 평균보다도
- 일부 핵심 token이 극단적으로 불안정한 경우가 많기 때문이다.

중요한 제한:

- 이 feature는 `blur/no-image control`은 없다.
- attention dump도 없다.
- 하지만 아직 `완전한 1-pass decode-time`은 아니다.
- 현재는 `extra real-image teacher-forced 1 pass`가 필요하다.

즉 이 라인의 정확한 표현은:

- `cheap same-runtime proxy`
- `single-extra-pass proxy`

이지,

- `zero-extra-pass proxy`
- `pure decode-time no-replay method`

는 아니다.

---

## 7. 현재 method를 단계별로 쓰면

현재 method는 이렇게 정의하는 것이 가장 정확하다.

### 7.1 Inputs

- canonical VGA prediction jsonl
- question file
- image folder
- discovery에서 고정한 proxy policy

### 7.2 Discovery calibration

discovery에서만 다음을 고른다.

- feature choice
- direction (`low` / `high`)
- threshold `tau`

현재 canonical aligned result에서 선택된 policy는:

- feature: `cheap_lp_content_min`
- direction: `low`
- rescue threshold는 discovery에서 고정된 절대값

### 7.3 Held-out apply

held-out에서는:

1. VGA prediction text는 canonical artifact 그대로 사용
2. sample마다 cheap feature 계산
3. frozen threshold로 rescue 여부 결정
4. rescue면 baseline fallback
5. 최종 정확도 계산

중요한 점:

- held-out percentile recalibration 없음
- held-out mean/z-score normalization 없음
- taxonomy/correctness label이 decision logic에 직접 들어가지 않음

즉 sample-wise frozen policy application 조건을 만족한다.

---

## 8. 지금 논문에 넣을 수 있는 것과 없는 것

### 넣을 수 있는 것

1. `reference B+C`는 expensive scientific reference로 넣을 수 있다.
2. `aligned cheap proxy`는 main method candidate로 넣을 수 있다.
3. discovery-fit -> held-out fixed-policy 결과는 본문 수치로 넣을 수 있다.

### 넣으면 안 되는 것

1. held-out 내부에서 다시 threshold를 sweep한 best row
2. stale output 기준 target으로 calibration한 proxy 결과
3. VGA path가 canonical과 다른 fresh proxy generator 결과

즉 지금 본문에 들어갈 수 있는 건 `aligned cheap proxy`와 `reference B+C`다.

---

## 9. 지금 아직 안 끝난 것: live sequential replay

현재 aligned cheap proxy 결과는 매우 좋지만, 아직 `materialized frozen-policy apply`다.

즉 아직 안 한 것은:

- sample-wise sequential replay
- 실제 wall-clock latency 측정
- 실제 live baseline rescue generation

그래서 다음 실험이 필요하다.

### 목적

`cheap proxy를 실제로 live처럼 돌렸을 때도 accuracy가 유지되고 latency가 reference보다 충분히 낮은가?`

### 현재 구조상 비용

aligned cheap proxy의 구조적 비용은:

1. VGA generation 1회
2. cheap feature extraction 1회
3. rescue된 샘플만 baseline generation 1회

즉 pass proxy는 대략:

`2 + rescue_rate`

현재 held-out rescue rate가 `2.2%`이므로 구조적으로는 약 `2.022x`다.

하지만 실제 latency는 pass 수와 정확히 같지 않다.

- cheap pass는 full verifier보다 훨씬 가볍다.
- blur pass와 attention dump가 없다.
- output 길이도 짧다.

그래서 실제 기대 latency는 `2.022x`보다 더 낮을 가능성이 있다.

이걸 확인하려고 `sequential replay` 실험이 필요하다.

---

## 10. 새로 추가한 sequential replay 실험의 역할

이 저장소에는 이제 다음 실험용 runner가 추가되어 있다.

- [run_aligned_cheap_proxy_sequential_replay.py](./scripts/run_aligned_cheap_proxy_sequential_replay.py)
- [run_aligned_cheap_proxy_sequential_replay.sh](./scripts/run_aligned_cheap_proxy_sequential_replay.sh)

이 실험의 목적은:

- canonical VGA prediction을 유지한 상태에서
- cheap feature extraction을 live로 다시 돌리고
- rescue된 샘플만 baseline을 live로 생성해서
- post-VGA latency와 end-to-end final accuracy를 측정하는 것이다

현재 이 runner가 하는 일:

1. canonical VGA prediction jsonl을 읽는다.
2. sample마다 cheap feature를 live로 다시 계산한다.
3. frozen aligned proxy policy를 적용한다.
4. rescue면 baseline을 live로 생성한다.
5. 최종 output / 최종 accuracy / cheap pass 시간 / baseline rescue 시간을 저장한다.

중요:

- 이 runner는 canonical VGA output을 재생산하지 않는다.
- 즉 VGA path를 바꾸지 않는다.
- 대신 `cheap stage + rescue stage`를 live로 replay한다.

이건 현재 단계에서 가장 안전한 latency sanity check다.

---

## 11. 현재 저장소 기준 주요 스크립트

### Reference

- [run_paper_main_b_c_vga_full.sh](./scripts/run_paper_main_b_c_vga_full.sh)
- [run_main_b_c_fixed_policy_vga.sh](./scripts/run_main_b_c_fixed_policy_vga.sh)
- [run_b_c_fixed_policy.py](./scripts/run_b_c_fixed_policy.py)

### Aligned cheap proxy

- [run_aligned_cheap_proxy_from_reference_vga.sh](./scripts/run_aligned_cheap_proxy_from_reference_vga.sh)
- [run_decode_time_proxy_policy.py](./scripts/run_decode_time_proxy_policy.py)
- [extract_c_stage_cheap_online_features.py](./scripts/extract_c_stage_cheap_online_features.py)
- [build_vga_failure_taxonomy.py](./scripts/build_vga_failure_taxonomy.py)

### Sequential replay

- [run_aligned_cheap_proxy_sequential_replay.sh](./scripts/run_aligned_cheap_proxy_sequential_replay.sh)
- [run_aligned_cheap_proxy_sequential_replay.py](./scripts/run_aligned_cheap_proxy_sequential_replay.py)

---

## 12. 현재 최종 판단

현재 상태를 가장 정확히 쓰면 이렇다.

1. `FRG-only online controller`는 실패했다.
2. `Stage-B`는 risk detector로 살아남았다.
3. `reference B+C`는 expensive scientific reference로는 성립한다.
4. `aligned cheap proxy`는 held-out fixed policy에서 실제로 강한 gain을 냈다.
5. 따라서 현재 main method candidate는 `aligned cheap same-runtime proxy`다.
6. 이제 남은 핵심은 `sequential replay`로 실제 latency와 live end-to-end 정확도를 확인하는 것이다.

즉 현재 method line은 더 이상

`offline FRG scalar -> online threshold transfer`

가 아니라,

`canonical VGA -> cheap same-runtime proxy -> selective baseline rescue`

다.

---

## 13. 지금 가장 방어 가능한 논문 문장

지금 시점에서 가장 방어 가능한 문장은 이 정도다.

> We first establish an expensive verifier as a scientific reference, then show that a much cheaper same-runtime proxy, computed from the intervention output itself, can recover substantial rescue-worthiness signal under a frozen discovery-to-held-out protocol.

그리고 practical claim은 아직 이렇게 제한해야 한다.

> The current cheap proxy is promising but still requires sequential replay measurement before making a final deployment-efficiency claim.

