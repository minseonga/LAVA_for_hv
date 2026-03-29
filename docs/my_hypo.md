Algorithm 1: Adaptive FAD Stage-2 with Limited Branching + Early Abort
# Purpose:
#   Replace full beam search with (i) limited branching on early anchor tokens
#   and (ii) early abort using running visual-fragility statistics, while keeping
#   the same selector/trigger logic as in Stage-2.

Inputs:
  - Image v, question q
  - Greedy champion c_greedy and its gate scores (VPMI(c_greedy), S_full(c_greedy))
  - Gate thresholds τ_vpmi, τ_sfull
  - Branching parameters: L (branching depth), K (branching width)
  - Early-abort thresholds: η_pm, η_word   (for running VPMI_min^pm and VPMI_min^word)
  - Selector parameters: δ_full, θ (prior-mask threshold), etc.
  - Trigger rule (Stage-2): switch if VPMI(c_safe) > VPMI(c_champ) and VPMI(c_champ) < 0

Outputs:
  - Final answer c_final

------------------------------------------------------------
Stage-1 (Gate):
1:  if NOT Expand(v,q,c_greedy):        # (VPMI(c_greedy)<τ_vpmi) AND (S_full(c_greedy)<τ_sfull)
2:      return c_greedy                # early exit

------------------------------------------------------------
Stage-2 (Limited Branching Candidate Generation):
# We create a small pool C_pool without full beam search.

3:  P ← { ε }                           # prefix set, initially empty prefix
4:  for t = 1..L do
5:      P_new ← ∅
6:      for each prefix p in P do
7:          z_vq ← ForwardLogits(v,q,p)          # logits for next token under image+question
8:          TopK ← TopKTokens(z_vq, K)           # get K candidate next tokens
9:          for each token a in TopK do
10:             P_new ← P_new ∪ { p ⊕ a }        # extend prefix by anchor token
11:     P ← Deduplicate(P_new)                   # optional: merge equivalent tokens/strings
12:     if |P| > K then                          # keep at most K prefixes overall (optional)
13:         P ← PruneByScore(P, criterion=S_full under vq)

------------------------------------------------------------
Stage-2 (Greedy Completion with Early Abort):
14: C_pool ← ∅
15: for each prefix p in P do
16:     stats ← InitRunningStats()               # holds running mins for pm/word + sums for VPMI
17:     y ← p
18:     aborted ← False
19:     while NOT StopCondition(y) do
20:         # 1) generate next token greedily (beam=1)
21:         z_vq ← ForwardLogits(v,q,y)
22:         a* ← Argmax(z_vq)
23:         y ← y ⊕ a*
24:
25:         # 2) update microscopic VPMI stats online (no storing per-step tensors)
26:         #    compute token-level logprobs for chosen token a* under vq and q-only:
27:         lp_vq ← LogProbChosenToken(v,q,y_prev,a*)     # log P(a* | v,q,y_prev)
28:         lp_q  ← LogProbChosenToken(q,  y_prev,a*)     # log P(a* | q,  y_prev)
29:         vpmi_tok ← lp_vq - lp_q
30:
31:         UpdateCoreSpanStats(stats, vpmi_tok, lp_q, token=a*, θ)
32:         # stats maintains:
33:         #   - running VPMI_min^pm over tokens with lp_q ≤ θ
34:         #   - running VPMI_min^word over completed words
35:         #   - running averages needed for VPMI(y) over T(y)
36:
37:         # 3) early abort if the candidate collapses into hallucination
38:         if stats.VPMI_min_pm < η_pm OR stats.VPMI_min_word < η_word then
39:             aborted ← True
40:             break
41:
42:     if aborted == False then
43:         c ← BuildCandidate(y, stats)          # attach S_full(c), VPMI(c), VPMI_min metrics
44:         C_pool ← C_pool ∪ {c}

------------------------------------------------------------
Stage-2 (Champion, Consensus Selector, Regularizer, Trigger):
45: if C_pool is empty then
46:     return c_greedy                           # conservative fallback (or keep greedy)

47: c_champ ← argmax_{c in C_pool} S_full(c)      # Stage-2 champion (beam-free)
48:
49: c_safe ← ConsensusTop1(C_pool, key1=VPMI_min_pm, key2=VPMI_min_word)
50: # ConsensusTop1 returns the candidate that is top-1 in both metrics, else None.
51:
52: if c_safe exists then
53:     if S_full(c_safe) - S_full(c_champ) ≤ δ_full then        # prior-defiance regularizer
54:         if VPMI(c_safe) > VPMI(c_champ) AND VPMI(c_champ) < 0 then
55:             return c_safe                                     # intervention
56:
57: return c_champ                                                # fallback to Stage-2 champion