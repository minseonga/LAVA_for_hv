# RF-HAR (RF-Guided Head-Aware Reweighting)

RF-HAR is a training-free late-layer attention-logit intervention for LLaVA/LLaMA-style decoding.

## Core behavior
- Applies only on decoder self-attention.
- Applies only on late layers (`rfhar_late_start..rfhar_late_end`).
- Applies only on the last query row.
- Applies only on image-token columns.
- Returns modified logits; normal model softmax is unchanged.

## Inputs
RF-HAR expects externally computed per-image-token features:

- `C`: faithful routing backbone (`[B,K_img]`)
- `A`: token-level grounding strength (`[B,K_img]`)
- `D`: harmful routing strength (`[B,K_img]`)
- `B`: temporal instability (`[B,K_img]`)

They are passed at generation time as `rfhar_feats={"C":..., "A":..., "D":..., "B":...}`.

## CLI (model_vqa_loader)

- `--enable-rfhar`
- `--rfhar-late-start`, `--rfhar-late-end`
- `--rfhar-r-percent`
- `--rfhar-gamma`
- `--rfhar-lambda-penalty`
- `--rfhar-eps`
- `--rfhar-debug-log`
- `--rfhar-feats-json` (per-sample feature file keyed by `id`/`question_id`)

If RF-HAR is enabled and no features are loaded, evaluation stops with an error.

## Safety checks
- `enable_rfhar=false` => baseline path.
- `rfhar_gamma=0` => baseline path.
- Text-token columns are untouched by design.
