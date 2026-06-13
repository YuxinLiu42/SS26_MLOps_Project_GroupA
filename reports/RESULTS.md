# Results — PaliGemma2-3B fine-tuned on ScienceQA (image subset)

> **STATUS: TEMPLATE / CURRENT RESULT (2026-06-13).** These numbers are from
> the current production adapter (`vague-sweep-3`, 64.1%), trained on the old
> 1,677-example slice of validation. A retrain on the **full** ScienceQA train
> split (~6,218 examples, LoRA r=16) is queued; when it lands, re-promote the
> winner and refresh the numbers/figures here (the pipeline and figure scripts
> stay identical — only the values change).

Self-contained results summary for the exam report (paste into Q12/Q14/Q17).
All numbers are exact-match accuracy of the generated answer **letter** on the
held-out ScienceQA-IMG test split (2017 samples).

## Headline

| Model | Test accuracy | Notes |
|---|---|---|
| Pre-fix baseline | 42.9% | prompt truncated the answer choices |
| Baseline (sweep #1 winner) | 58.85% | prompt fix; `base_lr` 1e-4 |
| **Production (sweep #2 winner, `vague-sweep-3`)** | **64.1%** (1293/2017) | **+5.3 pts over baseline** |

The production adapter lives at
`gs://mlops-paligemma-west4/models/production/` (W&B artifact
`scienceqa-paligemma2-lora:production`, version v3).

## Winning hyperparameters (`vague-sweep-3`)

| Hyperparameter | Value | Swept? |
|---|---|---|
| `model.base_learning_rate` | 1.89e-4 | yes (log-uniform 7e-5–2e-4) |
| effective learning rate | 1.89e-4 | derived: `base × √(eff_batch/16)` |
| `data.batch_size` | 4 | fixed |
| `trainer.accumulate_grad_batches` | 4 | yes ({2,4,8}) |
| effective batch size | 16 | = batch_size × accum |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 | fixed |
| LoRA target modules | q,k,v,o\_proj | fixed |
| vision encoder | frozen | fixed |
| gradient checkpointing | on | fixed |
| `max_length` | 512 | fixed |
| epochs | ≤ 8, EarlyStopping(patience=3) on `val/accuracy` (max) | fixed |

The learning rate is decoupled from accumulation via a √ batch-size rule
(`resolve_learning_rate` in `train.py`): the sweep searches `base_learning_rate`
defined at a reference effective batch of 16, so trials at different
accumulation are compared at a comparable LR.

## Sweep #2 — all trials (W&B sweep `xptwdnis`, Bayesian, metric `val/accuracy` max)

| Run | val/accuracy | val/loss | base_lr | accum (eff. batch) |
|---|---|---|---|---|
| **vague-sweep-3** (winner) | **0.7024** | 0.5111 | 1.89e-4 | 4 (16) |
| devout-sweep-7 | 0.6738 | 0.6007 | 1.78e-4 | 4 (16) |
| vague-sweep-4 | 0.6690 | 0.5303 | 1.96e-4 | 8 (32) |
| comfy-sweep-1 | 0.6500 | 0.6480 | 1.88e-4 | 2 (8) |
| playful-sweep-2 | 0.6381 | 0.6487 | 1.11e-4 | 2 (8) |
| daily-sweep-6 | 0.6357 | 0.5477 | 1.82e-4 | 8 (32) |
| azure-sweep-8 | 0.6310 | 0.7022 | 8.31e-5 | 2 (8) |
| dutiful-sweep-5 | 0.6190 | **0.4643** | 8.22e-5 | 8 (32) |

## Per-subject accuracy (production model)

| Subject | Accuracy | n |
|---|---|---|
| social science | 76.2% | 764 |
| natural science | 57.2% | 1209 |
| language science | 45.5% | 44 |

## Methodology note — why we optimise `val/accuracy`, not `val/loss`

Sweep #1 optimised `val/loss` and promoted a trial that lost to the baseline on
test accuracy. Sweep #2 confirms why: the two metrics **disagree**.
`dutiful-sweep-5` has the *best* `val/loss` (0.464) but nearly the *worst*
`val/accuracy` (0.619); the winner has a *higher* loss (0.511) but the *best*
accuracy (0.702). Because the task is scored on exact-match of the answer
letter, we log a generation-based `val/accuracy` each epoch and select
checkpoints / early-stop on it (`mode=max`). See `reports/figures/sweep2_comparison.png`.

The LR pattern also held: trials at `base_lr ≈ 1.8–1.96e-4` reached
0.65–0.70 `val/accuracy`; the two low-LR trials (~8e-5) sat at the bottom —
which is why sweep #2 raised the LR floor above sweep #1's dead zone.

## Artifact layout (`reports/`)

| Folder | Contents |
|---|---|
| `figures/` | `.png` visualizations (below) |
| `eval/` | evaluation data: `production_eval_results.json`, `sweep2_summary.json` |
| `monitoring/` | `drift_report.html` (Evidently) |
| `load/` | load-test summary + locust CSVs |

### Figures (`reports/figures/`)

| File | Shows |
|---|---|
| `accuracy_by_subject.png` | production model per-subject accuracy |
| `sweep2_comparison.png` | per-trial `val/accuracy` vs baseline line; `val/loss`↔`val/accuracy` disagreement |
| `prediction_length_dist.png` | predicted answer length (sanity: single letters) |
| `error_samples.png` | qualitative grid of misclassified samples |

Reproduce with the committed source JSONs:

```bash
python -m project_name.visualize subject-accuracy reports/eval/production_eval_results.json
python -m project_name.visualize sweep-comparison  reports/eval/sweep2_summary.json
python -m project_name.visualize pred-lengths       reports/eval/production_eval_results.json
```

## Cloud workload inventory (what runs where, and why)

Every GPU-bound workload runs on **Vertex AI custom jobs** (single L4,
`europe-west4`, Flex Start to queue for capacity). Everything else is CPU and
stays local/CI by design.

| Workload | Where | Entry point |
|---|---|---|
| Training (baseline) | Vertex L4 | `cloud/run_baseline_and_sweep.sh` (`SKIP_BASELINE=0`) |
| Hyperparameter sweep | Vertex L4 | same script → `wandb agent` |
| Best-adapter eval (chained) | Vertex L4 | same script, step [3/3] |
| Standalone adapter eval | Vertex L4 | `cloud/run_eval.sh` (any `ADAPTER_GCS`) |
| Image build | Cloud Build | `cloud/cloudbuild.train.yaml` |
| Serving / inference | on-demand container (local or Cloud Run) | `dockerfiles/api.dockerfile` |
| Data preprocessing | local / CI | `project_name.data` (CPU: resize + tokenise) |
| Report figures | local | `project_name.visualize` (reads eval JSON) |

Notes:
- Secrets (W&B key, HF token) are fetched at container start from Secret
  Manager via google-auth ADC (`cloud/fetch_secrets.sh`); job specs carry only
  secret **names**. Jobs run as the compute service account, which holds
  `secretmanager.secretAccessor`.
- Job images are pinned by **digest** at submit time (Vertex resolves `:latest`
  at container start, which can drift across a long Flex Start queue).
- Serving is intentionally **not** an always-on GPU endpoint: a 3B model needs a
  GPU for interactive latency, and an always-on L4 endpoint costs more than this
  project warrants. The API reads its adapter from `CHECKPOINT_PATH`, which
  accepts a `gs://` path, so promoting a new adapter needs no redeploy.
