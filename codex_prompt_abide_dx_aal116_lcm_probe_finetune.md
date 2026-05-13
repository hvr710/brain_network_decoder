# Codex Prompt: ABIDE dx AAL116 LCM Finetune + LP/MLP

You are Codex running on the 4090 server. Please implement and prepare a runnable experiment in the existing LCM repo.

## 0. Context

Repo path:

```bash
/mnt/dataset3/nzh/fmri_baseline/brain_network_decoder
```

Conda env:

```bash
conda activate lcm
```

Task:

```text
ABIDE dx / diagnosis classification with AAL116
```

This is binary classification:

```text
ASD vs Control
```

Metadata columns are expected to include:

```text
Subject, SITE_ID, FILE_ID, DX_GROUP, DSM_IV_TR, age, Gender, HANDEDNESS_CATEGORY
```

The current CSV appears to use:

```text
DX_GROUP = 0 -> ASD
DX_GROUP = 3 -> Control
```

Please verify this with `DSM_IV_TR`: ASD rows usually have DSM_IV_TR in {1,2,3,4}, while controls usually have DSM_IV_TR = 0.

For training, map labels as:

```text
ASD      -> 1
Control  -> 0
```

## 1. Required experiments

Please support three experiment types:

```text
1. ABIDE dx + AAL116 + LCM pretrained frozen features + LP
2. ABIDE dx + AAL116 + LCM pretrained frozen features + MLP
3. ABIDE dx + AAL116 + LCM pretrained full finetune
```

Priority:
1. Make the official-style LCM finetune runnable.
2. Make LP/MLP probe runnable.
3. Keep code clean and do not break original repo.

Do not implement any other atlas. This task is AAL116 only.

## 2. Pretrained weights

The pretrained weights have already been placed under:

```bash
/mnt/dataset3/nzh/fmri_baseline/brain_network_decoder/model_weights/none_decoder32_ppmi-abide-taowu-neurocon-hcpa-adni-hcpya_boldwin500_FCFC
```

The two confirmed files are:

```bash
/mnt/dataset3/nzh/fmri_baseline/brain_network_decoder/model_weights/none_decoder32_ppmi-abide-taowu-neurocon-hcpa-adni-hcpya_boldwin500_FCFC/bb_fold0_hcpaBest_2025-01-20-02-09-00-551325.pt
/mnt/dataset3/nzh/fmri_baseline/brain_network_decoder/model_weights/none_decoder32_ppmi-abide-taowu-neurocon-hcpa-adni-hcpya_boldwin500_FCFC/head_fold0_hcpaBest_2025-01-20-02-09-00-551325 .pt
```

Important: the `head_fold0...` filename contains a space before `.pt`:

```text
...551325 .pt
```

Please handle this exact filename correctly.

Before any run, print whether both files exist.

## 3. Data paths

AAL116 ROI data should be under:

```bash
/mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL
```

Try these metadata CSV paths first:

```bash
/mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/ABIDE.csv
/mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/ABIDE.csv
/mnt/dataset3/nzh/fmri_baseline/brain_network_decoder/data/ABIDE.csv
/mnt/dataset3/nzh/fmri_baseline/brain_network_decoder/ABIDE.csv
```

If not found, search conservatively:

```bash
find /mnt/dataset4/DATASETS/fmri_pretraining -iname "*ABIDE*.csv" -o -iname "*abide*.csv" | head
```

If still not found, stop with a clear error telling me where to place `ABIDE.csv`.

## 4. Add files, avoid breaking original code

Please add new scripts rather than heavily rewriting original code.

Suggested files:

```text
scripts/abide_dx_aal116_lcm_probe.py
scripts/abide_dx_aal116_lcm_finetune.py
scripts/run_abide_dx_aal116_lcm_probe.sh
scripts/run_abide_dx_aal116_lcm_finetune.sh
scripts/run_abide_dx_aal116_all.sh
README_abide_dx_aal116_lcm.md
```

You may add helper modules if needed:

```text
scripts/abide_lcm_utils.py
```

Do not delete or move original files.

Do not change original `finetune_lbnm.py`, `datasets.py`, or model files unless absolutely necessary. If you must change them, keep the change minimal and explain why.

## 5. Data loading and matching

Inspect filenames under:

```bash
/mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL
```

Match each ROI file to metadata using either `FILE_ID` or `Subject`.

Examples of possible IDs:

```text
Pitt_0050003
50003
0050003
```

Please implement robust matching:
- normalize filename stem and metadata IDs
- try exact match first
- then try subject-number match
- print unmatched file examples and unmatched metadata examples
- do not silently drop many subjects without reporting

Support common formats:

```text
.npy, .npz, .txt, .csv, .tsv
```

For each subject:
- load time series
- infer whether shape is `[T, 116]` or `[116, T]`
- convert to `[T, 116]`
- require ROI number = 116
- compute FC matrix using Pearson correlation across ROI
- replace NaN/Inf with zero

For LCM input, follow the original repo's AAL116 FC input convention as closely as possible. Inspect `datasets.py`, `finetune_lbnm.py`, and model code to determine whether the model expects full FC matrix, row-wise FC node features, adjacency FC, and normalization/tensor shape.

## 6. LP/MLP probe requirements

Implement:

```bash
--feature_source lcm_frozen
```

This should:
1. Load the confirmed pretrained backbone and head files.
2. Build the same AAL116 FC input format expected by the model.
3. Freeze all model parameters.
4. Extract one fixed representation per subject.
5. Save extracted features to disk so LP/MLP can reuse them.

Suggested cache:

```text
outputs/abide_dx_aal116_lcm_probe/feature_cache/lcm_frozen_fold0_features.npz
```

The cache should include:

```python
X
y
subject_id
file_id
site_id
age
gender
feature_source
```

Also implement a simple fallback for debugging:

```bash
--feature_source fc_vector
```

For this fallback, use upper triangular FC values excluding diagonal:

```python
idx = np.triu_indices(116, k=1)
x = fc[idx]
```

This gives 6670-dimensional features.

Default probe feature source should be:

```bash
--feature_source lcm_frozen
```

### LP config

```python
sklearn.linear_model.LogisticRegression(
    C=1.0,
    max_iter=8000,
    random_state=seed,
    solver="lbfgs",
    class_weight="balanced"
)
```

Report:
- accuracy
- macro-F1
- ASD-F1 if easy
- confusion matrix

### MLP config

```python
hidden_dim = 64
dropout = 0.1
num_layers = 2
input_norm = "layernorm"
```

Architecture:

```python
LayerNorm(input_dim)
Linear(input_dim, hidden_dim)
ReLU
Dropout(0.1)
Linear(hidden_dim, 2)
```

Training defaults:
- AdamW
- lr = 1e-3
- weight_decay = 1e-4
- max_epochs = 200
- patience = 30
- batch_size = 32
- best checkpoint selected by val cross-entropy loss
- CUDA if available
- class weights in CE loss if imbalanced

## 7. Finetune requirements

Implement full LCM finetune for ABIDE dx AAL116.

### 7.1 Use original repo logic as much as possible

Before implementing, inspect:
- `README.md`
- `finetune_lbnm.py`
- `datasets.py`
- model definitions under `models/`
- any existing ABIDE/PPMI/ADNI data logic

Goal: run the official-style LCM finetuning, not a new model.

The finetune should use:
- AAL116 input
- pretrained `bb_fold0...pt` and `head_fold0...pt`
- original LCM decoder/head logic
- original loss/training logic where possible
- subject-level split
- ABIDE dx label

Do not replace finetune with frozen-feature MLP. Full finetune means the LCM backbone/head parameters are trainable according to the original repo's finetune procedure.

### 7.2 Preferred implementation strategy

If original `finetune_lbnm.py` already supports ABIDE dx AAL116 with the current data path and metadata, create a shell wrapper that calls it with correct arguments.

If original `finetune_lbnm.py` does not support the current data layout, add a new script:

```text
scripts/abide_dx_aal116_lcm_finetune.py
```

that reuses original model classes and training logic as much as possible.

If you need a dataset adapter, implement it in a helper file instead of rewriting original `datasets.py`.

### 7.3 Finetune split and runs

Use 5 runs/seeds:

```python
base_seed = 23
num_runs = 5
seeds = [23, 24, 25, 26, 27]
```

Default split:
- 70% train
- 15% val
- 15% test
- subject-level
- stratified by diagnosis label if possible

For each run:
- initialize from pretrained weights
- train on train set
- choose best checkpoint by val macro-F1 or val loss; prefer the repo's original criterion if it has one
- evaluate once on test set using best-val checkpoint
- save metrics/checkpoints/logs

If the original repo uses fold-based logic, keep it compatible, but still allow 5-run seed-based splits for this task.

### 7.4 Finetune hyperparameters

Start with original repo defaults. Do not invent aggressive changes.

If a smoke test is needed, allow:

```bash
--epochs 2
--batch_size 8
```

For full run, choose reasonable defaults from original repo, or expose args:

```bash
--epochs 50
--batch_size 16
--lr <repo default>
```

Print the actual values used.

### 7.5 Finetune outputs

Save under:

```text
outputs/abide_dx_aal116_lcm_finetune/
```

Suggested structure:

```text
outputs/abide_dx_aal116_lcm_finetune/
  data_check.json
  run_0_seed_23/
    config.json
    train_log.csv
    best_checkpoint.pt
    metrics.json
    confusion_matrix.csv
  run_1_seed_24/
    ...
  summary.json
  summary.csv
```

Metrics:
- test accuracy
- test macro-F1
- ASD-F1 if easy
- val best epoch
- confusion matrix

## 8. Splitting logic for probe and finetune

For all modes:
- Use subject-level split.
- Use the same split seeds for LP, MLP, and finetune.
- Save split files so comparisons are fair.

Suggested split cache:

```text
outputs/abide_dx_aal116_splits/
  run_0_seed_23_split.json
  run_1_seed_24_split.json
  ...
```

Each split JSON should include subject IDs for train/val/test.

Make the scripts reuse existing split files if present.

## 9. Output structure

Probe outputs:

```text
outputs/abide_dx_aal116_lcm_probe/
  data_check.json
  feature_cache/
    lcm_frozen_fold0_features.npz
  lcm_frozen/
    lp/
      run_0_seed_23/
        config.json
        metrics.json
        confusion_matrix.csv
      ...
      summary.json
      summary.csv
    mlp/
      run_0_seed_23/
        config.json
        metrics.json
        best_model.pt
        train_log.csv
      ...
      summary.json
      summary.csv
```

Finetune outputs:

```text
outputs/abide_dx_aal116_lcm_finetune/
  data_check.json
  run_0_seed_23/
    config.json
    train_log.csv
    best_checkpoint.pt
    metrics.json
    confusion_matrix.csv
  ...
  summary.json
  summary.csv
```

## 10. Shell runners

Create:

```bash
scripts/run_abide_dx_aal116_lcm_probe.sh
```

It should run:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /mnt/dataset3/nzh/fmri_baseline/brain_network_decoder
source ~/.bashrc || true
conda activate lcm

python scripts/abide_dx_aal116_lcm_probe.py \
  --roi_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL \
  --feature_source lcm_frozen \
  --mode both \
  --num_runs 5 \
  --base_seed 23 \
  --out_dir outputs/abide_dx_aal116_lcm_probe
```

Create:

```bash
scripts/run_abide_dx_aal116_lcm_finetune.sh
```

It should run the full finetune, for example:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /mnt/dataset3/nzh/fmri_baseline/brain_network_decoder
source ~/.bashrc || true
conda activate lcm

python scripts/abide_dx_aal116_lcm_finetune.py \
  --roi_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL \
  --num_runs 5 \
  --base_seed 23 \
  --epochs 50 \
  --batch_size 16 \
  --out_dir outputs/abide_dx_aal116_lcm_finetune
```

Create:

```bash
scripts/run_abide_dx_aal116_all.sh
```

It should run probe then finetune:

```bash
bash scripts/run_abide_dx_aal116_lcm_probe.sh
bash scripts/run_abide_dx_aal116_lcm_finetune.sh
```

Make all scripts executable:

```bash
chmod +x scripts/run_abide_dx_aal116_lcm_probe.sh
chmod +x scripts/run_abide_dx_aal116_lcm_finetune.sh
chmod +x scripts/run_abide_dx_aal116_all.sh
```

## 11. Smoke tests

First run LP smoke test:

```bash
python scripts/abide_dx_aal116_lcm_probe.py \
  --roi_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL \
  --feature_source lcm_frozen \
  --mode lp \
  --num_runs 1 \
  --base_seed 23 \
  --out_dir outputs/abide_dx_aal116_lcm_probe_smoke
```

Then run finetune smoke test:

```bash
python scripts/abide_dx_aal116_lcm_finetune.py \
  --roi_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL \
  --num_runs 1 \
  --base_seed 23 \
  --epochs 2 \
  --batch_size 8 \
  --out_dir outputs/abide_dx_aal116_lcm_finetune_smoke
```

Smoke tests must print:
- whether bb/head pretrained files exist
- number of ROI files found
- number of matched subjects
- label distribution
- model input shape
- extracted feature shape for probe
- train/val/test counts
- test ACC/F1

## 12. Quality checks

Fail loudly if:
- AAL ROI dir does not exist
- metadata CSV is missing
- pretrained bb/head files are missing
- fewer than 100 matched subjects are found
- ROI dimension is not 116
- only one class remains after matching
- train/val/test split loses a class
- finetune script is accidentally training from scratch instead of loading pretrained weights

Also save:

```text
README_abide_dx_aal116_lcm.md
```

Explain:
- how to run probe
- how to run finetune
- where outputs are
- label mapping
- exact pretrained weights used
- how features were extracted
- whether the finetune uses original repo logic or a new adapter

## 13. Do not do these

Do not implement any non-AAL atlas.
Do not download anything from the internet.
Do not delete or move existing repo files.
Do not silently train from scratch.
Do not replace full finetune with frozen-feature MLP.
Do not use sample-level split if subject IDs are available.
Do not make performance-changing model modifications unless required for code compatibility; if required, document clearly.

## 14. Final response expected from Codex

After implementation, please tell me:

1. What files you created/modified.
2. The exact command to run LP/MLP smoke test.
3. The exact command to run finetune smoke test.
4. The exact command to run full 5-run LP/MLP.
5. The exact command to run full 5-run finetune.
6. Whether `lcm_frozen` successfully loads the confirmed weights.
7. Whether finetune successfully loads the confirmed pretrained weights.
8. Where the results will be saved.
