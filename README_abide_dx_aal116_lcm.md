# ABIDE dx AAL116 LCM Probe + Finetune

本目录新增 ABIDE diagnosis / AAL116 的两条实验线：

- `LCM frozen features + LP/MLP probe`
- `LCM full finetune`

原始 `datasets.py`、`finetune_lbnm.py`、`models/` 没有修改。新增脚本通过 adapter 读取这次的 ABIDE AAL116 数据。

## 数据和标签

服务器默认路径：

```bash
ROI=/mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL
CSV=/mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/data_csv/ABIDE.csv
SPLIT=/mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL_crop_split
```

`SPLIT/train.txt`、`val.txt`、`test.txt` 里是 crop 文件路径。脚本只从文件名提取 subject id，例如 `0050642`，再映射到原始非 crop 的 AAL `.npy` 文件。

标签固定为：

```text
DX_GROUP = 0 -> ASD     -> 1
DX_GROUP = 3 -> Control -> 0
```

脚本会保存 `data_check.json`，其中包含 ROI 数量、CSV 行数、matched subjects、split 标签分布，以及 `DX_GROUP x DSM_IV_TR` 检查。

## 预训练权重

默认加载：

```text
model_weights/none_decoder32_ppmi-abide-taowu-neurocon-hcpa-adni-hcpya_boldwin500_FCFC/bb_fold0_hcpaBest_2025-01-20-02-09-00-551325.pt
model_weights/none_decoder32_ppmi-abide-taowu-neurocon-hcpa-adni-hcpya_boldwin500_FCFC/head_fold0_hcpaBest_2025-01-20-02-09-00-551325 .pt
```

注意第二个文件名在 `.pt` 前有一个空格。脚本启动时会打印两个文件是否存在，缺任意一个都会直接失败。

当前默认 ABIDE diagnosis token 是 `--diagnosis_token_ids 6 7`，分别对应 Control/ASD 的预训练 ABIDE token。这个值会写进每个 run 的 config。

## Probe

完整 LP/MLP：

```bash
bash scripts/run_abide_dx_aal116_lcm_probe.sh
```

等价 Python 命令：

```bash
python scripts/abide_dx_aal116_lcm_probe.py \
  --roi_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL \
  --label_csv /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/data_csv/ABIDE.csv \
  --split_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL_crop_split \
  --feature_source lcm_frozen \
  --mode both \
  --seeds 23 24 25 26 27 \
  --out_dir outputs/abide_dx_aal116_lcm_probe
```

Probe 逻辑：

- 先抽取并缓存全体 871 个 subject 的特征。
- train 始终固定使用 `train.txt` subjects。
- 原始 `val.txt + test.txt` 合并成 eval pool。
- 每个 seed `[23,24,25,26,27]` 对 eval pool 的 unique subject 做 subject-level half split。
- LP 只在固定 train features 上 fit。
- MLP 只用 train features 统计量标准化，并按 run-val loss 选择 best checkpoint。

特征缓存：

```text
outputs/abide_dx_aal116_lcm_probe/feature_cache/lcm_frozen_features.npz
outputs/abide_dx_aal116_lcm_probe/feature_cache/fc_vector_features.npz
```

默认复用 cache；需要重抽时加：

```bash
--recompute_features
```

轻量调试可以先跑 FC 向量 LP：

```bash
python scripts/abide_dx_aal116_lcm_probe.py \
  --roi_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL \
  --label_csv /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/data_csv/ABIDE.csv \
  --split_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL_crop_split \
  --feature_source fc_vector \
  --mode lp \
  --seeds 23 \
  --out_dir outputs/abide_dx_aal116_lcm_probe_fc_smoke
```

## Finetune

完整 3-seed finetune：

```bash
bash scripts/run_abide_dx_aal116_lcm_finetune.sh
```

等价 Python 命令：

```bash
python scripts/abide_dx_aal116_lcm_finetune.py \
  --roi_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL \
  --label_csv /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/data_csv/ABIDE.csv \
  --split_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL_crop_split \
  --seeds 4 44 444 \
  --epochs 50 \
  --batch_size 16 \
  --out_dir outputs/abide_dx_aal116_lcm_finetune
```

Smoke test：

```bash
python scripts/abide_dx_aal116_lcm_finetune.py \
  --roi_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL \
  --label_csv /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/data_csv/ABIDE.csv \
  --split_dir /mnt/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL_crop_split \
  --seeds 4 \
  --epochs 2 \
  --batch_size 8 \
  --out_dir outputs/abide_dx_aal116_lcm_finetune_smoke
```

Finetune 使用原 repo 的 LCM 结构：

- `Identity` backbone
- `BNDecoder(decoder_layer=32)`
- 输入为 `116 x 116` FC matrix
- `edge_index_fc = where(fc > 0.5)`
- backbone 和 decoder/head 全部 trainable
- 从确认的 bb/head 权重初始化，绝不静默从头训练

## HTML 表格

完整跑 probe + finetune + HTML：

```bash
bash scripts/run_abide_dx_aal116_all.sh
```

只重新渲染表格：

```bash
python scripts/render_abide_dx_aal116_tables.py \
  --probe_dir outputs/abide_dx_aal116_lcm_probe \
  --finetune_dir outputs/abide_dx_aal116_lcm_finetune \
  --feature_source lcm_frozen \
  --out_dir outputs/abide_dx_aal116_tables
```

输出：

```text
outputs/abide_dx_aal116_tables/summary_table.html
outputs/abide_dx_aal116_tables/per_run_table.html
```

表格用论文横线风格，汇总表对每个指标标红 best、下划线 second-best。

## 本地 UNC 调试示例

如果在 Windows/PowerShell 本地测试，把路径显式传成 UNC：

```powershell
python scripts/abide_dx_aal116_lcm_probe.py `
  --roi_dir "\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ABIDE\AAL" `
  --label_csv "\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\data_csv\ABIDE.csv" `
  --split_dir "\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ABIDE\AAL_crop_split" `
  --feature_source fc_vector `
  --mode lp `
  --seeds 23 `
  --out_dir outputs/abide_dx_aal116_lcm_probe_fc_smoke
```

本地 16GB 显存不建议跑完整 finetune。`lcm_frozen` 特征抽取可先用小 `--feature_batch_size 1` 验证权重加载和 feature shape。
## Volcano / vePFS

Volcano 涓婄殑浠撳簱銆佹暟鎹拰鐜璺緞锛?

```bash
REPO=/vePFS-0x0d/nzh/baseline/brain_network_decoder
ENV=/vePFS-0x0d/nzh/envs/lcm
ROI=/vePFS-0x0d/nzh/data/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL
CSV=/vePFS-0x0d/nzh/data/dataset1/ningzh/labels/age/ABIDE.csv
SPLIT=/vePFS-0x0d/nzh/data/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/split
```

宸叉柊澧?Volcano 涓撶敤 runner锛?

```bash
bash scripts/run_abide_dx_aal116_lcm_probe_volcano.sh
bash scripts/run_abide_dx_aal116_lcm_finetune_volcano.sh
bash scripts/run_abide_dx_aal116_all_volcano.sh
```

鑰冭檻鍒?A800 80GB 鏄惧瓨鏇村瑁曪紝榛樿閰嶇疆姣?mnt/4090 鍙堟縺杩涗竴浜涳細

- Probe 榛樿 `--feature_batch_size 8 --num_workers 8`
- Finetune 榛樿 `--batch_size 64 --num_workers 8`

濡傛灉浣犳兂鍦?tmux 閲岀洿鎺ヨ窇 Volcano finetune锛屽彲浠ョ敤锛?

```bash
cd /vePFS-0x0d/nzh/baseline/brain_network_decoder
source /vePFS-0x0d/nzh/envs/lcm/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/abide_dx_aal116_lcm_finetune.py \
  --roi_dir /vePFS-0x0d/nzh/data/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/AAL \
  --label_csv /vePFS-0x0d/nzh/data/dataset1/ningzh/labels/age/ABIDE.csv \
  --split_dir /vePFS-0x0d/nzh/data/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi/ABIDE/split \
  --seeds 4 44 444 \
  --epochs 50 \
  --batch_size 64 \
  --num_workers 8 \
  --device cuda:0 \
  --out_dir outputs/abide_dx_aal116_lcm_finetune
```
