# LCM Table 3 下游微调交付说明

更新时间：2026-04-27

## 1. 执行口径

本次只复用 LCM repo 的模型框架和预训练权重，用来跑 Omni-fMRI Table 3 对齐的 10 个下游任务。数据、split、指标都按 Table 3 下游任务重新组织，不再使用原始 LCM 论文里的多 fold split 逻辑。

固定约束如下：

- atlas 统一使用 `AAL_116`。原因是当前使用的 LCM 预训练权重是在 AAL atlas 下训练的，下游微调输入必须和预训练 atlas 对齐。
- 预训练权重使用 repo 内配置的 `pretrain_weights_fold0/none_decoder32_ppmi-abide-taowu-neurocon-hcpa-adni-hcpya_boldwin500_FCFC`，对应你本地整理的 `D:\NCClab\LCM_omni_table3_10ds\pretrain_weights_fold0`。
- 所有任务只跑固定 split 的 `fold1`，包括 NKI、SALD、BHRC。虽然它们有 3 套现成 split，但这里不做 3-fold CV，避免把“split 方差”和“随机 seed 方差”混在一起，也和 Table 3 这种固定下游 benchmark 口径更一致。
- 每个任务跑 3 个随机种子：`4, 44, 444`。最终汇总按这 3 个 run 取均值和标准差。
- 回归任务报告 Table 3 口径的 `MSE` 和 `R`：实现里保存 `mse_raw`、`rmse`、`mse_standardized`、`pearson_r`，Table 3 汇总使用 `mse_standardized` 和 `pearson_r`。
- 分类任务报告 Table 3 口径的 `ACC` 和 `F1`：实现里保存 `acc_percent` 和 `f1_weighted_percent`，Table 3 汇总使用这两个百分比指标。

## 2. 代码入口

主要文件：

- `our_plan/table3_tasks.yaml`：10 个任务的唯一配置源，包括数据路径、label 路径、split 路径、任务类型、fold 和 seeds。
- `table3_dataset.py`：按任务配置读取固定 split，解析 subject/file id，join label，并生成 audit。
- `finetune_table3.py`：单个 `task_id + fold + seed` 的微调入口，加载预训练 backbone/head，训练、验证、测试并保存结果。
- `launch_table3_queue.py`：正式批量运行入口，按任务、fold、seed 展开队列。
- `summarize_table3.py`：全部 run 完成后汇总 CSV 和 Table 3 风格 HTML。

路径兼容：

- Windows UNC 路径如 `\\10.20.33.82\dataset4\...`
- 服务器挂载路径如 `/mnt/dataset4/...`

`table3_utils.resolve_path()` 会在两套路径之间自动尝试映射。

## 3. 任务与 split 统计

下面统计来自 `audit_only`。格式为 `samples/subjects`。

| task_id | Table 3 任务 | sample 口径 | split 来源 | train | val | test | 合计 | audit 备注 |
|---|---|---:|---|---:|---:|---:|---:|---|
| `abide_age` | Age Regression | subject | crop txt 去重到 subject | 522/522 | 174/174 | 175/175 | 871 | ABIDE crop split 原始行数为 1223/365/395，去重后为 522/174/175，合计 871，对上要求 |
| `nki_age` | Age Regression | subject | `100ROI_split/*1.txt` | 501/501 | 71/71 | 145/145 | 717 | 无 missing file/label |
| `sald_age` | Age Regression | subject | `100ROI_split/*1.txt` | 345/345 | 49/49 | 99/99 | 493 | 无 missing file/label |
| `abcd_sex` | Sex Classif. | file | AAL 下 train/val/test 文件夹 | 699/350 | 100/50 | 200/100 | 999 | test 文件夹有 201 个 `.npy`，其中 `first200_frames_000050-000249.npy` 无法解析 subject，所以有效样本 200 |
| `hcp_sex` | Sex Classif. | file | AAL 下 train/val/test 文件夹 | 3611/602 | 1205/201 | 1181/197 | 5997 | 有 24/6/30 个 crop 文件找不到 label；有效 subject 合计 1000 |
| `bhrc_sex` | Sex Classif. | subject | `100ROI_split/*1.txt` | 325/325 | 46/46 | 94/94 | 465 | 无 missing file/label |
| `ppmi_pd_diagnosis` | Diagnosis | subject | 参考 `PPMI/100ROI/{train,val,test}` 文件夹 | 331/331 | 47/47 | 96/96 | 474 | PPMI split subject 合计 474，对上要求 |
| `adni_mci` | Diagnosis | subject | AAL 下 `MCI_*_abs_AAL.txt` | 176/176 | 58/58 | 58/58 | 292 | CN/MCI 二分类；split 和 label join 后无缺失，类别为 train 88/88、val 29/29、test 29/29 |
| `adni_ad` | Diagnosis | subject | AAL 下 `AD_*_abs_AAL.txt` | 94/94 | 32/32 | 31/31 | 157 | CN/AD 二分类；split 和 label join 后无缺失，类别为 train 69/25、val 23/9、test 23/8 |
| `nki_education` | Education Classif. | subject | NKI `100ROI_split/*1.txt` | 225/225 | 28/28 | 78/78 | 331 | NKI split 原本是 717 subject，但只有 331 个 subject 有 `education_group` label；missing label 为 276/43/67 |

重点结论：

- ABIDE 支线任务已完成：crop txt 中的重复 crop 已归并到 subject，train/val/test subject 数为 `522/174/175`，合计 `871`。
- PPMI 支线任务已完成：从 `PPMI/100ROI` 三个 split 文件夹抽取 subject 后，train/val/test 为 `331/47/96`，合计 `474`。
- ABCD、HCP 是 file-level 任务，subject 数小于 sample 数是预期现象，因为一个 subject 会有多个 run/window/crop。
- NKI education 的有效样本数低于 NKI age 是因为 education label 缺失，不是时间序列文件缺失。

## 4. 正式运行命令

单卡串行跑全部 10 个任务：

```bash
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder

GPU_DEVICE=cuda:2
RUN_TAG=table3_all10_fold1_seed4_44_444_$(date +%Y%m%d_%H%M%S)

python launch_table3_queue.py \
  --config our_plan/table3_tasks.yaml \
  --task_ids all \
  --folds 1 \
  --seeds 4,44,444 \
  --device $GPU_DEVICE \
  --output_root outputs/$RUN_TAG
```

如果显存紧张，推荐用小物理 batch 和梯度累积，例如：

```bash
python launch_table3_queue.py \
  --config our_plan/table3_tasks.yaml \
  --task_ids all \
  --folds 1 \
  --seeds 4,44,444 \
  --device cuda:2 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --required_mem_mb 23000 \
  --reserve_cuda_mem_mb 22000 \
  --output_root outputs/$RUN_TAG
```

跑完后汇总：

```bash
python summarize_table3.py \
  --config our_plan/table3_tasks.yaml \
  --paper_baselines our_plan/table3_paper_baselines.yaml \
  --output_root outputs/$RUN_TAG
```

会生成：

- `task_summary.csv`
- `table3_style_summary.csv`
- `omni_alignment_summary.csv`
- `table3_lcm_only.html`
- `table3_with_paper_baselines.html`

## 5. 交付检查点

正式跑之前至少检查：

```bash
python -m py_compile finetune_table3.py table3_dataset.py table3_utils.py launch_table3_queue.py summarize_table3.py
```

建议先 audit 关键任务：

```bash
python finetune_table3.py \
  --config our_plan/table3_tasks.yaml \
  --task_id abide_age \
  --fold 1 \
  --seed 4 \
  --device cpu \
  --audit_only \
  --output_root outputs/$RUN_TAG

python finetune_table3.py \
  --config our_plan/table3_tasks.yaml \
  --task_id ppmi_pd_diagnosis \
  --fold 1 \
  --seed 4 \
  --device cpu \
  --audit_only \
  --output_root outputs/$RUN_TAG
```

audit 里重点看：

- `split_counts` 是否非空，且 train/val/test 都存在。
- `disjoint_subjects` 是否为 `true`。
- `split_audit.missing_files` 是否为 0。
- `split_audit.missing_labels` 是否符合预期。当前已知 HCP 和 NKI education 会有 label 缺失，ABCD test 有 1 个不可解析文件名。

## 6. 当前遗留备注

- `table3_plan.md` 是原始需求记录，里面有一些问号和旧假设；真正执行以 `our_plan/table3_tasks.yaml` 为准。
- `old_3fold不用.md` 只保留历史 3-fold 方案，不参与本次正式跑。
- ADNI 的有效样本数以 AAL split txt 和 label filter 后的 audit 为准，不再沿用旧备注里的“三组和 497”。
- 如果后续要改成 NKI/SALD/BHRC 真 3-fold，需要同时改 `fold_ids`、汇总 expected runs 和 runbook；当前交付版本明确只跑 `fold1`。
