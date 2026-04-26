# Table3 下游重跑运行手册

## 1. 登录后先做这几步
```bash
ssh labserver
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder
```

## 2. 每次开始前，先手动改这两行
```bash
GPU_DEVICE=cuda:2
RUN_TAG=table3_all10_fold1_20260419_141859
```

说明：

- `GPU_DEVICE` 就是你这次想手动指定的卡，比如可以改成 `cuda:7`、`cuda:6`。
- `RUN_TAG` 不要只写时间戳，建议带上“这次跑了几个任务、是不是 fold1”这种信息，后面看输出目录会清楚很多。

推荐命名：

```bash
RUN_TAG=table3_all10_fold1_20260419_141859
RUN_TAG=table3_age3_fold1_20260419_141859
RUN_TAG=table3_cls7_fold1_20260419_141859
```

如果你要开多个 `tmux` 并行跑：

- 所有 tmux 里都要写**完全一样的** `RUN_TAG`
- 不然结果会被写到不同输出目录里，后面不好汇总

## 3. 正式开跑前先做轻检查
### 3.1 Python 语法检查
```bash
python -m py_compile finetune_table3.py launch_table3_queue.py summarize_table3.py
```

### 3.2 先检查 ADNI(MCI)
```bash
python finetune_table3.py \
  --config our_plan/table3_tasks.yaml \
  --task_id adni_mci \
  --fold 1 \
  --seed 4 \
  --device cpu \
  --audit_only \
  --output_root outputs/$RUN_TAG
```

### 3.3 再检查 ADNI(AD)
```bash
python finetune_table3.py \
  --config our_plan/table3_tasks.yaml \
  --task_id adni_ad \
  --fold 1 \
  --seed 4 \
  --device cpu \
  --audit_only \
  --output_root outputs/$RUN_TAG
```

看这两个文件是否正常：

```bash
cat outputs/$RUN_TAG/adni_mci/fold1/seed4/audit.json
cat outputs/$RUN_TAG/adni_ad/fold1/seed4/audit.json
```

重点看：

- train / val / test 的 sample 数和 subject 数是否正常
- 有没有大量 `missing_files`
- 有没有大量 `missing_labels`

## 4. 正式运行方案
### 方案 A：单卡串行跑全部 10 个任务

适合你只想占 1 张卡，稳稳跑完。

```bash
tmux new -s ds_all10
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder
GPU_DEVICE=cuda:2
RUN_TAG=table3_all10_fold1_20260419_141859

python launch_table3_queue.py \
  --config our_plan/table3_tasks.yaml \
  --task_ids all \
  --folds 1 \
  --seeds 4,44,444 \
  --device $GPU_DEVICE \
  --output_root outputs/$RUN_TAG
```

### 方案 B：多卡并行分组跑

适合你一次占 2 到 3 张卡，把 10 个任务拆开并行。

记住：

- 下面所有 tmux 都用同一个 `RUN_TAG`
- 只是 `--device`、`--batch_size` 和 `task_ids` 不同
- 每个 tmux 里面同一时刻只跑 1 个 job：`一个 task + fold1 + 一个 seed`，跑完才自动跑下一个，不会把同一组任务同时塞进一张卡。
- 保留 `--required_mem_mb`：如果目标卡空闲显存不够，队列会等待并每隔 `--poll_seconds` 重新检查，不会直接启动后 OOM 退出。
- 优先用“小物理 batch + 梯度累积”：比如 `--batch_size 4 --grad_accum_steps 8`，显存按 4 占，更新节奏接近有效 batch 32。
- 如果是共享卡，建议再加 `--reserve_cuda_mem_mb`：训练进程启动后会把这部分显存预占到自己的 PyTorch CUDA cache 里，后续训练可以复用这块 cache，别人也不容易在训练中途插进来抢显存。这个参数只在你确定这张卡可以被当前任务占用时使用。
- 多行命令里每行末尾的 `\` 必须是最后一个字符，后面不能有空格；不确定时直接用一行命令。

### 4.1 tmux 1：年龄回归 3 个任务
```bash
tmux new -s ds_rem_age
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder
RUN_TAG=table3_all10_onlyfold1_bs48_64_64_20260420_220000

python launch_table3_queue.py \
  --config our_plan/table3_tasks.yaml \
  --task_ids abide_age,nki_age,sald_age \
  --folds 1 \
  --seeds 4,44,444 \
  --device cuda:0 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --required_mem_mb 23000 \
  --reserve_cuda_mem_mb 22000 \
  --output_root outputs/$RUN_TAG
```

### 4.2 tmux 2：性别分类 3 个任务
```bash
tmux new -s ds_sex_bs64
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder
RUN_TAG=table3_all10_onlyfold1_bs48_64_64_20260420_220000

python launch_table3_queue.py \
  --config our_plan/table3_tasks.yaml \
  --task_ids abcd_sex,hcp_sex,bhrc_sex \
  --folds 1 \
  --seeds 4,44,444 \
  --device cuda:5 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --required_mem_mb 23000 \
  --reserve_cuda_mem_mb 22000 \
  --output_root outputs/$RUN_TAG
```

### 4.3 tmux 3：疾病 + 教育分类 4 个任务
```bash
tmux new -s ds_rem_cls
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder
RUN_TAG=table3_all10_onlyfold1_bs48_64_64_20260420_220000

python launch_table3_queue.py \
  --config our_plan/table3_tasks.yaml \
  --task_ids ppmi_pd_diagnosis,adni_mci,adni_ad,nki_education \
  --folds 1 \
  --seeds 4,44,444 \
  --device cuda:4 \
  --batch_size 2 \
  --grad_accum_steps 16 \
  --required_mem_mb 23000 \
  --reserve_cuda_mem_mb 22000 \
  --output_root outputs/$RUN_TAG
```
tmux attach -t ds_rem_cls
## 5. 运行过程中怎么看日志
### 5.1 看队列调度日志
```bash
find outputs/$RUN_TAG -name launcher_stdout.log | sort
tail -f outputs/$RUN_TAG/abide_age/fold1/seed4/launcher_stdout.log
```

### 5.2 看训练日志
```bash
tail -f outputs/$RUN_TAG/abide_age/fold1/seed4/train.log
tail -f outputs/$RUN_TAG/adni_mci/fold1/seed4/train.log
```

### 5.3 看运行状态
```bash
find outputs/$RUN_TAG -name run_state.json -exec grep -H '"state"' {} \; | sort
```

如果你想快速判断有没有失败：

```bash
find outputs/$RUN_TAG -name run_state.json -exec grep -H '"failed"' {} \;
```

## 6. 全部跑完后做汇总和 HTML
```bash
python summarize_table3.py \
  --config our_plan/table3_tasks.yaml \
  --paper_baselines our_plan/table3_paper_baselines.yaml \
  --output_root outputs/$RUN_TAG
```

## 7. 最后重点看这些结果文件
```bash
ls outputs/$RUN_TAG
cat outputs/$RUN_TAG/table3_style_summary.csv
cat outputs/$RUN_TAG/omni_alignment_summary.csv
```

生成的 HTML 文件有两个：

```bash
outputs/$RUN_TAG/table3_lcm_only.html
outputs/$RUN_TAG/table3_with_paper_baselines.html
```

含义分别是：

- `table3_lcm_only.html`
  只有你这次 LCM baseline 的一行结果，但布局仿照论文 Table 3

- `table3_with_paper_baselines.html`
  把论文里的 Brain-LM / Brain-JEPA / BrainMass / SwiFT / NeuroSTORM 一起排进去，再加上你这次的 LCM 行

## 8. 这次 Table 3 的对齐口径

年龄任务：

- 用 `mse_standardized`
- 用 `pearson_r`

分类任务：

- 用 `acc_percent`
- 用 `f1_weighted_percent`

另外这次还有两个固定约束：

- 以前那几个 3-fold 任务，现在都只跑 `fold1`
- seeds 还是 `4,44,444`

## 9. 只重跑下游里的 ADNI(MCI) 和 ADNI(AD)

说明：

- 这两个任务会直接读取 `our_plan/table3_tasks.yaml` 里的最新 ADNI 原始数据路径和 AAL split 路径。
- 如果只是修正 ADNI，建议换一个新的 `RUN_TAG`，不要和之前全量结果混在一起。

### 9.1 先做 audit

```bash
RUN_TAG=table3_adni_only_$(date +%Y%m%d_%H%M%S)

python finetune_table3.py \
  --config our_plan/table3_tasks.yaml \
  --task_id adni_mci \
  --fold 1 \
  --seed 4 \
  --device cpu \
  --audit_only \
  --output_root outputs/$RUN_TAG

python finetune_table3.py \
  --config our_plan/table3_tasks.yaml \
  --task_id adni_ad \
  --fold 1 \
  --seed 4 \
  --device cpu \
  --audit_only \
  --output_root outputs/$RUN_TAG
```

### 9.2 正式重跑这两个任务

```bash
tmux new -s ds_adni_only
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder
GPU_DEVICE=cuda:7
RUN_TAG=table3_adni_only_$(date +%Y%m%d_%H%M%S)

python launch_table3_queue.py \
  --config our_plan/table3_tasks.yaml \
  --task_ids adni_mci,adni_ad \
  --folds 1 \
  --seeds 4,44,444 \
  --device $GPU_DEVICE \
  --output_root outputs/$RUN_TAG
```
