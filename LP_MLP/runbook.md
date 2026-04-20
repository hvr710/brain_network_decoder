# LP/MLP Runbook

## 标准起手式

```bash
ssh labserver
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder
RUN_TAG=probe_$(date +%Y%m%d_%H%M%S)
```

## 1. tmux A：抽特征

```bash
tmux new -s probe_extract
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder
RUN_TAG=probe_$(date +%Y%m%d_%H%M%S)

python LP_MLP/launch_probe_queue.py \
  --stage extract \
  --config our_plan/table3_tasks.yaml \
  --probe_config LP_MLP/probe_defaults.yaml \
  --task_ids all \
  --preferred_gpus 2,7,6 \
  --device cuda \
  --output_root LP_MLP/outputs/$RUN_TAG
```

## 2. tmux B：MLP Probe

```bash
tmux new -s probe_mlp
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder

python LP_MLP/launch_probe_queue.py \
  --stage probe \
  --modes mlp_probe \
  --config our_plan/table3_tasks.yaml \
  --probe_config LP_MLP/probe_defaults.yaml \
  --task_ids all \
  --preferred_gpus 7,2,6 \
  --device cuda \
  --wait_for_cache \
  --output_root LP_MLP/outputs/$RUN_TAG
```

## 3. tmux C：Linear Probe

```bash
tmux new -s probe_lp
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder

python LP_MLP/launch_probe_queue.py \
  --stage probe \
  --modes linear_probe \
  --config our_plan/table3_tasks.yaml \
  --probe_config LP_MLP/probe_defaults.yaml \
  --task_ids all \
  --device cpu \
  --wait_for_cache \
  --output_root LP_MLP/outputs/$RUN_TAG
```

## 4. 看运行状态

```bash
find LP_MLP/outputs/$RUN_TAG -name run_state.json -exec grep -l '"state": "running"' {} + | sort
```

```bash
tail -f LP_MLP/outputs/$RUN_TAG/feature_cache/abide_age/base_split1/extract.log
```

```bash
tail -f LP_MLP/outputs/$RUN_TAG/mlp_probe/abide_age/base_split1/run_0_seed_23/train.log
```

```bash
tail -f LP_MLP/outputs/$RUN_TAG/linear_probe/abide_age/base_split1/run_0_seed_23/train.log
```

## 5. 汇总

```bash
python LP_MLP/summarize_probe_runs.py \
  --config our_plan/table3_tasks.yaml \
  --probe_config LP_MLP/probe_defaults.yaml \
  --output_root LP_MLP/outputs/$RUN_TAG
```

## 6. Smoke Test

```bash
python LP_MLP/launch_probe_queue.py \
  --stage all \
  --config our_plan/table3_tasks.yaml \
  --probe_config LP_MLP/probe_defaults.yaml \
  --task_ids abide_age,abcd_sex \
  --preferred_gpus 2,7,6 \
  --device cuda \
  --wait_for_cache \
  --smoke_test \
  --output_root LP_MLP/outputs/smoke_$(date +%Y%m%d_%H%M%S)
```

## 7. 只重跑 ADNI(MCI) 和 ADNI(AD)

说明：

- 这两条任务会直接读取 `our_plan/table3_tasks.yaml` 里的最新 ADNI 原始数据路径和 split 路径。
- 这次如果是为了修正 ADNI split，建议一定换一个新的 `RUN_TAG`，不要覆盖旧结果。

### 7.1 先抽这两个任务的特征

```bash
tmux new -s probe_adni_extract
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder
RUN_TAG=probe_adni_fix_$(date +%Y%m%d_%H%M%S)

python LP_MLP/launch_probe_queue.py \
  --stage extract \
  --config our_plan/table3_tasks.yaml \
  --probe_config LP_MLP/probe_defaults.yaml \
  --task_ids adni_mci,adni_ad \
  --device cuda:7 \
  --output_root LP_MLP/outputs/$RUN_TAG
```

### 7.2 跑这两个任务的 MLP Probe

```bash
tmux new -s probe_adni_mlp
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder

python LP_MLP/launch_probe_queue.py \
  --stage probe \
  --modes mlp_probe \
  --config our_plan/table3_tasks.yaml \
  --probe_config LP_MLP/probe_defaults.yaml \
  --task_ids adni_mci,adni_ad \
  --device cuda:6 \
  --wait_for_cache \
  --output_root LP_MLP/outputs/$RUN_TAG
```

### 7.3 跑这两个任务的 Linear Probe

```bash
tmux new -s probe_adni_lp
conda activate /data/ningzh/LCM/envs/lcm
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder

python LP_MLP/launch_probe_queue.py \
  --stage probe \
  --modes linear_probe \
  --config our_plan/table3_tasks.yaml \
  --probe_config LP_MLP/probe_defaults.yaml \
  --task_ids adni_mci,adni_ad \
  --device cpu \
  --wait_for_cache \
  --output_root LP_MLP/outputs/$RUN_TAG
```
