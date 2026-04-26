# HuoShan2 A800 Table3 第一次复跑命令

本文件是在 HuoShan2/A800 上执行的完整 runbook。目标是把 repo、Table3 数据、labels、splits、预训练权重复制到 `/vePFS-0x0d/nzh` 本地目录，然后用 3 个 tmux 跑完 10 个 Table3 任务。

固定约定：

- 工作根目录：`/vePFS-0x0d/nzh`
- repo：`/vePFS-0x0d/nzh/repo/brain_network_decoder`
- conda env：`/vePFS-0x0d/nzh/envs/lcm` 和 `/vePFS-0x0d/nzh/envs/bsem`
- conda 包缓存：`/vePFS-0x0d/nzh/conda_pkgs`
- 数据副本：`/vePFS-0x0d/nzh/data/dataset1`、`/vePFS-0x0d/nzh/data/dataset4`
- 预训练权重：`/vePFS-0x0d/nzh/repo/brain_network_decoder/pretrain_weights_fold0`
- 任务：全部 10 个 Table3 任务，只跑 `fold1`
- seeds：`4,44,444`
- tmux 名称：`age_1`、`sex_1`、`disease_1`
- 输出目录：`outputs/one_$RUN_TS/<group>_$RUN_TS/<task_id>_$RUN_TS/fold1/seed{4,44,444}/`
- 每个 seed 的 `best_metrics.json`、`test_predictions.csv`、`epoch_history.csv`、checkpoint 都保留在各自 seed 目录里，不只保留均值方差。

## 0. 最省心执行方式：第 1 步在源服务器推数据，第 2-4 步在 HuoShan 跑

关键纠正：

- HuoShan 自己通常**看不到** `dataset1/dataset4` 这些 NAS 路径。
- 所以第 1 步不能在 HuoShan 上直接 `rsync /mnt/dataset4 ...`。
- 正确做法是：
  - 先在 **HuoShan** 上从 GitHub 拉 repo。
  - 再到 **能看到 `/mnt/dataset1`、`/mnt/dataset4`、`/mnt/dataset3` 的源/NCC 服务器** 上，把数据通过 SSH/rsync 推到 HuoShan。
  - 第 2-4 步再回到 HuoShan 执行。

### 0.0 先在 HuoShan 上拿 repo

```bash
mkdir -p /vePFS-0x0d/nzh/repo
cd /vePFS-0x0d/nzh/repo

git clone --depth 1 --filter=blob:none --no-checkout \
  -b omni_table3_10ds \
  https://github.com/hvr710/brain_network_decoder.git

cd brain_network_decoder
git sparse-checkout init --no-cone
cat > .git/info/sparse-checkout <<'EOF'
/*
!outputs/
!LP_MLP/outputs/
!data_old_outputs/
!pretrain_weights_fold0/
EOF
git checkout omni_table3_10ds

chmod +x scripts/a800_*.sh scripts/bootstrap_a800_envs.sh
```

### 0.1 在源/NCC 服务器上，把 NAS 数据推到 HuoShan

下面这一步要在**能看到 `/mnt/dataset1` 和 `/mnt/dataset4` 的那台源服务器**上执行，不是在 HuoShan 上执行。

进入同一个 repo 后，先按实际情况设置 HuoShan SSH 地址：

```bash
cd /mnt/dataset3/nzh/lcm_ds/brain_network_decoder
export HUOSHAN_HOST=root@di-20260417182420-kz6q9
```

如果源路径不是默认的 `/mnt/dataset1`、`/mnt/dataset4`、`/mnt/dataset3/nzh/lcm_ds/brain_network_decoder`，先改：

```bash
export SRC_REPO=/mnt/dataset3/nzh/lcm_ds/brain_network_decoder
export SRC_PRETRAIN=/mnt/dataset3/nzh/lcm_ds/brain_network_decoder/pretrain_weights_fold0
export SRC_DATASET1=/mnt/dataset1
export SRC_DATASET4=/mnt/dataset4
```

### 0.1 复制 repo、权重、数据、label、split

```bash
bash scripts/a800_run_step_in_tmux.sh prep_1 scripts/a800_01_push_nas_to_huoshan.sh
```

重新连回复制 tmux：

```bash
tmux attach -t prep_1
```

这一步完成后，回到 HuoShan，后续都用火山本地副本。先恢复变量：

```bash
source /vePFS-0x0d/nzh/run_table3_one.env
cd $REPO_DST
```

### 0.2 创建 lcm 和 bsem 环境

```bash
bash scripts/a800_run_step_in_tmux.sh env_1 scripts/a800_02_create_envs.sh
```

重新连回环境 tmux：

```bash
tmux attach -t env_1
```

### 0.3 正式训练前 audit

```bash
bash scripts/a800_run_step_in_tmux.sh audit_1 scripts/a800_03_audit.sh
```

重新连回 audit tmux：

```bash
tmux attach -t audit_1
```

### 0.4 启动三个 tmux 正式跑

默认用 `cuda:0/1/2`：

```bash
bash scripts/a800_run_step_in_tmux.sh launch_1 scripts/a800_04_launch_tmux.sh
```

重新连回 launch tmux：

```bash
tmux attach -t launch_1
```

如果要手动指定 3 张卡：

```bash
AGE_GPU=cuda:0 SEX_GPU=cuda:1 DISEASE_GPU=cuda:2 bash scripts/a800_run_step_in_tmux.sh launch_1 scripts/a800_04_launch_tmux.sh
```

重新连回 tmux：

```bash
tmux attach -t age_1
tmux attach -t sex_1
tmux attach -t disease_1
```

如果首个正式任务 OOM，用更小 batch 重新启动第 4 步：

```bash
BATCH_SIZE=32 GRAD_ACCUM_STEPS=1 bash scripts/a800_run_step_in_tmux.sh launch_1 scripts/a800_04_launch_tmux.sh
```

四个步骤对应的 tmux 名称：

```bash
tmux attach -t prep_1
tmux attach -t env_1
tmux attach -t audit_1
tmux attach -t launch_1
```

如果某一步 tmux 已经结束但 session 还占着同名窗口，可以先看一眼日志，确认没问题后关闭：

```bash
tmux attach -t prep_1
# 确认结束后在 tmux 里输入 exit
```

## 1. 准备目录和变量

先在 HuoShan2 上执行：

```bash
export A800_ROOT=/vePFS-0x0d/nzh
export REPO_DST=$A800_ROOT/repo/brain_network_decoder
export DATA_DST=$A800_ROOT/data
export RUN_TS=$(date +%Y%m%d_%H%M%S)
export ONE_ROOT=outputs/one_$RUN_TS

mkdir -p \
  $A800_ROOT/repo \
  $A800_ROOT/envs \
  $A800_ROOT/conda_pkgs \
  $DATA_DST/dataset1 \
  $DATA_DST/dataset3 \
  $DATA_DST/dataset4
```

把本次运行变量保存下来，三个 tmux 都会读取同一个时间戳：

```bash
cat > $A800_ROOT/run_table3_one.env <<EOF
export A800_ROOT=$A800_ROOT
export REPO_DST=$REPO_DST
export DATA_DST=$DATA_DST
export RUN_TS=$RUN_TS
export ONE_ROOT=$ONE_ROOT
export TABLE3_DATASET1_ROOT=$DATA_DST/dataset1
export TABLE3_DATASET3_ROOT=$DATA_DST/dataset3
export TABLE3_DATASET4_ROOT=$DATA_DST/dataset4
EOF
```

重新登录后恢复变量：

```bash
source /vePFS-0x0d/nzh/run_table3_one.env
```

## 2. 复制 repo

如果 HuoShan2 能看到当前源 repo，直接用下面命令。只复制代码和配置，不复制旧 outputs。

```bash
export SRC_REPO=/mnt/dataset3/nzh/lcm_ds/brain_network_decoder

rsync -a --info=progress2 \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'outputs' \
  --exclude 'data_old_outputs' \
  --exclude 'LP_MLP/outputs' \
  --exclude 'pretrain_weights_fold0' \
  $SRC_REPO/ $REPO_DST/
```

如果 HuoShan2 上不能直接看到 `$SRC_REPO`，先把 repo 用 `scp` 或 GitHub 放到 `$REPO_DST`，最终目录必须是：

```bash
ls $REPO_DST/finetune_table3.py
ls $REPO_DST/our_plan/table3_tasks.yaml
```

## 3. 复制预训练权重

```bash
export SRC_PRETRAIN=/mnt/dataset3/nzh/lcm_ds/brain_network_decoder/pretrain_weights_fold0

mkdir -p $REPO_DST/pretrain_weights_fold0
rsync -a --info=progress2 $SRC_PRETRAIN/ $REPO_DST/pretrain_weights_fold0/
```

检查：

```bash
find $REPO_DST/pretrain_weights_fold0 -type f -name '*.pt' | sort
du -sh $REPO_DST/pretrain_weights_fold0
```

## 4. 复制 Table3 数据、labels、splits

下面的源路径默认是 `/mnt/dataset1`、`/mnt/dataset4`。如果 HuoShan2 上源路径不同，只改这两个变量：

```bash
export SRC_DATASET1=/mnt/dataset1
export SRC_DATASET4=/mnt/dataset4

export SRC_ROI=$SRC_DATASET4/DATASETS/fmri_pretraining/fmri_dataset/roi
export DST_ROI=$DATA_DST/dataset4/DATASETS/fmri_pretraining/fmri_dataset/roi
export SRC_LABEL=$SRC_DATASET1/ningzh/labels
export DST_LABEL=$DATA_DST/dataset1/ningzh/labels

mkdir -p $DST_ROI $DST_LABEL
```

复制 dataset4 的 `.npy` 和 split：

```bash
rsync -a --info=progress2 $SRC_ROI/ABIDE/AAL/ $DST_ROI/ABIDE/AAL/
rsync -a --info=progress2 $SRC_ROI/ABIDE/Schaefer2018_100_crop_split/ $DST_ROI/ABIDE/Schaefer2018_100_crop_split/

rsync -a --info=progress2 $SRC_ROI/NKI/AAL/ $DST_ROI/NKI/AAL/
rsync -a --info=progress2 $SRC_ROI/NKI/100ROI_split/ $DST_ROI/NKI/100ROI_split/

rsync -a --info=progress2 $SRC_ROI/SALD/AAL/ $DST_ROI/SALD/AAL/
rsync -a --info=progress2 $SRC_ROI/SALD/100ROI_split/ $DST_ROI/SALD/100ROI_split/

rsync -a --info=progress2 $SRC_ROI/ABCD/AAL/ $DST_ROI/ABCD/AAL/
rsync -a --info=progress2 $SRC_ROI/HCP/AAL/ $DST_ROI/HCP/AAL/

rsync -a --info=progress2 $SRC_ROI/BHRC/AAL/ $DST_ROI/BHRC/AAL/
rsync -a --info=progress2 $SRC_ROI/BHRC/100ROI_split/ $DST_ROI/BHRC/100ROI_split/

rsync -a --info=progress2 $SRC_ROI/PPMI/AAL/ $DST_ROI/PPMI/AAL/
rsync -a --info=progress2 $SRC_ROI/PPMI/100ROI/ $DST_ROI/PPMI/100ROI/

mkdir -p $DST_ROI/ADNI/AAL
rsync -a --info=progress2 $SRC_ROI/ADNI/AAL/CN/ $DST_ROI/ADNI/AAL/CN/
rsync -a --info=progress2 $SRC_ROI/ADNI/AAL/MCI/ $DST_ROI/ADNI/AAL/MCI/
rsync -a --info=progress2 $SRC_ROI/ADNI/AAL/AD/ $DST_ROI/ADNI/AAL/AD/
rsync -a --info=progress2 $SRC_ROI/ADNI/AAL/MCI_train_abs_AAL.txt $DST_ROI/ADNI/AAL/
rsync -a --info=progress2 $SRC_ROI/ADNI/AAL/MCI_val_abs_AAL.txt $DST_ROI/ADNI/AAL/
rsync -a --info=progress2 $SRC_ROI/ADNI/AAL/MCI_test_abs_AAL.txt $DST_ROI/ADNI/AAL/
rsync -a --info=progress2 $SRC_ROI/ADNI/AAL/AD_train_abs_AAL.txt $DST_ROI/ADNI/AAL/
rsync -a --info=progress2 $SRC_ROI/ADNI/AAL/AD_val_abs_AAL.txt $DST_ROI/ADNI/AAL/
rsync -a --info=progress2 $SRC_ROI/ADNI/AAL/AD_test_abs_AAL.txt $DST_ROI/ADNI/AAL/
```

复制 dataset1 labels：

```bash
mkdir -p $DST_LABEL/age $DST_LABEL/sex $DST_LABEL/disease $DST_LABEL/education

rsync -a --info=progress2 $SRC_LABEL/age/ABIDE.csv $DST_LABEL/age/
rsync -a --info=progress2 $SRC_LABEL/age/NKI.csv $DST_LABEL/age/
rsync -a --info=progress2 $SRC_LABEL/age/SALD.csv $DST_LABEL/age/

rsync -a --info=progress2 $SRC_LABEL/sex/ABCD.csv $DST_LABEL/sex/
rsync -a --info=progress2 $SRC_LABEL/sex/HCP.csv $DST_LABEL/sex/
rsync -a --info=progress2 $SRC_LABEL/sex/BHRC.csv $DST_LABEL/sex/

rsync -a --info=progress2 $SRC_LABEL/disease/PPMI.csv $DST_LABEL/disease/
rsync -a --info=progress2 $SRC_LABEL/disease/adni_list.xlsx $DST_LABEL/disease/

rsync -a --info=progress2 $SRC_LABEL/education/NKI.csv $DST_LABEL/education/
```

复制完后检查大小：

```bash
du -sh $DATA_DST/dataset1 $DATA_DST/dataset4 $REPO_DST/pretrain_weights_fold0
```

检查 `.npy` 数量：

```bash
for d in \
  ABIDE/AAL NKI/AAL SALD/AAL ABCD/AAL HCP/AAL BHRC/AAL PPMI/AAL \
  ADNI/AAL/CN ADNI/AAL/MCI ADNI/AAL/AD
do
  echo "$d"
  echo -n "  source: "; find $SRC_ROI/$d -type f -name '*.npy' | wc -l
  echo -n "  local : "; find $DST_ROI/$d -type f -name '*.npy' | wc -l
done
```

## 5. 创建 lcm 和 bsem 环境

进入 repo 后执行 bootstrap：

```bash
source /vePFS-0x0d/nzh/run_table3_one.env
cd $REPO_DST

bash scripts/bootstrap_a800_envs.sh
```

检查 lcm 环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /vePFS-0x0d/nzh/miniconda3/etc/profile.d/conda.sh
conda activate /vePFS-0x0d/nzh/envs/lcm

python -c "import torch, torch_geometric, torch_scatter; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -m py_compile finetune_table3.py table3_dataset.py table3_utils.py launch_table3_queue.py summarize_table3.py
```

## 6. 跑正式任务前先 audit

```bash
source /vePFS-0x0d/nzh/run_table3_one.env
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /vePFS-0x0d/nzh/miniconda3/etc/profile.d/conda.sh
conda activate /vePFS-0x0d/nzh/envs/lcm
cd $REPO_DST

python launch_table3_queue.py \
  --config our_plan/table3_tasks.yaml \
  --task_ids all \
  --folds 1 \
  --seeds 4 \
  --device cuda:0 \
  --audit_only \
  --output_root $ONE_ROOT/audit_$RUN_TS \
  --task_dir_suffix $RUN_TS
```

重点看：

```bash
find $ONE_ROOT/audit_$RUN_TS -name audit.json | sort
find $ONE_ROOT/audit_$RUN_TS -name train.log -exec grep -H "Split summary\\|Audit warning" {} \; | sort
```

ADNI(MCI/AD) 的 `missing_files` 和 `missing_labels` 应该都是 0。

## 7. 三个 tmux 正式启动命令

启动前先确认 GPU：

```bash
nvidia-smi
```

下面示例占用 `cuda:0`、`cuda:1`、`cuda:2`，第四张卡留给别人。实际 GPU 编号可手动改。

### 7.1 age_1

```bash
tmux new -s age_1
```

进入 tmux 后执行：

```bash
source /vePFS-0x0d/nzh/run_table3_one.env
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /vePFS-0x0d/nzh/miniconda3/etc/profile.d/conda.sh
conda activate /vePFS-0x0d/nzh/envs/lcm
cd $REPO_DST

python launch_table3_queue.py \
  --config our_plan/table3_tasks.yaml \
  --task_ids abide_age,nki_age,sald_age \
  --folds 1 \
  --seeds 4,44,444 \
  --device cuda:0 \
  --batch_size 64 \
  --grad_accum_steps 1 \
  --output_root $ONE_ROOT/age_$RUN_TS \
  --task_dir_suffix $RUN_TS
```

### 7.2 sex_1

```bash
tmux new -s sex_1
```

进入 tmux 后执行：

```bash
source /vePFS-0x0d/nzh/run_table3_one.env
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /vePFS-0x0d/nzh/miniconda3/etc/profile.d/conda.sh
conda activate /vePFS-0x0d/nzh/envs/lcm
cd $REPO_DST

python launch_table3_queue.py \
  --config our_plan/table3_tasks.yaml \
  --task_ids abcd_sex,hcp_sex,bhrc_sex \
  --folds 1 \
  --seeds 4,44,444 \
  --device cuda:1 \
  --batch_size 64 \
  --grad_accum_steps 1 \
  --output_root $ONE_ROOT/sex_$RUN_TS \
  --task_dir_suffix $RUN_TS
```

### 7.3 disease_1

```bash
tmux new -s disease_1
```

进入 tmux 后执行：

```bash
source /vePFS-0x0d/nzh/run_table3_one.env
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /vePFS-0x0d/nzh/miniconda3/etc/profile.d/conda.sh
conda activate /vePFS-0x0d/nzh/envs/lcm
cd $REPO_DST

python launch_table3_queue.py \
  --config our_plan/table3_tasks.yaml \
  --task_ids ppmi_pd_diagnosis,adni_mci,adni_ad,nki_education \
  --folds 1 \
  --seeds 4,44,444 \
  --device cuda:2 \
  --batch_size 64 \
  --grad_accum_steps 1 \
  --output_root $ONE_ROOT/disease_$RUN_TS \
  --task_dir_suffix $RUN_TS
```

如果首个正式 job OOM，把对应 tmux 停掉后改成：

```bash
--batch_size 32 \
--grad_accum_steps 1 \
```

## 8. 重新连接 tmux

重新连回三个任务：

```bash
tmux attach -t age_1
tmux attach -t sex_1
tmux attach -t disease_1
```

在 tmux 里临时离开但不停止任务：按 `Ctrl-b`，再按 `d`。

查看 tmux 列表：

```bash
tmux ls
```

## 9. 监控运行状态

查看 GPU：

```bash
nvidia-smi
```

查看队列日志：

```bash
source /vePFS-0x0d/nzh/run_table3_one.env

find $ONE_ROOT -name launcher_stdout.log | sort
tail -f $ONE_ROOT/age_$RUN_TS/abide_age_$RUN_TS/fold1/seed4/launcher_stdout.log
```

查看训练日志：

```bash
tail -f $ONE_ROOT/age_$RUN_TS/abide_age_$RUN_TS/fold1/seed4/train.log
tail -f $ONE_ROOT/disease_$RUN_TS/adni_mci_$RUN_TS/fold1/seed4/train.log
```

查看 run state：

```bash
find $ONE_ROOT -name run_state.json -exec grep -H '"state"' {} \; | sort
find $ONE_ROOT -name run_state.json -exec grep -H '"failed"' {} \; | sort
```

确认每个任务都有 3 个 seed：

```bash
for task_dir in $(find $ONE_ROOT -maxdepth 3 -type d -name '*_*' | sort); do
  if [[ -d "$task_dir/fold1" ]]; then
    echo "$task_dir"
    find "$task_dir/fold1" -maxdepth 1 -type d -name 'seed*' | sort
  fi
done
```

## 10. 汇总结果

三个 tmux 都跑完后，在任意 shell 执行：

```bash
source /vePFS-0x0d/nzh/run_table3_one.env
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /vePFS-0x0d/nzh/miniconda3/etc/profile.d/conda.sh
conda activate /vePFS-0x0d/nzh/envs/lcm
cd $REPO_DST

python summarize_table3.py \
  --config our_plan/table3_tasks.yaml \
  --paper_baselines our_plan/table3_paper_baselines.yaml \
  --output_root $ONE_ROOT
```

汇总产物：

```bash
ls $ONE_ROOT
cat $ONE_ROOT/task_summary.csv
cat $ONE_ROOT/table3_style_summary.csv
cat $ONE_ROOT/omni_alignment_summary.csv
```

HTML：

```bash
$ONE_ROOT/table3_lcm_only.html
$ONE_ROOT/table3_with_paper_baselines.html
```

每个 seed 的单独结果仍在：

```bash
find $ONE_ROOT -path '*/fold1/seed*/best_metrics.json' | sort
find $ONE_ROOT -path '*/fold1/seed*/test_predictions.csv' | sort
find $ONE_ROOT -path '*/fold1/seed*/epoch_history.csv' | sort
```

## 11. 本次输出目录示例

实际目录会长这样：

```text
outputs/one_20260427_210000/
  age_20260427_210000/
    abide_age_20260427_210000/fold1/seed4/
    abide_age_20260427_210000/fold1/seed44/
    abide_age_20260427_210000/fold1/seed444/
    nki_age_20260427_210000/fold1/seed4/
    sald_age_20260427_210000/fold1/seed4/
  sex_20260427_210000/
    abcd_sex_20260427_210000/fold1/seed4/
    hcp_sex_20260427_210000/fold1/seed4/
    bhrc_sex_20260427_210000/fold1/seed4/
  disease_20260427_210000/
    ppmi_pd_diagnosis_20260427_210000/fold1/seed4/
    adni_mci_20260427_210000/fold1/seed4/
    adni_ad_20260427_210000/fold1/seed4/
    nki_education_20260427_210000/fold1/seed4/
```

## 12. 结束后不要自动删数据

本次复制到 `/vePFS-0x0d/nzh/data` 的副本先保留。确认所有结果交付完成后，再由你手动清理。
