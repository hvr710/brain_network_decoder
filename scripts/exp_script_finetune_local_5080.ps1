# Local Windows smoke tests for a 32 GB RTX 5080.
# This is additive: it does not replace the Linux/server scripts.

$RepoRoot = "D:\NCClab\LCM_fork_hvr710"
$WeightsDir = "$RepoRoot\pretrain_weights_fold0\none_decoder32_ppmi-abide-taowu-neurocon-hcpa-adni-hcpya_boldwin500_FCFC"

$AdniRoiRoot = "\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ADNI(ALL)\Pretraining_OUTPUT"
$AdniCsv = "$RepoRoot\our_data\ADNI.csv"

$PpmiRoiRoot = "\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ppmi_all"
$PpmiCsv = "$RepoRoot\our_data\PPMI.csv"

Set-Location $RepoRoot

# Conservative ADNI smoke test.
python finetune_lbnm.py `
  --dataset_name adni_our_2cls `
  --roi_root "$AdniRoiRoot" `
  --label_csv "$AdniCsv" `
  --decoder `
  --decoder_layer 32 `
  --cv_fold_n 2 `
  --fold_ids 0 `
  --checkpoint_fold 0 `
  --epochs 1 `
  --batch_size 1 `
  --device cuda:0 `
  --load_dname hcpa `
  --weights_dir "$WeightsDir"

# Conservative PPMI smoke test.
# Use the 2-class version for now because the current ROI export only contains 1 Control subject.
python finetune_lbnm.py `
  --dataset_name ppmi_our_2cls `
  --roi_root "$PpmiRoiRoot" `
  --label_csv "$PpmiCsv" `
  --decoder `
  --decoder_layer 32 `
  --cv_fold_n 2 `
  --fold_ids 0 `
  --checkpoint_fold 0 `
  --epochs 1 `
  --batch_size 1 `
  --device cuda:0 `
  --load_dname hcpa `
  --weights_dir "$WeightsDir"

# If the smoke test passes, try a slightly less tiny ADNI run like this:
# python finetune_lbnm.py `
#   --dataset_name adni_our_2cls `
#   --roi_root "$AdniRoiRoot" `
#   --label_csv "$AdniCsv" `
#   --decoder `
#   --decoder_layer 32 `
#   --cv_fold_n 5 `
#   --epochs 5 `
#   --batch_size 2 `
#   --device cuda:0 `
#   --load_dname hcpa `
#   --weights_dir "$WeightsDir"
