@echo off
setlocal

REM Local Windows CMD smoke tests for a 32 GB RTX 5080.
REM This is additive: it does not replace the Linux/server scripts.
REM Use 2-way CV for a valid split, but only run fold0 because the local debug weights currently only include fold0.

set REPO_ROOT=D:\NCClab\LCM_fork_hvr710
set WEIGHTS_DIR=%REPO_ROOT%\pretrain_weights_fold0\none_decoder32_ppmi-abide-taowu-neurocon-hcpa-adni-hcpya_boldwin500_FCFC

set ADNI_ROI_ROOT=\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ADNI(ALL)\Pretraining_OUTPUT
set ADNI_CSV=%REPO_ROOT%\our_data\ADNI.csv

set PPMI_ROI_ROOT=\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\ppmi_all
set PPMI_CSV=%REPO_ROOT%\our_data\PPMI.csv

cd /d %REPO_ROOT%

REM Quick dependency check.
python -c "import torch; import torch_geometric; print('torch ok'); print('torch_geometric ok')" || goto :fail

REM Conservative ADNI smoke test.
python finetune_lbnm.py --dataset_name adni_our_2cls --roi_root "%ADNI_ROI_ROOT%" --label_csv "%ADNI_CSV%" --decoder --decoder_layer 32 --cv_fold_n 2 --fold_ids 0 --checkpoint_fold 0 --epochs 1 --batch_size 1 --device cuda:0 --load_dname hcpa --weights_dir "%WEIGHTS_DIR%" > adni_local_smoke.log 2>&1
if errorlevel 1 goto :show_adni_log
echo ADNI local smoke test finished. Log saved to adni_local_smoke.log

REM Conservative PPMI smoke test.
REM Use the 2-class version for now because the current ROI export only contains 1 Control subject.
python finetune_lbnm.py --dataset_name ppmi_our_2cls --roi_root "%PPMI_ROI_ROOT%" --label_csv "%PPMI_CSV%" --decoder --decoder_layer 32 --cv_fold_n 2 --fold_ids 0 --checkpoint_fold 0 --epochs 1 --batch_size 1 --device cuda:0 --load_dname hcpa --weights_dir "%WEIGHTS_DIR%" > ppmi_local_smoke.log 2>&1
if errorlevel 1 goto :show_ppmi_log
echo PPMI local smoke test finished. Log saved to ppmi_local_smoke.log
goto :eof

:show_adni_log
echo ADNI local smoke test failed. Showing log:
type adni_local_smoke.log
goto :eof

:show_ppmi_log
echo PPMI local smoke test failed. Showing log:
type ppmi_local_smoke.log
goto :eof

:fail
echo Python environment check failed. Make sure this CMD session is using an environment that has torch and torch_geometric.
