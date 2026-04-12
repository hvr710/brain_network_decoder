# Custom downstream examples on the Linux server mount.
# These use the new ROI .npy adapter and only require the target dataset plus pre-trained weights.
# Data defaults to /mnt/dataset4/... for adni_our_2cls / ppmi_our_2cls, so roi_root/label_csv can be omitted.
# --device auto picks the least-used visible GPU via nvidia-smi.
# --checkpoint_fold 0 reuses the available fold0 pre-trained checkpoint for all downstream folds.
python finetune_lbnm.py --dataset_name adni_our_2cls --decoder --decoder_layer 32 --cv_fold_n 5 --checkpoint_fold 0 --device auto --load_dname hcpa --weights_dir pretrain_weights_fold0/none_decoder32_ppmi-abide-taowu-neurocon-hcpa-adni-hcpya_boldwin500_FCFC > adni_our_2cls_decoder32.log
python finetune_lbnm.py --dataset_name ppmi_our_2cls --decoder --decoder_layer 32 --cv_fold_n 5 --checkpoint_fold 0 --device auto --load_dname hcpa --weights_dir pretrain_weights_fold0/none_decoder32_ppmi-abide-taowu-neurocon-hcpa-adni-hcpya_boldwin500_FCFC > ppmi_our_2cls_decoder32.log

python finetune_lbnm.py --datanames taowu --pretrained_datanames hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_taowu_decoder32NoDis_attrFC.log 
python finetune_lbnm.py --datanames neurocon --pretrained_datanames hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_neurocon_decoder32NoDis_attrFC.log 
python finetune_lbnm.py --datanames taowu --pretrained_datanames ppmi abide taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_taowu_decoder32NoAD_attrFC.log 
python finetune_lbnm.py --datanames neurocon --pretrained_datanames ppmi abide taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_neurocon_decoder32NoAD_attrFC.log 
python finetune_lbnm.py --datanames taowu --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_taowu_decoder32NoPD_attrFC.log 
python finetune_lbnm.py --datanames neurocon --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_neurocon_decoder32NoPD_attrFC.log 
python finetune_lbnm.py --datanames taowu --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_taowu_decoder32NoAt_attrFC.log 
python finetune_lbnm.py --datanames neurocon --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_neurocon_decoder32NoAt_attrFC.log 
python finetune_lbnm.py --datanames taowu --pretrained_datanames ppmi abide taowu neurocon hcpa adni hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_taowu_decoder32Full_attrFC.log 
python finetune_lbnm.py --datanames neurocon --pretrained_datanames ppmi abide taowu neurocon hcpa adni hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_neurocon_decoder32Full_attrFC.log 
python finetune_lbnm.py --datanames adni --pretrained_datanames hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_adni_decoder32NoDis_attrFC.log 
python finetune_lbnm.py --datanames adni --pretrained_datanames ppmi abide taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_adni_decoder32NoAD_attrFC.log 
python finetune_lbnm.py --datanames adni --pretrained_datanames adni abide taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_adni_decoder32NoPD_attrFC.log 
python finetune_lbnm.py --datanames adni --pretrained_datanames adni ppmi taowu neurocon hcpa hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_adni_decoder32NoAt_attrFC.log 
python finetune_lbnm.py --datanames adni --pretrained_datanames ppmi abide taowu neurocon hcpa adni hcpya --device cuda:5 --decoder --decoder_layer 32 --cv_fold_n 5 > none_adni_decoder32Full_attrFC.log 
