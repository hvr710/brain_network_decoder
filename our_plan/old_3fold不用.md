nki_age:
    display_name: NKI age
    task_type: regression
    target_key: age
    sample_level: subject
    split_format: txt_basename
    id_rule: nki_subject
    split_id_rule: nki_subject
    fold_ids: [1, 2, 3]
    data_root: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\AAL'
    split_source:
      fold1:
        train: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\train1.txt'
        val: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\val1.txt'
        test: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\test1.txt'
        
      fold2:
        train: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\train2.txt'
        val: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\val2.txt'
        test: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\test2.txt'
      fold3:
        train: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\train3.txt'
        val: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\val3.txt'
        test: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\test3.txt'
    label_source: '\\10.16.57.94\dataset1\ningzh\labels\age\NKI.csv'
    label_format: csv
    label_subject_column: Subject
    label_id_rule: identity
    target_column: age
    label_agg: mean
    metrics: [mse, pearson_r]
    model_select_metric: val_mse
    class_map: {}

  sald_age:
    display_name: SALD age
    task_type: regression
    target_key: age
    sample_level: subject
    split_format: txt_basename
    id_rule: numeric_stem
    split_id_rule: numeric_stem
    fold_ids: [1, 2, 3]
    data_root: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\AAL'
    split_source:
      fold1:
        train: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\100ROI_split\train1.txt'
        val: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\100ROI_split\val1.txt'
        test: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\100ROI_split\test1.txt'
      fold2:
        train: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\100ROI_split\train2.txt'
        val: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\100ROI_split\val2.txt'
        test: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\100ROI_split\test2.txt'
      fold3:
        train: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\100ROI_split\train3.txt'
        val: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\100ROI_split\val3.txt'
        test: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\SALD\100ROI_split\test3.txt'
    label_source: '\\10.16.57.94\dataset1\ningzh\labels\age\SALD.csv'
    label_format: csv
    label_subject_column: Subject
    label_id_rule: identity
    target_column: age
    label_agg: first_non_null
    metrics: [mse, pearson_r]
    model_select_metric: val_mse
    class_map: {}

    bhrc_sex:
    display_name: BHRC sex
    task_type: classification
    target_key: sex
    model:
      epochs: 80
      max_patience: 12
    sample_level: subject
    split_format: txt_basename
    id_rule: sub_numeric
    split_id_rule: sub_numeric
    fold_ids: [1, 2, 3]
    data_root: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\AAL'
    split_source:
      fold1:
        train: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\100ROI_split\train1.txt'
        val: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\100ROI_split\val1.txt'
        test: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\100ROI_split\test1.txt'
      fold2:
        train: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\100ROI_split\train2.txt'
        val: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\100ROI_split\val2.txt'
        test: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\100ROI_split\test2.txt'
      fold3:
        train: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\100ROI_split\train3.txt'
        val: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\100ROI_split\val3.txt'
        test: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\BHRC\100ROI_split\test3.txt'
    label_source: '\\10.16.57.94\dataset1\ningzh\labels\sex\BHRC.csv'
    label_format: csv
    label_subject_column: Subject
    label_id_rule: identity
    target_column: Gender
    label_agg: first_non_null
    class_map: {"0": 0, "1": 1}
    metrics: [acc, f1]
    model_select_metric: val_f1

 nki_education:
    display_name: NKI education
    task_type: classification
    target_key: y
    model:
      epochs: 80
      max_patience: 12
    sample_level: subject
    split_format: txt_basename
    id_rule: nki_subject
    split_id_rule: nki_subject
    fold_ids: [1, 2, 3]
    data_root: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\AAL'
    split_source:
      fold1:
        train: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\train1.txt'
        val: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\val1.txt'
        test: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\test1.txt'
      fold2:
        train: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\train2.txt'
        val: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\val2.txt'
        test: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\test2.txt'
      fold3:
        train: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\train3.txt'
        val: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\val3.txt'
        test: '\\10.20.33.82\dataset4\DATASETS\fmri_pretraining\fmri_dataset\roi\NKI\100ROI_split\test3.txt'
    label_source: '\\10.16.57.94\dataset1\ningzh\labels\education\NKI.csv'
    label_format: csv
    label_subject_column: Subject
    label_id_rule: identity
    target_column: education_group
    label_agg: max
    class_map: {"0": 0, "1": 1, "2": 2}
    metrics: [acc, f1]
    model_select_metric: val_f1
