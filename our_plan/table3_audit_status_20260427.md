# Table3 Audit Status (2026-04-27)

本文件整理当前这轮最新 audit 的数据情况。当前配置已经切到 HuoShan2 本地路径，使用的是你新换的 split。

统一说明：

- `Split summary` = 最终真正参与训练/验证/测试的样本数与 subject 数
- `Source split members/files` = split 源文件或 split 目录里的原始条目统计
- 对于 `sample_level: file` 的任务，一个 subject 可能对应多个 `.npy` 文件，所以 `samples` 和 `subjects` 不会相等

## 总体结论

- 可以直接正式跑的任务：
  - `abide_age`
  - `nki_age`
  - `sald_age`
  - `bhrc_sex`
  - `ppmi_pd_diagnosis`
  - `adni_mci`
  - `adni_ad`
- 有 warning 但可以接受、仍可正式跑的任务：
  - `abcd_sex`
  - `hcp_sex`
  - `nki_education`

## 分任务明细

### 1. `abide_age`

- 结果：正常
- 最终样本：
  - train: `522 samples / 522 subjects`
  - val: `174 samples / 174 subjects`
  - test: `175 samples / 175 subjects`
- 原始 split 条目：
  - train: `raw=1223`, `unique_subjects=522`
  - val: `raw=365`, `unique_subjects=174`
  - test: `raw=395`, `unique_subjects=175`
- 说明：
  - 这是 crop split，raw 条目数大于最终 subject 数是正常现象
  - 没有 `missing_files`
  - 没有 `missing_labels`

### 2. `nki_age`

- 结果：正常
- 最终样本：
  - train: `501 / 501`
  - val: `71 / 71`
  - test: `145 / 145`
- 原始 split 条目：
  - train: `raw=1503`, `unique_subjects=501`
  - val: `raw=213`, `unique_subjects=71`
  - test: `raw=435`, `unique_subjects=145`
- 说明：
  - 当前新 split 是 crop 风格，所以 raw 条目数为 subject 数的约 3 倍
  - 没有 `missing_files`
  - 没有 `missing_labels`

### 3. `sald_age`

- 结果：正常
- 最终样本：
  - train: `345 / 345`
  - val: `49 / 49`
  - test: `99 / 99`
- 原始 split 条目：
  - train: `raw=1034`, `unique_subjects=345`
  - val: `raw=147`, `unique_subjects=49`
  - test: `raw=297`, `unique_subjects=99`
- 说明：
  - 这项前面出过 `0/0/0`，后来已修复为 crop split 解析正常
  - 没有 `missing_files`
  - 没有 `missing_labels`

### 4. `abcd_sex`

- 结果：可跑，带 1 个坏文件名 warning
- 最终样本：
  - train: `699 samples / 350 subjects`
  - val: `100 samples / 50 subjects`
  - test: `200 samples / 100 subjects`
- 原始 split 文件：
  - train: `raw=699`, `valid_names=699`, `unique_subjects=350`
  - val: `raw=100`, `valid_names=100`, `unique_subjects=50`
  - test: `raw=201`, `valid_names=200`, `unique_subjects=100`
- 具体问题：
  - `test invalid_file_names=1`
  - 例子：
    - `first200_frames_000050-000249.npy`
- 说明：
  - 这是单个文件名不符合当前 subject 解析规则
  - 影响非常小，可直接正式跑

### 5. `hcp_sex`

- 结果：可跑，但 label 源缺少一小部分 subject
- 最终样本：
  - train: `3611 samples / 602 subjects`
  - val: `1205 samples / 201 subjects`
  - test: `1181 samples / 197 subjects`
- 原始 split 文件：
  - train: `raw=3635`, `valid_names=3635`, `unique_subjects=606`
  - val: `raw=1211`, `valid_names=1211`, `unique_subjects=202`
  - test: `raw=1211`, `valid_names=1211`, `unique_subjects=202`
- 具体问题：
  - train `missing_labels=24`
  - val `missing_labels=6`
  - test `missing_labels=30`
- 缺失 label 的 file 例子：
  - train:
    - `202820__REST1_LR_hp2000_clean_0000-0199.npy`
    - `202820__REST1_LR_hp2000_clean_0200-0399.npy`
    - `202820__REST1_LR_hp2000_clean_0400-0599.npy`
    - `202820__REST1_LR_hp2000_clean_0600-0799.npy`
    - `202820__REST1_LR_hp2000_clean_0800-0999.npy`
    - `202820__REST1_LR_hp2000_clean_1000-1199.npy`
    - `757764__REST1_LR_hp2000_clean_0000-0199.npy`
    - `757764__REST1_LR_hp2000_clean_0200-0399.npy`
  - val:
    - `140319__REST1_LR_hp2000_clean_0000-0199.npy`
    - `140319__REST1_LR_hp2000_clean_0200-0399.npy`
    - `140319__REST1_LR_hp2000_clean_0400-0599.npy`
    - `140319__REST1_LR_hp2000_clean_0600-0799.npy`
    - `140319__REST1_LR_hp2000_clean_0800-0999.npy`
    - `140319__REST1_LR_hp2000_clean_1000-1199.npy`
  - test:
    - `104820__REST1_LR_hp2000_clean_0000-0199.npy`
    - `104820__REST1_LR_hp2000_clean_0200-0399.npy`
    - `104820__REST1_LR_hp2000_clean_0400-0599.npy`
    - `104820__REST1_LR_hp2000_clean_0600-0799.npy`
    - `104820__REST1_LR_hp2000_clean_0800-0999.npy`
    - `104820__REST1_LR_hp2000_clean_1000-1199.npy`
    - `157942__REST1_LR_hp2000_clean_0000-0199.npy`
    - `157942__REST1_LR_hp2000_clean_0200-0399.npy`
- 缺失 label 的 subject 例子：
  - train:
    - `202820`
    - `757764`
    - `902242`
    - `905147`
  - val:
    - `140319`
  - test:
    - `104820`
    - `157942`
    - `180937`
    - `211417`
    - `792867`
- 说明：
  - 这是 label 源本身缺 subject，不是路径或 split 解析错误
  - 这些 subject 会被自动丢掉

### 6. `bhrc_sex`

- 结果：正常
- 最终样本：
  - train: `325 / 325`
  - val: `46 / 46`
  - test: `94 / 94`
- 原始 split 条目：
  - train: `raw=325`, `unique_subjects=325`
  - val: `raw=46`, `unique_subjects=46`
  - test: `raw=94`, `unique_subjects=94`
- 说明：
  - 没有 `missing_files`
  - 没有 `missing_labels`

### 7. `ppmi_pd_diagnosis`

- 结果：正常
- 最终样本：
  - train: `331 / 331`
  - val: `47 / 47`
  - test: `96 / 96`
- 原始 split 条目：
  - train: `raw=331`, `unique_subjects=331`
  - val: `raw=47`, `unique_subjects=47`
  - test: `raw=96`, `unique_subjects=96`
- 说明：
  - 没有 `missing_files`
  - 没有 `missing_labels`

### 8. `adni_mci`

- 结果：正常
- 最终样本：
  - train: `176 / 176`
  - val: `58 / 58`
  - test: `58 / 58`
- 原始 split 条目：
  - train: `raw=176`, `unique_subjects=176`
  - val: `raw=58`, `unique_subjects=58`
  - test: `raw=58`, `unique_subjects=58`
- 说明：
  - 没有 `missing_files`
  - 没有 `missing_labels`

### 9. `adni_ad`

- 结果：正常
- 最终样本：
  - train: `94 / 94`
  - val: `32 / 32`
  - test: `31 / 31`
- 原始 split 条目：
  - train: `raw=94`, `unique_subjects=94`
  - val: `raw=32`, `unique_subjects=32`
  - test: `raw=31`, `unique_subjects=31`
- 说明：
  - 没有 `missing_files`
  - 没有 `missing_labels`

### 10. `nki_education`

- 结果：可跑，剩少量 label 缺失
- 最终样本：
  - train: `492 / 492`
  - val: `71 / 71`
  - test: `144 / 144`
- 原始 split 条目：
  - train: `raw=1503`, `unique_subjects=501`
  - val: `raw=213`, `unique_subjects=71`
  - test: `raw=435`, `unique_subjects=145`
- 具体问题：
  - train `missing_labels=9`
  - test `missing_labels=1`
  - val `missing_labels=0`
- 缺失 label 的 subject 例子：
  - train:
    - `56453`
    - `60806`
    - `61415`
    - `61709`
    - `62268`
    - `62361`
    - `77758`
    - `79353`
    - `...`
  - test:
    - `78238`
- 说明：
  - 前面已经把 `education_group` 从 `participant_education` 自动补了一轮
  - 现在剩下这批基本是原始 label 里确实没有可推断值
  - 影响已经很小，可直接正式跑

## 建议

- 正式训练可以继续
- 需要在结果解释里保留备注的任务：
  - `abcd_sex`：test 有 1 个无效文件名
  - `hcp_sex`：label 源缺一小部分 subject
  - `nki_education`：仍有极少量 subject 无 education label
