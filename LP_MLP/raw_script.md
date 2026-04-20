lp和mlp所有测试 都跑五次，保持相同的train set，然后每个run都要更新seed，把val和test mixup，然后重新跨被试对半分

师兄给的相应的伪代码：

for each lp/mlp task:
    load pretrained backbone
    load train_dataset, val_dataset, test_dataset

    # 1. train 始终固定
    train_features, train_labels = extract_features(train_dataset)

    # 2. 先把原来的 val 和 test 合并
    val_features, val_labels = extract_features(val_dataset)
    test_features, test_labels = extract_features(test_dataset)
    eval_pool = concat(val_features, test_features)
    eval_labels = concat(val_labels, test_labels)

    # 如果严格按你的需求，应该再保留 eval_subject_ids
    # 然后按 subject 做 split，而不是直接按 sample 做 split

    num_runs = 5
    base_seed = 23
    seed_stride = 1

    for run_idx in [0,1,2,3,4]:
        run_seed = base_seed + run_idx * seed_stride

        # 当前代码:
        #   shuffle(eval_pool, seed=run_seed)
        #   val_idx, test_idx = half_split(sample_level)

        # 你要的严格版:
        #   unique_subjects = unique(eval_subject_ids)
        #   shuffle(unique_subjects, seed=run_seed)
        #   val_subjects, test_subjects = half_split(subject_level)
        #   val_idx  = samples whose subject_id in val_subjects
        #   test_idx = samples whose subject_id in test_subjects

        run_val_X, run_val_y = eval_pool[val_idx], eval_labels[val_idx]
        run_test_X, run_test_y = eval_pool[test_idx], eval_labels[test_idx]

        if mode == "linear_probe":
            fit probe only on fixed train_features/train_labels
            evaluate on run_val
            evaluate on run_test

        if mode == "mlp_probe":
            standardize features using train_features stats only
            train MLP probe on fixed train_features/train_labels with seed=run_seed
            each epoch:
                eval on run_val
                eval on run_test
                keep best checkpoint by val loss
            report the test metrics from the best-val epoch

        save:
            run_{idx}_seed_{seed}/config.yaml
            run_{idx}_seed_{seed}/checkpoints/*
            run_{idx}_seed_{seed}/metrics

    aggregate 5 runs:
        mean/std over val metrics
        mean/std over test metrics
        write probe_summary.json



LP:
分类时：LogisticRegression(C=1.0, max_iter=8000, random_state=seed)
回归时：Ridge(alpha=1.0)

MLP:    
hidden_dim = int(mlp_cfg.get('hidden_dim', 64))
    dropout = float(mlp_cfg.get('dropout', 0.1))
    num_layers = max(int(mlp_cfg.get('num_layers', 2)), 1)
    input_norm = str(mlp_cfg.get('input_norm', 'layernorm')).lower()

    layers = []
    if input_norm == 'layernorm':
        layers.append(nn.LayerNorm(int(input_dim)))
    elif input_norm not in {'none', 'identity'}:
        raise ValueError(f"Unsupported mlp_probe.input_norm: {input_norm}")

    in_dim = int(input_dim)
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, int(output_dim)))
    return nn.Sequential(*layers)