import torch

class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # KIMORE ex5
    TRAIN_DATA_PATH = 'data/kimore_ex5_train.pkl'
    TRAIN_LABEL_PATH = 'data/kimore_ex5_train_y.pkl'
    TEST_DATA_PATH  = 'data/kimore_ex5_test.pkl'
    TEST_LABEL_PATH = 'data/kimore_ex5_test_y.pkl'

    DATA_PATH = TRAIN_DATA_PATH
    LABEL_PATH = TRAIN_LABEL_PATH

    NUM_JOINTS = 25
    coords_dim = 3
    max_frame = 100

    batch_size = 16
    learning_rate = 0.001
    weight_decay = 1e-4
    epochs = 80
    seed = 2025
    val_ratio = 0.1
    seeds = [2024, 2025, 2026]

    cf_lr = 0.05
    cf_steps = 100
    lambda_l1 = 0.01
    lambda_smooth = 0.5
    lambda_bone = 0.5
    cf_ymax = 50.0
    cf_margin = 5.0
    cf_vis_k_each = 2
    cf_eval_samples = 999  # 全测试集

    exp_name = "ex5"
    variant = "full"
    SAVE_DIR = f"checkpoints/{exp_name}/{variant}"

    use_augment = False
    lambda_rank = 0.05
    use_temporal_attention = False

