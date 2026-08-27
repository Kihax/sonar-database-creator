
config = {
    "name": "experience16",

    "epochs": 15,
    "learning_rate": 1e-4,
    "data": {
        "dataset": "HP-Centered-100",
        "dataset_folder": "./database/",
        "target_size": (3000, 100),
        "train_ratio": 0.60,
        "val_ratio": 0.20,
        "test_ratio": 0.20,
        "batch_size": 16,
        "min_imagettes": 0,
        "min_detection_range": 0.35,
        "max_detection_range": 1.0,
        "centered": False,
        "channel": ["image"],
        "meta": [],
        "depth_normalization": 50,
        "roll_normalization": 180,
        "pitch_normalization": 180,
        "heading_normalization": 180,
    },
    "loss": {
        "name": "ContrastiveLoss",
        "temperature": 0.07
    },
    "model": {
        "name": "SiameseSonarResNet50",
        "depth": 3,
        "base_channels": 16,
        "save_best_model": True
    },
}