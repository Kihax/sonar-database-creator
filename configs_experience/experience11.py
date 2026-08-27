
"""
Ce script est le fichier configuration de l'expérience 1.
Il contient les paramètres d'apprentissage, de modèle et de données pour l'entraînement du modèle de détection d'objets sonar.
"""

config = {
    "name": "experience11",

    "epochs": 15,
    "learning_rate": 1e-4,
    "data": {
        "dataset": "Grid-All-eq-sf100",
        "dataset_folder": "./database/",
        "target_size": (3000, 100),
        "train_ratio": 0.60,
        "val_ratio": 0.20,
        "test_ratio": 0.20,
        "batch_size": 64,
        "min_imagettes": 0,
        "min_detection_range": 0.35, # without nadir
        "max_detection_range": 1.0,
        "centered": False,
        "channel": ["image", "heading"],
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
        "name": "AttentionSiameseSonarNetwork",
        "depth": 3,
        "base_channels": 16,
        "save_best_model": True
    },
}