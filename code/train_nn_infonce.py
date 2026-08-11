import torch
import copy


from lib.SonarPairDataset import build_dataloader, prepare_datasets
from lib.SiameseSonarNetwork import SiameseSonarNetwork
from lib.SiameseSonarResNet50 import SiameseSonarResNet50
from lib.LossFunctions import ContrastiveLoss, evaluate_loss
from lib.utility_training import save_history_to_csv, ROOT_DIR, train, save_model, set_seed

CONFIG = {
    "epochs": 15,
    "learning_rate": 1e-4,
    "margin": 1.0,
    "model": {
        "model_name": "SiameseResnet50-o-img-No_Meta",
        "depth": 3,
        "base_channels": 16,
        "batch_size": 64,
    },
    "dataset": "Grid-All-eq-sf100-wn",
    "dataset_folder": "../",
    "target_size": (3000, 100),
}

if(__name__ == "__main__"):

    model_name = CONFIG.get("model", {}).get("model_name", "") + "_" + CONFIG.get("dataset_file", "data_set")

    print("prepare dataset")
    set_seed(CONFIG.get("seed", 42))
    train_groups, val_groups, test_groups = prepare_datasets(CONFIG)

    train_loader, train_dataset = build_dataloader(train_groups, CONFIG, shuffle=True)
    val_loader, val_dataset = build_dataloader(val_groups, CONFIG, shuffle=False)
    test_loader, test_dataset = build_dataloader(test_groups, CONFIG, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseSonarResNet50(CONFIG.get("model", {})).to(device)
    criterion = ContrastiveLoss(margin=float(CONFIG.get("margin", 1.0)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(CONFIG.get("learning_rate", 1e-4)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    print("start training ")
    # TRAINING
    best_model_weights, best_val_loss, history_rows = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=int(CONFIG.get("epochs", 15)), print_progress=True)
    save_history_to_csv(history_rows, ROOT_DIR / CONFIG.get("history_csv",  "training_history.csv"))
    save_model(ROOT_DIR / model_name, model, CONFIG.get("recommanded_threshold", 0.5))