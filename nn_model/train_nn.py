"""
Ce script permet d'entrainer le réseau de neurones, la configuration est adaptable
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from lib.get_model import get_model, get_loss
from lib.SonarPairDataset import build_dataloader, prepare_datasets
from lib.utility_training import save_history_to_csv, ROOT_DIR, train, save_model, set_seed

from configs_experience.experience1 import config as CONFIG

name = CONFIG.get("name", "default_model_name")
print(f"Model name: {name}")

print("prepare dataset")
set_seed(CONFIG.get("seed", 43))
train_groups, val_groups, test_groups = prepare_datasets(CONFIG["data"])

train_loader, train_dataset = build_dataloader(train_groups, CONFIG["data"], shuffle=True)
val_loader, val_dataset = build_dataloader(val_groups, CONFIG["data"], shuffle=False)
test_loader, test_dataset = build_dataloader(test_groups, CONFIG["data"], shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(CONFIG, device)
criterion = get_loss(CONFIG)

optimizer = torch.optim.Adam(model.parameters(), lr=float(CONFIG.get("learning_rate", 1e-4)))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

print("start training ")
# TRAINING
best_model_weights, best_val_loss, history_rows = train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs=int(CONFIG.get("epochs", 15)),
    print_progress=True,
    best_model=CONFIG.get("model", {}).get("save_best_model", True)
)
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

save_history_to_csv(history_rows, ROOT_DIR / "epochs/" / f"{name}_history.csv")
save_model(ROOT_DIR / "model/" / f"{name}.pt", model, CONFIG.get("recommanded_threshold", 0.5))