import sys
import torch
import numpy as np

from train_nn_resnet50 import CONFIG
from lib.SiameseSonarResNet50 import SiameseSonarResNet50
from lib.SonarPairDataset import prepare_datasets, build_dataloader
from lib.utility_training import set_seed, compute_predictions, ROOT_DIR
from lib.LossFunctions import evaluate_loss, ContrastiveLoss


def stats_from_distances(distances, labels):
    pos = distances[labels == 1]
    neg = distances[labels == 0]
    def s(a):
        if len(a) == 0:
            return {"mean": None}
        return {"mean": float(np.mean(a)), "median": float(np.median(a)), "std": float(np.std(a)), "count": int(len(a))}
    return {"pos": s(pos), "neg": s(neg)}


def load_checkpoint(path, device):
    try:
        ck = torch.load(path, map_location=device)
        return ck
    except TypeError:
        # older code sometimes passes unknown kwargs
        return torch.load(path)
    except Exception as e:
        print(f"Could not load checkpoint {path}: {e}")
        return None


def main():
    set_seed(CONFIG.get("seed", 42))

    train_groups, val_groups, test_groups = prepare_datasets(CONFIG)
    train_loader, _ = build_dataloader(train_groups, CONFIG, shuffle=True)
    val_loader, _ = build_dataloader(val_groups, CONFIG, shuffle=False)
    test_loader, _ = build_dataloader(test_groups, CONFIG, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseSonarResNet50(CONFIG.get("model", {})).to(device)

    model_name = CONFIG.get("model", {}).get("model_name", "") + "_" + CONFIG.get("dataset", "dataset")
    checkpoint_path = ROOT_DIR / "model" / f"{model_name}.pt"
    print(f"Looking for checkpoint: {checkpoint_path}")

    ck = load_checkpoint(checkpoint_path, device)
    if ck is None:
        print("No checkpoint loaded — aborting diagnostics.")
        sys.exit(1)

    if "model_state_dict" in ck:
        model.load_state_dict(ck["model_state_dict"])
    else:
        try:
            model.load_state_dict(ck)
        except Exception as e:
            print(f"Failed to load state_dict directly: {e}")
            sys.exit(1)

    model.eval()

    print("Computing distances and labels for splits...")
    train_dist, train_lbl = compute_predictions(model, train_loader, device)
    val_dist, val_lbl = compute_predictions(model, val_loader, device)
    test_dist, test_lbl = compute_predictions(model, test_loader, device)

    print("Label balance (train):", int((train_lbl==1).sum()), "pos /", int((train_lbl==0).sum()), "neg")
    print("Label balance (val):", int((val_lbl==1).sum()), "pos /", int((val_lbl==0).sum()), "neg")
    print("Label balance (test):", int((test_lbl==1).sum()), "pos /", int((test_lbl==0).sum()), "neg")

    print("Train distance stats:", stats_from_distances(train_dist, train_lbl))
    print("Val distance stats:", stats_from_distances(val_dist, val_lbl))
    print("Test distance stats:", stats_from_distances(test_dist, test_lbl))

    # Compute losses with ContrastiveLoss in eval
    criterion = ContrastiveLoss(margin=float(CONFIG.get("margin", 1.0)))
    train_loss = evaluate_loss(model, train_loader, criterion, device)
    val_loss = evaluate_loss(model, val_loader, criterion, device)
    test_loss = evaluate_loss(model, test_loader, criterion, device)

    print(f"Evaluate loss (Contrastive) — Train: {train_loss:.6f}, Val: {val_loss:.6f}, Test: {test_loss:.6f}")


if __name__ == "__main__":
    main()
