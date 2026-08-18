from .SiameseSonarNetwork import SiameseSonarNetwork
from .AttentionSiameseSonarNetwork import AttentionSiameseSonarNetwork
from .SiameseSonarResNet50 import SiameseSonarResNet50
from lib.LossFunctions import ContrastiveLoss, evaluate_loss, InfoNCELoss

def get_model(CONFIG, device):
    if(CONFIG.get("model", {}).get("name") == "AttentionSiameseSonarNetwork"):
        model = AttentionSiameseSonarNetwork(CONFIG).to(device)
    elif(CONFIG.get("model", {}).get("name") == "SiameseSonarNetwork"):
        model = SiameseSonarNetwork(CONFIG).to(device)
    elif(CONFIG.get("model", {}).get("name") == "SiameseSonarResNet50"):
        model = SiameseSonarResNet50(CONFIG).to(device)
    else:
        raise ValueError(f"Unknown model name: {CONFIG.get('model', {}).get('name')}")
    
    return model

def get_loss(CONFIG):
    if(CONFIG.get("loss", {}).get("name") == "ContrastiveLoss"):
        criterion = ContrastiveLoss(margin=float(CONFIG.get("loss", {}).get("margin", 1.0)))
    elif(CONFIG.get("loss", {}).get("name") == "InfoNCELoss"):
        criterion = InfoNCELoss(temperature=float(CONFIG.get("loss", {}).get("temperature", 0.07)))
    else:
        raise ValueError(f"Unknown loss function name: {CONFIG.get('loss', {}).get('name')}")
    
    return criterion