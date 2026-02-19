import argparse
from pathlib import Path

import torch

from data import get_cifar100_dataloaders, get_svhn_dataloader, get_textures_dataloader
from resnet import ResidualBlock, ResNet_CIFAR
from utils import save_features_to_csv


def extract_weights_and_features(checkpoint_path, dataloader, device=None, only_weights=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet_CIFAR(ResidualBlock, [2, 2, 2], num_classes=100)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint["net"]

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    fc = model.fc
    weight = fc.weight.detach().cpu().numpy()
    bias = fc.bias.detach().cpu().numpy()

    if only_weights:
        return weight, bias, None, None

    features = []
    labels = []

    def hook_fn(module, inputs, output):
        features.append(inputs[0].detach().cpu())

    hook = fc.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            for images, batch_labels in dataloader:
                images = images.to(device)
                model(images)
                labels.append(batch_labels.detach().cpu())
    finally:
        hook.remove()

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return weight, bias, features, labels


OOD_LOADERS = {
    "svhn": get_svhn_dataloader,
    "textures": get_textures_dataloader,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/resnet_cifar_best_adam.pth")
    parser.add_argument(
        "--ood", choices=list(OOD_LOADERS.keys()), nargs="+", default=list(OOD_LOADERS.keys())
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    save_dir = checkpoint_path.parent / checkpoint_path.stem
    save_dir.mkdir(exist_ok=True)

    train_loader, test_loader, _ = get_cifar100_dataloaders()

    weight, bias, feat_train, labels_train = extract_weights_and_features(
        checkpoint_path, train_loader
    )
    save_features_to_csv(feat_train, labels_train, save_dir / "ID_cifar100_train.csv")
    print("Saved ID train features")

    _, _, feat_test_id, labels_test_id = extract_weights_and_features(checkpoint_path, test_loader)
    save_features_to_csv(feat_test_id, labels_test_id, save_dir / "ID_cifar100_test.csv")
    print("Saved ID test features")

    for ood_name in args.ood:
        ood_loader = OOD_LOADERS[ood_name]()
        _, _, feat_ood, labels_ood = extract_weights_and_features(checkpoint_path, ood_loader)
        save_features_to_csv(feat_ood, labels_ood, save_dir / f"OOD_{ood_name}_test.csv")
        print(f"Saved OOD {ood_name} features")
