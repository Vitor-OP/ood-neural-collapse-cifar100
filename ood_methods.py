import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import norm
from scipy.special import logsumexp, softmax
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from run_networks import extract_weights_and_features
from utils import load_features_from_csv


def evaluate_ood(ind_conf, ood_conf):
    num_ind = len(ind_conf)
    num_ood = len(ood_conf)

    if num_ind == 0 or num_ood == 0:
        return 0.0, 0.0

    recall_num = int(np.floor(0.95 * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    fpr_at_95 = num_fp / num_ood

    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    auroc = metrics.auc(fpr, tpr)

    return auroc, fpr_at_95


def format_latex_table(rows, caption, label):
    headers = ["Method", "AUROC", "FPR@95\\%"]
    body = [[name, f"{auroc:.4f}", f"{fpr_95:.4f}"] for name, auroc, fpr_95 in rows]
    tabular_block = tabulate(body, headers=headers, tablefmt="latex_raw")

    lines = ["\\begin{table}[ht]", "\\centering", tabular_block]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def format_latex_table_side_by_side(left_rows, right_rows, left_title, right_title, caption, label):
    headers = [left_title, "AUROC", "FPR@95\\%", right_title, "AUROC", "FPR@95\\%"]
    right_map = {name: (auroc, fpr_95) for name, auroc, fpr_95 in right_rows}
    body = []
    for name, auroc, fpr_95 in left_rows:
        right_vals = right_map.get(name)
        if right_vals is None:
            right_auroc, right_fpr = "", ""
        else:
            right_auroc, right_fpr = right_vals
            right_auroc = f"{right_auroc:.4f}"
            right_fpr = f"{right_fpr:.4f}"
        body.append([name, f"{auroc:.4f}", f"{fpr_95:.4f}", name, right_auroc, right_fpr])

    tabular_block = tabulate(body, headers=headers, tablefmt="latex_raw")
    tabular_lines = tabular_block.splitlines()
    if tabular_lines and tabular_lines[0].startswith("\\begin{tabular}"):
        tabular_lines[0] = "\\begin{tabular}{lcc|lcc}"
    tabular_block = "\n".join(tabular_lines)

    lines = ["\\begin{table}[ht]", "\\centering", tabular_block]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def softmax_score(logits):
    return np.max(softmax(logits, axis=-1), axis=-1)


def max_logit_score(logits):
    return np.max(logits, axis=-1)


def energy_score(logits):
    return logsumexp(logits, axis=-1)  # higher logsumexp = more ID confidence


def mahalanobis_score(features, train_features, train_labels, num_classes):
    scores = np.zeros(len(features))
    train_means = []
    centered_feats = []

    for i in range(num_classes):
        class_mask = train_labels == i
        if class_mask.sum() > 0:
            class_feats = train_features[class_mask]
            mean = class_feats.mean(axis=0)
            train_means.append(mean)
            centered_feats.append(class_feats - mean)

    centered_feats = np.vstack(centered_feats)
    cov = np.cov(centered_feats.T)
    cov = cov + 1e-4 * np.eye(cov.shape[0])

    try:
        prec = np.linalg.inv(cov)
    except Exception:
        prec = np.linalg.pinv(cov)

    for i, mean in enumerate(train_means):
        diff = features - mean
        dist = np.sqrt(np.sum((diff @ prec) * diff, axis=-1))
        if i == 0:
            scores = dist.copy()
        else:
            scores = np.minimum(scores, dist)

    return -scores


def vim_score(features, logits, train_features, train_logits):
    feat_dim = train_features.shape[1]
    # null space takes the smallest eigenvectors; dim is the principal subspace size
    dim = min(int(feat_dim * 0.75), feat_dim - 1)

    u = train_features.mean(axis=0)
    train_centered = train_features - u

    cov = np.cov(train_centered.T)
    eig_vals, eig_vecs = np.linalg.eig(cov)

    sort_idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, sort_idx]

    null_space = np.ascontiguousarray(eig_vecs[:, dim:])

    vlogit_train = norm(train_centered @ null_space, axis=-1)
    alpha = np.max(train_logits, axis=-1).mean() / vlogit_train.mean()

    feat_centered = features - u
    vlogit = norm(feat_centered @ null_space, axis=-1) * alpha
    energy = logsumexp(logits, axis=-1)

    return -vlogit + energy


def neco_score(features, train_features, neco_dim):
    # Scale: StandardScaler is used because ResNet features are not guaranteed
    # to already live on a unit-norm manifold (unlike ViT [CLS] tokens).
    ss = StandardScaler()
    train_scaled = ss.fit_transform(train_features)
    features_scaled = ss.transform(features)

    # Full PCA on training features (all 64 components for 64-dim features).
    pca = PCA(n_components=train_features.shape[1])
    pca.fit(train_scaled)
    features_reduced = pca.transform(features_scaled)  # shape (N, 64)

    # NECO score: fraction of feature norm explained by the top neco_dim PCs.
    # Higher ratio -> sample is well-explained by the leading ID subspace -> more ID.
    num = np.linalg.norm(features_reduced[:, :neco_dim], axis=-1)
    den = np.linalg.norm(features_scaled, axis=-1)
    return num / (den + 1e-10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/resnet_cifar_best_adam.pth")
    parser.add_argument(
        "--ood", choices=["svhn", "textures"], nargs="+", default=["svhn", "textures"]
    )
    parser.add_argument(
        "--neco-dim",
        type=int,
        default=10,
        help="Number of leading PCA components used by NECO (default: 10)",
    )
    parser.add_argument("--latex", action="store_true", help="Print LaTeX tables")
    parser.add_argument("--caption", type=str, default="", help="LaTeX table caption")
    parser.add_argument("--label", type=str, default="", help="LaTeX table label")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    results_dir = checkpoint_path.parent / checkpoint_path.stem

    feature_id_train, train_labels = load_features_from_csv(results_dir / "ID_cifar100_train.csv")
    feature_id_val, _ = load_features_from_csv(results_dir / "ID_cifar100_test.csv")

    weight, bias, _, _ = extract_weights_and_features(checkpoint_path, None, only_weights=True)

    feature_id_train = np.asarray(feature_id_train, dtype=np.float32)
    feature_id_val = np.asarray(feature_id_val, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    train_labels = np.asarray(train_labels)

    logit_id_train = feature_id_train @ weight.T + bias
    logit_id_val = feature_id_val @ weight.T + bias

    results = []
    for ood_name in args.ood:
        feature_ood, _ = load_features_from_csv(results_dir / f"OOD_{ood_name}_test.csv")
        feature_ood = np.asarray(feature_ood, dtype=np.float32)
        logit_ood = feature_ood @ weight.T + bias

        methods = {
            "softmax": (softmax_score(logit_id_val), softmax_score(logit_ood)),
            "max_logit": (max_logit_score(logit_id_val), max_logit_score(logit_ood)),
            "energy": (energy_score(logit_id_val), energy_score(logit_ood)),
            "mahalanobis": (
                mahalanobis_score(feature_id_val, feature_id_train, train_labels, 100),
                mahalanobis_score(feature_ood, feature_id_train, train_labels, 100),
            ),
            "vim": (
                vim_score(feature_id_val, logit_id_val, feature_id_train, logit_id_train),
                vim_score(feature_ood, logit_ood, feature_id_train, logit_id_train),
            ),
            "neco": (
                neco_score(feature_id_val, feature_id_train, args.neco_dim),
                neco_score(feature_ood, feature_id_train, args.neco_dim),
            ),
        }

        rows = []
        for method_name, (id_scores, ood_scores) in methods.items():
            auroc, fpr_95 = evaluate_ood(id_scores, ood_scores)
            rows.append((method_name, auroc, fpr_95))

        results.append((ood_name, rows))

        if not args.latex:
            print(f"\n--- OOD: {ood_name} ---")
            for method_name, auroc, fpr_95 in rows:
                print(f"{method_name:15} | AUROC: {auroc:.4f} | FPR@95%: {fpr_95:.4f}")

    if args.latex:
        if len(results) == 2:
            left_name, left_rows = results[0]
            right_name, right_rows = results[1]
            caption = args.caption or f"OOD results on {left_name} and {right_name}"
            label = args.label or f"tab:ood_{left_name}_{right_name}"
            print(
                format_latex_table_side_by_side(
                    left_rows, right_rows, left_name, right_name, caption, label
                )
            )
        else:
            for ood_name, rows in results:
                caption = args.caption or f"OOD results on {ood_name}"
                label = args.label or f"tab:ood_{ood_name}"
                print(format_latex_table(rows, caption, label))
