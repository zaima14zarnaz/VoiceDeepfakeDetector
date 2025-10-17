from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import torch
import torch.nn.functional as F
import numpy as np

def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x1, yb in test_loader:
            x1, yb = x1.to(device), yb.to(device)
            x1 = x1.view(x1.size(0), -1)

            preds = model(x1)
            probs = F.softmax(preds, dim=1)

            # store positive class probabilities (for binary classification)
            if preds.size(1) == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())

            pred_labels = preds.argmax(1)
            correct += (pred_labels == yb).sum().item()
            total += yb.size(0)

            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    # convert to numpy
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs) if len(all_probs) > 0 else None

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # AUC & EER (binary only)
    auc, eer = None, None
    if len(set(all_labels)) == 2 and all_probs is not None:
        auc = roc_auc_score(all_labels, all_probs)

        # compute EER
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs, pos_label=1)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1-score:      {f1:.4f}")
    if auc is not None:
        print(f"AUC:           {auc:.4f}")
        print(f"EER:           {eer:.4f} (at threshold {eer_threshold:.4f})")
    else:
        print("AUC/EER:       N/A (multi-class task)")

    return accuracy, precision, recall, f1, auc, eer
