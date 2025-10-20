from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import torch
import torch.nn.functional as F
import numpy as np

def evaluate(model, test_loader, device):
    """
    Evalute the given model on the test split and return the following computed metrics:
        1. Accuracy
        2. Precision
        3. Recall
        4. F1-Score
        5. AUC Score (if batch size > 1)
        6. EER rate and threshold
    """
    model.eval() # Set model to evaluation mode

    # Initialization for calculating metrics later
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad(): # Disable gradient calculation to reduce memory usage and speed up inference
        # Loop through batches of test data (inputs x1, x2, and labels yb)
        for x1, x2, yb in test_loader:
            x1, x2, yb = x1.to(device), x2.to(device), yb.to(device) # Move all feature and label tensors to GPU
            x1 = x1.view(x1.size(0), -1) # Flatten x1 from (batch_size, ..., ...) into (batch_size, num_features)
            x2 = x2.view(x2.size(0), -1) # Flatten x2 similarly for model input

            preds = model(x1, x2) # Perform a forward pass through the model to get predictions
            probs = F.softmax(preds, dim=1) # Softmax on the predictions to get predicted labels

            # store positive class probabilities (for binary classification)
            if preds.size(1) == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())

            pred_labels = preds.argmax(1) # Get the predicted label with the highest softmax probability

            # Compute number of correct labels for accuracy calculations
            correct += (pred_labels == yb).sum().item()
            total += yb.size(0)

            # Store predicted and gt results
            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    # convert to numpy
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs) if len(all_probs) > 0 else None

    # ------ Compute test results -------
    accuracy = correct / total # Accuracy
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) # Precision
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0) # Recall
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) # F1 Score

    # AUC & EER (binary only)
    auc, eer = None, None  # Initialize AUC and EER metrics

    # Compute AUC and EER only for binary classification
    if len(set(all_labels)) == 2 and all_probs is not None:
        auc = roc_auc_score(all_labels, all_probs)  # Area Under the ROC Curve

        # Compute ROC curve points
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs, pos_label=1)

        fnr = 1 - tpr  # False Negative Rate
        # Find threshold where FPR and FNR are closest (Equal Error Rate)
        eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    # ---- Print results summary ----
    print(f"Test Accuracy: {accuracy:.4f}")  # Overall classification accuracy
    print(f"Precision:     {precision:.4f}")  # Positive predictive value
    print(f"Recall:        {recall:.4f}")     # True positive rate
    print(f"F1-score:      {f1:.4f}")         # Harmonic mean of precision and recall
    if auc is not None:
        print(f"AUC:           {auc:.4f}")  # Area under ROC curve
        print(f"EER:           {eer:.4f} (at threshold {eer_threshold:.4f})")  # Equal Error Rate info
    else:
        print("AUC/EER:       N/A (multi-class task)")  # Skip for multi-class cases


    return accuracy, precision, recall, f1, auc, eer
