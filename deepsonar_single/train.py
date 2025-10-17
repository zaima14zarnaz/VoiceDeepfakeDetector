import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, optimizer, criterion, epochs, model_save_path, auc_fig_path, device):
    best_val_acc = 0.0
    prev_acc = 0.0
    best_epoch = -1
    epochs_without_improv = 0

    for epoch in range(epochs+1):
        model.train()
        total_loss = 0
        all_preds, all_labels, all_probs = [], [], []

        for x1, yb in train_loader:
            x1, yb = x1.to(device), yb.to(device)
            optimizer.zero_grad()
            x1 = x1.view(x1.size(0), -1)
            preds = model(x1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(preds.argmax(1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            all_probs.extend(torch.softmax(preds, dim=1)[:,1].detach().cpu().numpy())

        avg_loss = total_loss / len(train_loader)

        # ----- Training metrics -----
        train_acc = accuracy_score(all_labels, all_preds)
        train_prec = precision_score(all_labels, all_preds, average="binary")
        train_rec = recall_score(all_labels, all_preds, average="binary")
        train_f1 = f1_score(all_labels, all_preds, average="binary")
        try:
            train_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            train_auc = float('nan')  # happens if only one class in batch

        # ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {train_auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Train ROC Curve - Epoch {epoch+1}")
        plt.legend(loc="lower right")
        plt.savefig(auc_fig_path)
        plt.close()

        # ----- Validation -----
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x1, yb in val_loader:
                x1, yb = x1.to(device), yb.to(device)
                x1 = x1.view(x1.size(0), -1)
                preds = model(x1)
                val_preds.extend(preds.argmax(1).cpu().numpy())
                val_labels.extend(yb.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}, "
              f"Loss {avg_loss:.4f}, "
              f"Train Acc {train_acc:.4f}, Prec {train_prec:.4f}, "
              f"Rec {train_rec:.4f}, F1 {train_f1:.4f}, AUC {train_auc:.4f}, "
              f"Val Acc {val_acc:.4f}")

        # ----- Save best model -----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f"  Saved best model (Val Acc: {best_val_acc:.4f})")
        else:
            if val_acc < prev_acc:
                epochs_without_improv += 1
                if epochs_without_improv >= 5:
                    break
            else:
                epochs_without_improv = 0

    print(f"Training complete. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
    return model



