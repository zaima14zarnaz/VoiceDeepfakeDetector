import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, optimizer, criterion, epochs, model_save_path, device):
    """
    Train the model for the given epochs, then compute results for the validation split.
    Save the model that performed the best on the val split in the directory model_save_path
    """
    best_val_acc = 0.0
    prev_acc = 0.0
    best_epoch = -1
    epochs_without_improv = 0

    for epoch in range(epochs+1):
        model.train() # Set model to train mode

        # Initialization for calculating metrics later
        total_loss = 0
        all_preds, all_labels, all_probs = [], [], []

        
        # Load a batch of input tensors (x1, x2) and labels (yb) from the training data loader
        for x1, x2, yb in train_loader:  
            x1, x2, yb = x1.to(device), x2.to(device), yb.to(device) # Move all tensors to the GPU or CPU based on the selected device
            optimizer.zero_grad() # Reset (zero out) all gradients before computing new ones for this batch
            x1 = x1.view(x1.size(0), -1) # Flatten x1 from (batch_size, ..., ...) into (batch_size, num_features)
            x2 = x2.view(x2.size(0), -1) # Flatten x2 similarly for model input
            preds = model(x1, x2) # Perform a forward pass through the model to get predictions
            loss = criterion(preds, yb) # Compute the loss comparing predictions and true labels
            loss.backward() # Backpropagate the loss to compute gradients for all model parameters
            optimizer.step() # Update model weights using the optimizer
            total_loss += loss.item() # Accumulate the scalar loss value for monitoring training progress
            all_preds.extend(preds.argmax(1).cpu().numpy()) # Store predicted class indices (from highest logit) for later evaluation
            all_labels.extend(yb.cpu().numpy())  # Store true labels for metric computation (e.g., accuracy, F1)
            all_probs.extend(torch.softmax(preds, dim=1)[:, 1].detach().cpu().numpy()) # Store predicted probabilities for the positive class (useful for ROC/AUC)


        # Compute average loss
        avg_loss = total_loss / len(train_loader)

        # ----- Training metrics -----
        train_acc = accuracy_score(all_labels, all_preds) # Accuracy
        train_prec = precision_score(all_labels, all_preds, average="binary") # Precision
        train_rec = recall_score(all_labels, all_preds, average="binary") # Recall
        train_f1 = f1_score(all_labels, all_preds, average="binary") # F1-score
        try:
            train_auc = roc_auc_score(all_labels, all_probs) # AUC Score
        except ValueError:
            train_auc = float('nan')  # happens if only one class in batch


        # ----- Validation -----
        model.eval() # Set the model to evaluation mode (disables dropout, batch norm updates)
        val_preds, val_labels = [], [] # Initialize lists to store predicted labels and true labels for the validation set
        with torch.no_grad(): # Disable gradient calculation to reduce memory usage and speed up inference
            # Loop through batches of validation data (inputs x1, x2, and labels yb)
            for x1, x2, yb in val_loader:
                x1, x2, yb = x1.to(device), x2.to(device), yb.to(device) # Move the batch to the appropriate device (GPU or CPU)
                x1 = x1.view(x1.size(0), -1) # Flatten x1 into a 2D tensor for model input
                x2 = x2.view(x2.size(0), -1) # Flatten x2 similarly
                preds = model(x1, x2) # Perform a forward pass through the model to obtain predictions
                val_preds.extend(preds.argmax(1).cpu().numpy()) # Store the predicted class indices for this batch
                val_labels.extend(yb.cpu().numpy()) # Store the true labels for this batch
        # Compute validation accuracy
        val_acc = accuracy_score(val_labels, val_preds)

        # Print results summary
        print(f"Epoch {epoch+1}, "
              f"Loss {avg_loss:.4f}, "
              f"Train Acc {train_acc:.4f}, Prec {train_prec:.4f}, "
              f"Rec {train_rec:.4f}, F1 {train_f1:.4f}, AUC {train_auc:.4f}, "
              f"Val Acc {val_acc:.4f}")

        # Save best model based on accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print(f"  Saved best model (Val Acc: {best_val_acc:.4f})")
        else:
            # If patience=5 subsequent epochs without improvement, stop training
            if val_acc < prev_acc:
                epochs_without_improv += 1
                if epochs_without_improv >= 5:
                    break
            else:
                epochs_without_improv = 0

    print(f"Training complete. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
    return model



