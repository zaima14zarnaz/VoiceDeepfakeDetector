import torch

def evaluate(model, test_loader, device):
    # evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.view(xb.size(0), -1)  # flatten (B, 1, 321) → (B, 321)
            preds = model(xb)
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)
    accuracy = correct/total
    print("Test Accuracy:", accuracy)
    return accuracy
