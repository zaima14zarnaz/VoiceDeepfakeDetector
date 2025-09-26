import torch

def evaluate(model, test_loader, device):
    # evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x1, x2, yb in test_loader:
            x1, x2, yb = x1.to(device), x2.to(device), yb.to(device)
            x1 = x1.view(x1.size(0), -1)  # flatten (B, 1, 321) → (B, 321)
            x2 = x2.view(x2.size(0), -1)
            preds = model(x1, x2)
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)
    accuracy = correct/total
    print("Test Accuracy:", accuracy)
    return accuracy
