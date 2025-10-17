import torch

def evaluate(model, test_loader, device):
    # evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, yb in test_loader:
            x, yb = x.to(device), yb.to(device)
            x = x.view(x.size(0), -1)  # flatten (B, 1, 321) → (B, 321)
            preds = model(x)
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)
    accuracy = correct/total
    print("Test Accuracy:", accuracy)
    return accuracy
