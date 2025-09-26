
import torch
def train(model, train_loader, optimizer, criterion, save_path, device):
    # training loop
    for epoch in range(200):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x1, x2, yb in train_loader:
            x1, x2, yb = x1.to(device), x2.to(device), yb.to(device)
            optimizer.zero_grad()
            x1 = x1.view(x1.size(0), -1)  # flatten (B, 1, 321) → (B, 321)
            x2 = x2.view(x2.size(0), -1) 
            preds = model(x1, x2)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)
        print(f"Epoch {epoch+1}, Loss {total_loss/len(train_loader):.4f}, "
            f"Train Acc {correct/total:.4f}")
    torch.save(model.state_dict(), save_path)
    return model
        
