import torch
import torch.optim as optim

# Dice Loss Function
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

# Training Function
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for frames, labels in train_loader:
            frame, frame_prev, frame_next = frames
            frame, frame_prev, frame_next = frame.to(device), frame_prev.to(device), frame_next.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            seg_pred, _ = model(frame, frame_prev, frame_next)

            # Compute loss
            loss = criterion(seg_pred, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
