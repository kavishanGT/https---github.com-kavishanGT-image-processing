import torch
import torch.nn.functional as F

def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def evaluate_model(model, test_loader, device):
    model.eval()
    dice_scores, iou_scores = [], []
    
    with torch.no_grad():
        for frames, labels in test_loader:
            frame, frame_prev, frame_next = frames
            frame, frame_prev, frame_next = frame.to(device), frame_prev.to(device), frame_next.to(device)
            labels = labels.to(device)

            # Forward pass
            seg_pred, _ = model(frame, frame_prev, frame_next)

            # Calculate metrics
            dice = dice_score(seg_pred, labels)
            iou = iou_score(seg_pred, labels)

            dice_scores.append(dice.item())
            iou_scores.append(iou.item())

    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_iou = sum(iou_scores) / len(iou_scores)

    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")