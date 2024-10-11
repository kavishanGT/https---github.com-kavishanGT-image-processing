import torch
import matplotlib.pyplot as plt

def visualize_results(frame, mask_true, mask_pred):
    frame = frame.cpu().numpy().transpose(1, 2, 0)  # Reshape for visualization
    mask_true = mask_true.cpu().numpy()
    mask_pred = (torch.sigmoid(mask_pred) > 0.5).cpu().numpy()

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(frame)
    plt.title('Input Frame')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_true, cmap='gray')
    plt.title('Ground Truth Mask')
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask_pred, cmap='gray')
    plt.title('Predicted Mask')
    
    plt.show()