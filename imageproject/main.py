import torch
from dataset import UltrasoundVideoDataset
from torch.utils.data import DataLoader
from model import FLANet
from training import train_model, dice_loss
from evaluation import evaluate_model
from visualization import visualize_results
import torch.optim as optim
from torchvision import transforms

if __name__ == '__main__':
    # Setup code
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = FLANet(in_channels=3, out_channels=1).to(device)
    
    # Loss function and optimizer
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Paths to your dataset
    train_frame_paths = ['path_to_train_frame1.jpg', 'path_to_train_frame2.jpg', ...]  # Update with actual paths
    train_mask_paths = ['path_to_train_mask1.jpg', 'path_to_train_mask2.jpg', ...]    # Update with actual paths
    test_frame_paths = ['path_to_test_frame1.jpg', 'path_to_test_frame2.jpg', ...]    # Update with actual paths
    test_mask_paths = ['path_to_test_mask1.jpg', 'path_to_test_mask2.jpg', ...]       # Update with actual paths

    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((352, 352)),  # Resize to match model input size
        transforms.ToTensor()           # Convert images to tensors
    ])

    # Create datasets
    train_dataset = UltrasoundVideoDataset(train_frame_paths, train_mask_paths, transform=transform)
    test_dataset = UltrasoundVideoDataset(test_frame_paths, test_mask_paths, transform=transform)

    # Initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=50, device=device)

    # Evaluate the model
    evaluate_model(model, test_loader, device)

    # Optionally visualize results (update with actual data to visualize)
    #visualize_results(frame, mask, seg_pred)
