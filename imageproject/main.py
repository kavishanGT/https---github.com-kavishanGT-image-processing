import torch
from dataset import UltrasoundVideoDataset
from torch.utils.data import Dataset, DataLoader
from model import FLANet
from training import train_model, dice_loss
from evaluation import evaluate_model
from visualization import visualize_results
import torch.optim as optim

if __name__ == '__main__':
    # Setup code
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, datasets, etc.
    model = FLANet(in_channels=3, out_channels=1).to(device)
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Initialize DataLoader
    train_loader = DataLoader(...)
    test_loader = DataLoader(...)

    # Train and evaluate
    train_model(model, train_loader, criterion, optimizer, num_epochs=50, device=device)
    evaluate_model(model, test_loader, device)

    # Visualize results
    # visualize_results(...)
