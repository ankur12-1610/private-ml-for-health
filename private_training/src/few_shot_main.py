import os
import torch
import numpy as np
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader

from options import args_parser
from datasets import get_dataset
from models import FeatureExtractor
from few_shot import PrototypicalNetwork, EpisodicDataset, train_prototypical, test_prototypical
from utils import test_inference

if __name__ == '__main__':
    # Parse arguments
    args = args_parser()
    
    # Set device
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'
    
    # Load datasets
    train_dataset, test_dataset, _ = get_dataset(args)
    
    # Create episodic datasets
    n_way = 5  # Number of classes per episode
    n_shot = args.n_shot  # Number of support examples per class
    n_query = 15  # Number of query examples per class
    
    train_episodic = EpisodicDataset(train_dataset, n_way, n_shot, n_query)
    test_episodic = EpisodicDataset(test_dataset, n_way, n_shot, n_query)
    
    # Create data loaders
    train_loader = DataLoader(train_episodic, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_episodic, batch_size=1, shuffle=False)
    
    # Create model
    if args.dataset == 'dr':
        base_model = models.squeezenet1_1(pretrained=True)
        base_model.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1,1), stride=(1,1))
        base_model.num_classes = 5
    else:
        # Handle other datasets
        base_model = models.resnet18(pretrained=True)
        base_model.fc = nn.Linear(base_model.fc.in_features, args.num_classes)
        base_model.num_classes = args.num_classes
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(base_model, feature_dim=512)
    
    # Create prototypical network
    model = PrototypicalNetwork(feature_extractor, feature_dim=512)
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    best_acc = 0
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_prototypical(model, train_loader, optimizer, device, epoch)
        
        # Test
        test_acc = test_prototypical(model, test_loader, device)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'./model/few_shot_{args.n_shot}shot.pt')
    
    print(f'Best Test Accuracy: {best_acc:.2f}%')
