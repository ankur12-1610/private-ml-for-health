import torch
import torch.nn.functional as F
import copy
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset

def knowledge_distillation(args, device, local_models, train_dataset, test_dataset, temperature=2.0):
    """
    Knowledge distillation approach for one-shot federated learning.
    Uses the ensemble of local models to train a global model.
    
    Args:
        args: Arguments
        device: Device to run computations on
        local_models: List of trained local models
        train_dataset: Training dataset
        test_dataset: Test dataset
        temperature: Temperature parameter for softening probabilities
        
    Returns:
        Global model state dict
    """
    # Create a new global model
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            if args.activation == 'relu':
                from models import CNNMnistRelu
                global_model = CNNMnistRelu()
            else:
                from models import CNNMnistTanh
                global_model = CNNMnistTanh()
        elif args.dataset == 'fmnist':
            if args.activation == 'relu':
                from models import CNNFashion_MnistRelu
                global_model = CNNFashion_MnistRelu()
            else:
                from models import CNNFashion_MnistTanh
                global_model = CNNFashion_MnistTanh()
        elif args.dataset == 'cifar10':
            if args.activation == 'relu':
                from models import CNNCifar10Relu
                global_model = CNNCifar10Relu()
            else:
                from models import CNNCifar10Tanh
                global_model = CNNCifar10Tanh()
        elif args.dataset == 'dr':
            from torchvision import models
            global_model = models.squeezenet1_1(pretrained=True)
            global_model.classifier[1] = torch.nn.Conv2d(512, 5, kernel_size=(1,1), stride=(1,1))
            global_model.num_classes = 5
    
    global_model.to(device)
    global_model.train()
    
    # Set optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)
    
    # Create data loader
    trainloader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
    
    # Train global model using knowledge distillation
    for epoch in range(args.local_ep):
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            # Get ensemble predictions from local models
            ensemble_logits = []
            for local_model in local_models:
                local_model.eval()
                with torch.no_grad():
                    logits = local_model(images)
                    ensemble_logits.append(logits)
            
            # Average logits
            avg_logits = torch.zeros_like(ensemble_logits[0])
            for logits in ensemble_logits:
                avg_logits += logits
            avg_logits /= len(ensemble_logits)
            
            # Knowledge distillation
            optimizer.zero_grad()
            student_logits = global_model(images)
            
            # Compute soft targets
            soft_targets = F.softmax(avg_logits / temperature, dim=1)
            
            # Compute distillation loss
            distillation_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                soft_targets,
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # Compute standard cross-entropy loss
            ce_loss = F.cross_entropy(student_logits, labels)
            
            # Combine losses (alpha controls the balance)
            alpha = 0.5
            loss = alpha * distillation_loss + (1 - alpha) * ce_loss
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
    
    return global_model.state_dict()

def federated_averaging_oneshot(args, local_weights, local_models, train_dataset):
    """
    FedFisher approach for one-shot federated learning.
    Uses Fisher Information Matrix to weight the importance of each model parameter.
    
    Args:
        args: Arguments
        local_weights: List of local model weights
        local_models: List of trained local models
        train_dataset: Training dataset
        
    Returns:
        Global model state dict
    """
    device = 'cuda' if args.gpu else 'cpu'
    
    # Simple implementation: Use diagonal approximation of Fisher Information Matrix
    # to weight the parameters from different clients
    
    # Initialize global weights with zeros
    global_weights = copy.deepcopy(local_weights[0])
    for key in global_weights.keys():
        global_weights[key] = torch.zeros_like(global_weights[key])
    
    # Calculate Fisher Information Matrix for each model (diagonal approximation)
    fisher_information = []
    for idx, model in enumerate(local_models):
        model.eval()
        
        # Create a subset of data for Fisher calculation
        subset_size = min(1000, len(train_dataset))
        indices = np.random.choice(range(len(train_dataset)), subset_size, replace=False)
        subset = [train_dataset[i] for i in indices]
        
        # Create dataloader
        dataloader = DataLoader(subset, batch_size=args.local_bs, shuffle=True)
        
        # Initialize Fisher Information
        fisher = {}
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param.data)
        
        # Calculate Fisher Information
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            model.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            
            # Accumulate squared gradients (diagonal Fisher approximation)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Normalize Fisher Information
        for name in fisher:
            fisher[name] /= len(dataloader)
        
        fisher_information.append(fisher)
    
    # Aggregate models using Fisher Information as weights
    weight_keys = list(local_weights[0].keys())
    
    for key in weight_keys:
        # Extract parameter name from key
        param_name = key
        
        # Initialize weighted sum and sum of weights
        weighted_sum = torch.zeros_like(local_weights[0][key])
        sum_of_weights = torch.zeros_like(local_weights[0][key])
        
        for idx in range(len(local_weights)):
            # Get Fisher Information for this parameter
            if param_name in fisher_information[idx]:
                # Use Fisher Information as weight
                weight = fisher_information[idx][param_name]
                
                # Add weighted parameter
                weighted_sum += weight * local_weights[idx][key]
                sum_of_weights += weight
        
        # Avoid division by zero
        sum_of_weights[sum_of_weights == 0] = 1.0
        
        # Compute weighted average
        global_weights[key] = weighted_sum / sum_of_weights
    
    return global_weights
