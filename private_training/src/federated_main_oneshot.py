#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import time
import pickle
import numpy as np
import torch
from torch import nn

from torchsummary import summary

from options import args_parser
from update_s4 import LocalUpdate
from utils import test_inference
from models import CNNMnistRelu, CNNMnistTanh
from models import CNNFashion_MnistRelu, CNNFashion_MnistTanh
from models import CNNCifar10Relu, CNNCifar10Tanh
from utils import average_weights, exp_details
from datasets import get_dataset
from torchvision import models
from logging_results import logging

from opacus.dp_model_inspector import DPModelInspector
from opacus.utils import module_modification
from opacus import PrivacyEngine

# For one-shot learning
from oneshot_utils import knowledge_distillation, federated_averaging_oneshot

if __name__ == '__main__':
    
    ############# Common ###################
    args = args_parser()    
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'    
    
    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            if args.activation == 'relu':
                global_model = CNNMnistRelu()
            elif args.activation == 'tanh':
                global_model = CNNMnistTanh()
            global_model.to(device)
            summary(global_model, input_size=(1, 28, 28), device=device)
        elif args.dataset == 'fmnist':
            if args.activation == 'relu':
                global_model = CNNFashion_MnistRelu()
            elif args.activation == 'tanh':
                global_model = CNNFashion_MnistTanh()
            global_model.to(device)
            summary(global_model, input_size=(1, 28, 28), device=device)
        elif args.dataset == 'cifar10':
            if args.activation == 'relu':
                global_model = CNNCifar10Relu()
            elif args.activation == 'tanh':
                global_model = CNNCifar10Tanh()
            global_model.to(device)
            summary(global_model, input_size=(3, 32, 32), device=device)
        elif args.dataset == 'dr':    
            global_model = models.squeezenet1_1(pretrained=True)           
            global_model.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1,1), stride=(1,1))
            global_model.num_classes = 5
            global_model.to(device)
            summary(global_model, input_size=(3, 224, 224), device=device)
    else:
        exit('Error: unrecognized model')
    ############# Common ###################

    ######### DP Model Compatibility #######
    if args.withDP:
        try:
            inspector = DPModelInspector()
            inspector.validate(global_model)
            print("Model's already Valid!\n")
        except:
            global_model = module_modification.convert_batchnorm_modules(global_model)
            inspector = DPModelInspector()
            print(f"Is the model valid? {inspector.validate(global_model)}")
            print("Model is convereted to be Valid!\n")        
    ######### DP Model Compatibility #######

    # Sample the users
    idxs_users = np.random.choice(range(args.num_users),
                                  max(int(args.frac * args.num_users), 1),
                                  replace=False)
    
    print(f"Selected {len(idxs_users)} users for one-shot federated learning")
    
    # Initialize local models for each user
    local_models = []
    local_optimizers = []
    local_privacy_engine = []

    for u in idxs_users:
        local_models.append(copy.deepcopy(global_model))

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(local_models[-1].parameters(), lr=args.lr, 
                                        momentum=args.momentum)        
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(local_models[-1].parameters(), lr=args.lr)             

        if args.withDP:
            privacy_engine = PrivacyEngine(
                local_models[-1],
                batch_size = int(len(train_dataset)*args.sampling_prob), 
                sample_size = len(train_dataset), 
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier = args.noise_multiplier/np.sqrt(len(idxs_users)),
                max_grad_norm = args.max_grad_norm,
            )

            privacy_engine.attach(optimizer)            
            local_privacy_engine.append(privacy_engine)

        local_optimizers.append(optimizer)
    
    # Train local models in parallel
    local_weights = []
    local_losses = []
    
    print("Training local models...")
    for idx, u in enumerate(idxs_users):
        print(f"Training user {u+1}/{args.num_users}")
        
        local_model = LocalUpdate(args=args, dataset=train_dataset, 
                                  u_id=u, idxs=user_groups[u], 
                                  sampling_prob=args.sampling_prob,
                                  optimizer=local_optimizers[idx])
        
        w, loss, _ = local_model.update_weights(
                                        model=local_models[idx],
                                        global_round=0)  # Only one round in one-shot learning
        
        local_weights.append(copy.deepcopy(w))
        local_losses.append(copy.deepcopy(loss))
    
    # Aggregate models using one-shot federated learning approach
    if args.oneshot_method == 'fedavg':
        # Simple federated averaging for one-shot learning
        global_weights = average_weights(local_weights)
    elif args.oneshot_method == 'knowledge_distillation':
        # Knowledge distillation approach
        global_weights = knowledge_distillation(args, device, local_models, train_dataset, test_dataset)
    elif args.oneshot_method == 'fedfisher':
        # FedFisher approach (approximating Fisher Information)
        global_weights = federated_averaging_oneshot(args, local_weights, local_models, train_dataset)
    else:
        print(f"Unknown one-shot method: {args.oneshot_method}, defaulting to federated averaging")
        global_weights = average_weights(local_weights)
    
    # Update global model
    global_model.load_state_dict(global_weights)
    
    # Test the global model
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    
    print(f"\nResults after one-shot federated learning:")
    print(f"Test Accuracy: {100*test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Log results
    train_loss = [sum(local_losses) / len(local_losses)]
    test_log = [[test_acc, test_loss]]
    
    if args.withDP:
        # Calculate privacy budget
        local_privacy_engine[0].steps = 1  # One-shot learning has only one step
        epsilons, _ = local_privacy_engine[0].get_privacy_spent(args.delta)
        epsilon_log = [[epsilons]]
        print(f"Privacy budget (epsilon): {epsilons}")
    else:
        epsilon_log = None
    
    # Save the model
    model_path = f'./models/{args.exp_name}_oneshot.pt'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(global_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Log results
    logging(args, 0, train_loss, test_log, epsilon_log)
