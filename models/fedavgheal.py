import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
import copy
from utils.args import *
from models.utils.federated_model import FederatedModel

class FedAvGHEAL(FederatedModel):
    NAME = 'fedavgheal'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedAvGHEAL, self).__init__(nets_list, args, transform)
        
        self.client_update = {}
        self.increase_history = {}
        self.mask_dict = {}
        
        self.euclidean_distance = {}
        self.previous_weights = {}
        self.previous_delta_weights = {}

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        online_clients = self.online_clients_sequence[self.epoch_index]
        self.online_clients = online_clients
        print(self.online_clients)

        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i])
            
            if self.args.wHEAL == 1:
                net_params = self.nets_list[i].state_dict()
                global_params = self.global_net.state_dict()
                param_names = [name for name, _ in self.nets_list[i].named_parameters()]
                update_diff = {key: global_params[key] - net_params[key] for key in global_params}
                
                # Call consistency_mask for each client update
                mask = self.consistency_mask(i, update_diff)
                self.mask_dict[i] = mask
                masked_update = {key: update_diff[key] * mask[key] for key in update_diff}
                self.client_update[i] = masked_update
                    
                self.compute_distance(i, self.client_update[i], param_names)
                
        freq = self.get_params_diff_weights()
        self.aggregate_nets_parameter(freq)

    def consistency_mask(self, client_id, update_diff):
        # Raise an error if update_diff is None
        if update_diff is None:
            raise ValueError(f"Updates for client {client_id} are None.")
        
        # Initialize increase_history for the client if it doesn't exist
        if client_id not in self.increase_history:
            self.increase_history[client_id] = {key: torch.zeros_like(val) for key, val in update_diff.items()}

        # Epoch 0 specific processing
        if self.epoch_index == 0:
            for key in update_diff:
                # Set initial increase history based on whether updates[key] is non-negative
                self.increase_history[client_id][key] = (update_diff[key] >= 0).float()
            # Return mask with all ones for epoch 0
            return {key: torch.ones_like(val) for key, val in update_diff.items()}

        # Generate mask based on consistency calculations
        mask = {}
        for key in update_diff:
            positive_consistency = self.increase_history[client_id][key]
            negative_consistency = 1 - self.increase_history[client_id][key]
            
            # Choose consistency based on update sign
            consistency = torch.where(update_diff[key] >= 0, positive_consistency, negative_consistency)
            
            # Apply threshold to generate mask
            mask[key] = (consistency > self.args.threshold).float()

        # Update increase history for each key
        for key in update_diff:
            increase = (update_diff[key] >= 0).float()
            # Update increase history using exponential moving average
            self.increase_history[client_id][key] = (
                self.increase_history[client_id][key] * self.epoch_index + increase
            ) / (self.epoch_index + 1)
        
        return mask

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Pariticipant %d loss = %0.3f" % (index, loss)
                optimizer.step()
