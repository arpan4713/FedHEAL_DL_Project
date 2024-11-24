import os
import sys
import torch
import warnings
import numpy as np
from argparse import ArgumentParser
from models import get_all_models, get_model
from datasets import get_prive_dataset
from utils.args import add_management_args
from utils.training import train
from utils.conf import set_random_seed
from utils.checkpoints import save_checkpoint, load_checkpoint
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")

# Checkpoints directory
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Clustering settings
NUM_CLUSTERS = 3
RECLUSTER_INTERVAL = 5

def parse_args():
    parser = ArgumentParser(description='FedHEAL Training', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=0, help='GPU Device ID')
    parser.add_argument('--communication_epoch', type=int, default=200, help='Number of communication rounds')
    parser.add_argument('--local_epoch', type=int, default=10, help='Local epochs for each client')
    parser.add_argument('--local_lr', type=float, default=0.01, help='Local learning rate for clients')
    parser.add_argument('--parti_num', type=int, default=20, help='Number of participants')
    parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models(), help='Model name')
    parser.add_argument('--dataset', type=str, default='fl_digits', help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--rand_dataset', type=int, default=1, help='Random dataset seed')
    parser.add_argument('--structure', type=str, default='homogeneity', help='Data distribution structure')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for Dirichlet sampling')
    parser.add_argument('--online_ratio', type=float, default=1.0, help='Ratio of online clients')
    parser.add_argument('--learning_decay', type=int, default=0, help='Learning rate decay option')
    parser.add_argument('--averaging', type=str, default='weight', help='Averaging strategy option')
    parser.add_argument('--wHEAL', type=int, default=1, help='Enable HEAL mechanism')
    parser.add_argument('--threshold', type=float, default=0.3, help='HEAL threshold')
    parser.add_argument('--beta', type=float, default=0.4, help='Momentum update beta')
    add_management_args(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    return args

def dynamic_clustering(round_num, client_data, num_clusters):
    if round_num % RECLUSTER_INTERVAL == 0:
        similarity_matrix = cosine_similarity(client_data)
        clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
        cluster_labels = clustering.fit_predict(similarity_matrix)
        print(f"Clustering performed at round {round_num}: {cluster_labels}")
        return cluster_labels
    return None

def train_clustered(model, priv_dataset, args):
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    client_data = [np.random.rand(100) for _ in range(args.parti_num)]
    optimizer = torch.optim.SGD(model.parameters(), lr=args.local_lr, momentum=0.9)

    start_round = load_checkpoint(model, optimizer, args, CHECKPOINT_DIR)
    for round_num in range(start_round, args.communication_epoch + 1):
        print(f"Starting round {round_num}...")
        cluster_labels = dynamic_clustering(round_num, client_data, NUM_CLUSTERS)
        if cluster_labels is not None:
            print(f"Clusters formed: {cluster_labels}")
        train(model, priv_dataset, args)
        save_checkpoint(round_num, model, optimizer, args, CHECKPOINT_DIR)
        print(f"Round {round_num} completed.\n")

def main():
    args = parse_args()
    priv_dataset = get_prive_dataset(args)
    backbone_list = priv_dataset.get_backbone(args.parti_num, None)
    model = get_model(backbone_list, args, priv_dataset.get_transform())

    if not list(model.parameters()):
        raise ValueError("The model has no trainable parameters. Check the model initialization.")

    train_clustered(model, priv_dataset, args)

if __name__ == '__main__':
    main()




# import os
# import sys
# import socket
# import torch
# import torch.multiprocessing
# import warnings
# import numpy as np
# import uuid
# import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# import matplotlib.pyplot as plt
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models
# from argparse import ArgumentParser
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from models import get_model
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed

# torch.multiprocessing.set_sharing_strategy('file_system')
# warnings.filterwarnings("ignore")

# # Paths and configs
# CHECKPOINT_DIR = "./checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# # Constants for clustering
# NUM_CLUSTERS = 3
# RECLUSTER_INTERVAL = 5

# # Adaptive learning rate scheduler
# def get_lr_scheduler(optimizer):
#     return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# # Save checkpoints
# def save_checkpoint(round_num, model, optimizer, args):
#     checkpoint = {
#         'round_num': round_num,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'args': vars(args)
#     }
#     checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_round_{round_num}.pth")
#     torch.save(checkpoint, checkpoint_path)
#     print(f"Checkpoint saved: {checkpoint_path}")

# # Load checkpoints
# def load_checkpoint(model, optimizer, args):
#     checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint")])
#     if not checkpoint_files:
#         print("No checkpoint found. Starting training from scratch.")
#         return 1
#     latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
#     checkpoint = torch.load(latest_checkpoint)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     for key, value in checkpoint['args'].items():
#         setattr(args, key, value)
#     print(f"Resuming from checkpoint: {latest_checkpoint}")
#     return checkpoint['round_num']

# # Perform clustering
# def perform_clustering(round_num, client_data):
#     if round_num % RECLUSTER_INTERVAL == 0:
#         print(f"Performing clustering at round {round_num}...")
#         similarity_matrix = cosine_similarity(client_data)
#         clustering = SpectralClustering(n_clusters=NUM_CLUSTERS, affinity='precomputed', random_state=42)
#         cluster_labels = clustering.fit_predict(similarity_matrix)
#         print(f"Clusters formed: {cluster_labels}")
#         return cluster_labels
#     return None

# # Cluster-aware training
# def train_clustered(model, priv_dataset, args):
#     client_data = [np.random.rand(100) for _ in range(args.parti_num)]
#     optimizer = torch.optim.SGD(model.parameters(), lr=args.local_lr, momentum=0.9)
#     lr_scheduler = get_lr_scheduler(optimizer)
#     start_round = load_checkpoint(model, optimizer, args)
#     for round_num in range(start_round, args.communication_epoch + 1):
#         print(f"Round {round_num} starting...")
#         cluster_labels = perform_clustering(round_num, client_data)
#         if cluster_labels is not None:
#             # Cluster-wise aggregation (placeholder logic for demonstration)
#             print("Performing cluster-wise aggregation...")
#         train(model, priv_dataset, args)
#         save_checkpoint(round_num, model, optimizer, args)
#         lr_scheduler.step()
#         print(f"Round {round_num} completed.\n")

# # Evaluation functions (unchanged from original script)
# def evaluate_model(model, test_loader, device):
#     model.eval()
#     y_true, y_scores = [], []
#     with torch.no_grad():
#         for data, labels in test_loader:
#             data, labels = data.to(device), labels.to(device)
#             outputs = model(data)
#             probabilities = torch.softmax(outputs, dim=1)[:, 1]
#             y_scores.extend(probabilities.cpu().numpy())
#             y_true.extend(labels.cpu().numpy())
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     return {
#         'fpr': fpr, 'tpr': tpr, 'roc_auc': auc(fpr, tpr),
#         'precision': precision, 'recall': recall, 'pr_auc': average_precision_score(y_true, y_scores)
#     }

# def plot_metrics(metrics):
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(metrics['fpr'], metrics['tpr'], label=f"ROC Curve (AUC = {metrics['roc_auc']:.2f})")
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(metrics['recall'], metrics['precision'], label=f"PR Curve (AP = {metrics['pr_auc']:.2f})")
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def parse_args():
#     parser = ArgumentParser(description='Improved FedAvg Training', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0, help='Device ID')
#     parser.add_argument('--communication_epoch', type=int, default=200, help='Communication epochs')
#     parser.add_argument('--local_epoch', type=int, default=10, help='Local epochs')
#     parser.add_argument('--local_lr', type=float, default=0.01, help='Local learning rate')  # Fixed missing argument
#     parser.add_argument('--parti_num', type=int, default=20, help='Number of participants')
#     parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models(), help='Model name')
#     parser.add_argument('--dataset', type=str, default='fl_digits', choices=DATASET_NAMES, help='Dataset name')
#     parser.add_argument('--seed', type=int, default=0, help='Random seed')
#     parser.add_argument('--rand_dataset', type=int, default=0, help='Random dataset seed')
#     parser.add_argument('--structure', type=str, default='homogeneity', help='Data distribution structure')
#     parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for Dirichlet sampler')
#     parser.add_argument('--online_ratio', type=float, default=1.0, help='Ratio of online clients')
#     parser.add_argument('--learning_decay', type=int, default=0, help='Learning rate decay option')
#     parser.add_argument('--averaging', type=str, default='weight', help='Averaging strategy option')
#     parser.add_argument('--wHEAL', type=int, default=1, help='Enable HEAL mechanism')
#     parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for HEAL')
#     parser.add_argument('--beta', type=float, default=0.4, help='Momentum update beta')
#     add_management_args(parser)
#     args = parser.parse_args()
#     set_random_seed(args.seed)
#     return args

# def main(args=None):
#     if args is None:
#         args = parse_args()
#     priv_dataset = get_prive_dataset(args)
#     model = get_model(priv_dataset.get_backbone(args.parti_num, None), args, priv_dataset.get_transform())
#     print(f"Model parameters: {list(model.parameters())}")  # Debugging print
#     train_clustered(model, priv_dataset, args)
#     device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     evaluate_and_plot(model, priv_dataset.get_test_loader(), device)

# if __name__ == '__main__':
#     main()





# import os
# import sys
# import socket
# import torch
# import torch.multiprocessing
# import warnings
# import numpy as np
# import uuid
# import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# import matplotlib.pyplot as plt
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models
# from argparse import ArgumentParser
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from models import get_model
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed

# torch.multiprocessing.set_sharing_strategy('file_system')
# warnings.filterwarnings("ignore")

# # Paths and configs
# CHECKPOINT_DIR = "./checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# # Constants for clustering
# NUM_CLUSTERS = 3
# RECLUSTER_INTERVAL = 5

# # Adaptive learning rate scheduler
# def get_lr_scheduler(optimizer):
#     return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# # Save checkpoints
# def save_checkpoint(round_num, model, optimizer, args):
#     checkpoint = {
#         'round_num': round_num,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'args': vars(args)
#     }
#     checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_round_{round_num}.pth")
#     torch.save(checkpoint, checkpoint_path)
#     print(f"Checkpoint saved: {checkpoint_path}")

# # Load checkpoints
# def load_checkpoint(model, optimizer, args):
#     checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint")])
#     if not checkpoint_files:
#         print("No checkpoint found. Starting training from scratch.")
#         return 1
#     latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
#     checkpoint = torch.load(latest_checkpoint)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     for key, value in checkpoint['args'].items():
#         setattr(args, key, value)
#     print(f"Resuming from checkpoint: {latest_checkpoint}")
#     return checkpoint['round_num']

# # Perform clustering
# def perform_clustering(round_num, client_data):
#     if round_num % RECLUSTER_INTERVAL == 0:
#         print(f"Performing clustering at round {round_num}...")
#         similarity_matrix = cosine_similarity(client_data)
#         clustering = SpectralClustering(n_clusters=NUM_CLUSTERS, affinity='precomputed', random_state=42)
#         cluster_labels = clustering.fit_predict(similarity_matrix)
#         print(f"Clusters formed: {cluster_labels}")
#         return cluster_labels
#     return None

# # Cluster-aware training
# def train_clustered(model, priv_dataset, args):
#     client_data = [np.random.rand(100) for _ in range(args.parti_num)]
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#     lr_scheduler = get_lr_scheduler(optimizer)
#     start_round = load_checkpoint(model, optimizer, args)
#     for round_num in range(start_round, args.communication_epoch + 1):
#         print(f"Round {round_num} starting...")
#         cluster_labels = perform_clustering(round_num, client_data)
#         if cluster_labels is not None:
#             # Cluster-wise aggregation (placeholder logic for demonstration)
#             print("Performing cluster-wise aggregation...")
#         train(model, priv_dataset, args)
#         save_checkpoint(round_num, model, optimizer, args)
#         lr_scheduler.step()
#         print(f"Round {round_num} completed.\n")

# # Evaluation functions (unchanged from original script)
# def evaluate_model(model, test_loader, device):
#     model.eval()
#     y_true, y_scores = [], []
#     with torch.no_grad():
#         for data, labels in test_loader:
#             data, labels = data.to(device), labels.to(device)
#             outputs = model(data)
#             probabilities = torch.softmax(outputs, dim=1)[:, 1]
#             y_scores.extend(probabilities.cpu().numpy())
#             y_true.extend(labels.cpu().numpy())
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     return {
#         'fpr': fpr, 'tpr': tpr, 'roc_auc': auc(fpr, tpr),
#         'precision': precision, 'recall': recall, 'pr_auc': average_precision_score(y_true, y_scores)
#     }

# def plot_metrics(metrics):
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(metrics['fpr'], metrics['tpr'], label=f"ROC Curve (AUC = {metrics['roc_auc']:.2f})")
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(metrics['recall'], metrics['precision'], label=f"PR Curve (AP = {metrics['pr_auc']:.2f})")
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def parse_args():
#     parser = ArgumentParser(description='Improved FedAvg Training', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0, help='Device ID')
#     parser.add_argument('--communication_epoch', type=int, default=200, help='Communication epochs')
#     parser.add_argument('--local_epoch', type=int, default=10, help='Local epochs')
#     parser.add_argument('--local_lr', type=float, default=0.01, help='Local learning rate')  # <-- Add this line
#     parser.add_argument('--parti_num', type=int, default=20, help='Number of participants')
#     parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models(), help='Model name')
#     parser.add_argument('--dataset', type=str, default='fl_digits', choices=DATASET_NAMES, help='Dataset name')
#     parser.add_argument('--seed', type=int, default=0, help='Random seed')
#     parser.add_argument('--rand_dataset', type=int, default=0, help='Random dataset seed')
#     parser.add_argument('--structure', type=str, default='homogeneity', help='Data distribution structure')
#     parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for Dirichlet sampler')
#     parser.add_argument('--online_ratio', type=float, default=1.0, help='Ratio of online clients')
#     parser.add_argument('--learning_decay', type=int, default=0, help='Learning rate decay option')
#     parser.add_argument('--averaging', type=str, default='weight', help='Averaging strategy option')
#     parser.add_argument('--wHEAL', type=int, default=1, help='Enable HEAL mechanism')
#     parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for HEAL')
#     parser.add_argument('--beta', type=float, default=0.4, help='Momentum update beta')
#     add_management_args(parser)
#     args = parser.parse_args()
#     set_random_seed(args.seed)
#     return args



# def main(args=None):
#     if args is None:
#         args = parse_args()
#     priv_dataset = get_prive_dataset(args)
#     model = get_model(priv_dataset.get_backbone(args.parti_num, None), args, priv_dataset.get_transform())
#     train_clustered(model, priv_dataset, args)
#     device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     evaluate_and_plot(model, priv_dataset.get_test_loader(), device)

# if __name__ == '__main__':
#     main()





# import os
# import sys
# import socket
# import torch
# import torch.multiprocessing
# import warnings
# import numpy as np
# import uuid
# import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# import matplotlib.pyplot as plt
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models
# from argparse import ArgumentParser
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from models import get_model
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed

# torch.multiprocessing.set_sharing_strategy('file_system')
# warnings.filterwarnings("ignore")

# # Config paths
# conf_path = os.getcwd()
# sys.path.append(conf_path)
# sys.path.append(conf_path + '/datasets')
# sys.path.append(conf_path + '/backbone')
# sys.path.append(conf_path + '/models')

# # Clustering configuration
# NUM_CLUSTERS = 3
# RECLUSTER_INTERVAL = 5
# CHECKPOINT_DIR = "./checkpoints"

# # Ensure checkpoint directory exists
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# def save_checkpoint(round_num, model, args):
#     """
#     Save the current training state as a checkpoint.
#     Args:
#         round_num (int): Current communication round.
#         model: Federated learning model.
#         args: Command-line arguments and other metadata.
#     """
#     checkpoint = {
#         'round_num': round_num,
#         'model_state_dict': model.state_dict(),
#         'args': vars(args)
#     }
#     checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_round_{round_num}.pth")
#     torch.save(checkpoint, checkpoint_path)
#     print(f"Checkpoint saved: {checkpoint_path}")

# def load_checkpoint(model, args):
#     """
#     Load the latest checkpoint to resume training.
#     Args:
#         model: Federated learning model.
#         args: Command-line arguments.
#     Returns:
#         int: The communication round to resume from.
#     """
#     checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint")])
#     if not checkpoint_files:
#         print("No checkpoint found. Starting training from scratch.")
#         return 1  # Start from round 1 if no checkpoint exists

#     latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
#     checkpoint = torch.load(latest_checkpoint)

#     model.load_state_dict(checkpoint['model_state_dict'])
#     for key, value in checkpoint['args'].items():
#         setattr(args, key, value)

#     print(f"Resuming from checkpoint: {latest_checkpoint}")
#     return checkpoint['round_num']

# def calculate_similarity_matrix(client_data):
#     return cosine_similarity(client_data)

# def perform_spectral_clustering(similarity_matrix, num_clusters):
#     clustering = SpectralClustering(
#         n_clusters=num_clusters,
#         affinity='precomputed',
#         random_state=42
#     )
#     return clustering.fit_predict(similarity_matrix)

# def dynamic_clustering(round_num, client_data, num_clusters):
#     if round_num % RECLUSTER_INTERVAL == 0:
#         print(f"Performing clustering at round {round_num}...")
#         similarity_matrix = calculate_similarity_matrix(client_data)
#         cluster_labels = perform_spectral_clustering(similarity_matrix, num_clusters)
#         print(f"Cluster labels: {cluster_labels}")
#         return cluster_labels
#     return None

# def get_client_data(participants):
#     return [np.random.rand(100) for _ in range(participants)]

# def train_with_clustering(model, priv_dataset, args):
#     client_data = get_client_data(args.parti_num)

#     # Resume from last checkpoint if available
#     start_round = load_checkpoint(model, args)

#     for round_num in range(start_round, args.communication_epoch + 1):
#         print(f"Round {round_num} starting...")

#         # Dynamic clustering step
#         cluster_labels = dynamic_clustering(round_num, client_data, NUM_CLUSTERS)
#         if cluster_labels is not None:
#             print(f"Clusters formed: {cluster_labels}")

#         # Standard federated training
#         train(model, priv_dataset, args)

#         # Save checkpoint at the end of each round
#         save_checkpoint(round_num, model, args)

#         print(f"Round {round_num} completed.\n")

# def evaluate_model(model, test_loader, device):
#     """
#     Evaluate the model on the test dataset and calculate ROC-AUC and Precision-Recall metrics.
#     Args:
#         model: Trained global model.
#         test_loader: DataLoader for the test dataset.
#         device: Device to run the evaluation (CPU or GPU).
#     Returns:
#         dict: Contains ROC-AUC, PR-AUC, and other evaluation metrics.
#     """
#     model.eval()
#     y_true = []
#     y_scores = []
    
#     with torch.no_grad():
#         for data, labels in test_loader:
#             data, labels = data.to(device), labels.to(device)
#             outputs = model(data)
#             probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability for the positive class
#             y_scores.extend(probabilities.cpu().numpy())
#             y_true.extend(labels.cpu().numpy())
    
#     # Calculate metrics
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)
    
#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     pr_auc = average_precision_score(y_true, y_scores)
    
#     return {
#         'fpr': fpr,
#         'tpr': tpr,
#         'roc_auc': roc_auc,
#         'precision': precision,
#         'recall': recall,
#         'pr_auc': pr_auc
#     }

# def plot_metrics(metrics):
#     """
#     Plot ROC and Precision-Recall curves.
#     Args:
#         metrics (dict): Contains metrics such as fpr, tpr, precision, recall, etc.
#     """
#     # ROC Curve
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(metrics['fpr'], metrics['tpr'], label=f"ROC Curve (AUC = {metrics['roc_auc']:.2f})")
#     plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend()

#     # Precision-Recall Curve
#     plt.subplot(1, 2, 2)
#     plt.plot(metrics['recall'], metrics['precision'], label=f"PR Curve (AP = {metrics['pr_auc']:.2f})")
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def evaluate_and_plot(model, test_loader, device):
#     """
#     Evaluate the model and plot AUC-ROC and PR curves.
#     Args:
#         model: Trained global model.
#         test_loader: DataLoader for the test dataset.
#         device: Device to run evaluation (CPU or GPU).
#     """
#     metrics = evaluate_model(model, test_loader, device)
#     print(f"ROC AUC: {metrics['roc_auc']:.4f}")
#     print(f"PR AUC: {metrics['pr_auc']:.4f}")
#     plot_metrics(metrics)

# def parse_args():
#     parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
#     parser.add_argument('--communication_epoch', type=int, default=200, help='The Communication Epoch in Federated Learning')
#     parser.add_argument('--local_epoch', type=int, default=10, help='The Local Epoch for each Participant')
#     parser.add_argument('--parti_num', type=int, default=20, help='The Number for Participants')
#     parser.add_argument('--seed', type=int, default=0, help='The random seed.')
#     parser.add_argument('--rand_dataset', type=int, default=0, help='The random dataset seed.')
#     parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models(), help='Model name.')
#     parser.add_argument('--structure', type=str, default='homogeneity')
#     parser.add_argument('--dataset', type=str, default='fl_digits', choices=DATASET_NAMES, help='Dataset to use.')
#     parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for Dirichlet sampler.')
#     parser.add_argument('--online_ratio', type=float, default=1, help='Ratio of online clients.')
#     parser.add_argument('--learning_decay', type=int, default=0, help='Learning rate decay option.')
#     parser.add_argument('--averaging', type=str, default='weight', help='Averaging strategy option.')
#     parser.add_argument('--wHEAL', type=int, default=1, help='CORE of FedHEAL to add HEAL.')
#     parser.add_argument('--threshold', type=float, default=0.3, help='Threshold of HEAL.')
#     parser.add_argument('--beta', type=float, default=0.4, help='Momentum update beta.')

#     torch.set_num_threads(4)
#     add_management_args(parser)
#     args = parser.parse_args()

#     best = best_args[args.dataset][args.model]
#     for key, value in best.items():
#         setattr(args, key, value)

#     if args.seed is not None:
#         set_random_seed(args.seed)
#     return args

# def main(args=None):
#     if args is None:
#         args = parse_args()

#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()

#     priv_dataset = get_prive_dataset(args)
#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)
#     model = get_model(backbones_list, args, priv_dataset.get_transform())
#     args.arch = model.nets_list[0].name

#     print('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))

#     # Train the model
#     train_with_clustering(model, priv_dataset, args)

#     # Evaluate the model after training
#     test_loader = priv_dataset.get_test_loader()
#     device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     evaluate_and_plot(model, test_loader, device)

# if __name__ == '__main__':
#     main()





















# # main.py

# import os
# import sys
# import socket
# import torch
# import torch.multiprocessing
# import warnings
# import numpy as np
# import uuid
# import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# import matplotlib.pyplot as plt
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models
# from argparse import ArgumentParser
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from datasets.svhn import get_svhn_dataset  # Import SVHN-specific logic
# from models import get_model
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed

# # Initialize multiprocessing sharing strategy for PyTorch
# torch.multiprocessing.set_sharing_strategy('file_system')
# warnings.filterwarnings("ignore")

# # Configure paths
# conf_path = os.getcwd()
# sys.path.append(conf_path)
# sys.path.append(conf_path + '/datasets')
# sys.path.append(conf_path + '/backbone')
# sys.path.append(conf_path + '/models')

# # Constants for clustering and checkpointing
# NUM_CLUSTERS = 3
# RECLUSTER_INTERVAL = 5
# CHECKPOINT_DIR = "./checkpoints"

# # Ensure checkpoint directory exists
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# def save_checkpoint(round_num, model, args):
#     """
#     Save a model checkpoint.
#     """
#     checkpoint = {
#         'round_num': round_num,
#         'model_state_dict': model.state_dict(),
#         'args': vars(args)
#     }
#     checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_round_{round_num}.pth")
#     torch.save(checkpoint, checkpoint_path)
#     print(f"Checkpoint saved: {checkpoint_path}")


# def load_checkpoint(model, args):
#     """
#     Load the latest model checkpoint.
#     """
#     checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint")])
#     if not checkpoint_files:
#         print("No checkpoint found. Starting training from scratch.")
#         return 1

#     latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
#     checkpoint = torch.load(latest_checkpoint)

#     model.load_state_dict(checkpoint['model_state_dict'])
#     for key, value in checkpoint['args'].items():
#         setattr(args, key, value)

#     print(f"Resuming from checkpoint: {latest_checkpoint}")
#     return checkpoint['round_num']


# def calculate_similarity_matrix(client_data):
#     """
#     Calculate similarity matrix for clustering.
#     """
#     return cosine_similarity(client_data)


# def perform_spectral_clustering(similarity_matrix, num_clusters):
#     """
#     Perform spectral clustering on the similarity matrix.
#     """
#     clustering = SpectralClustering(
#         n_clusters=num_clusters,
#         affinity='precomputed',
#         random_state=42
#     )
#     return clustering.fit_predict(similarity_matrix)


# def dynamic_clustering(round_num, client_data, num_clusters):
#     """
#     Dynamically perform clustering at specified intervals.
#     """
#     if round_num % RECLUSTER_INTERVAL == 0:
#         print(f"Performing clustering at round {round_num}...")
#         similarity_matrix = calculate_similarity_matrix(client_data)
#         cluster_labels = perform_spectral_clustering(similarity_matrix, num_clusters)
#         print(f"Cluster labels: {cluster_labels}")
#         return cluster_labels
#     return None


# def get_client_data(participants):
#     """
#     Generate random client data (placeholder).
#     """
#     return [np.random.rand(100) for _ in range(participants)]


# def train_with_clustering(model, priv_dataset, args):
#     """
#     Train the model with dynamic clustering.
#     """
#     client_data = get_client_data(args.parti_num)

#     # Resume training from the last checkpoint if available
#     start_round = load_checkpoint(model, args)

#     for round_num in range(start_round, args.communication_epoch + 1):
#         print(f"Round {round_num} starting...")

#         # Perform dynamic clustering
#         cluster_labels = dynamic_clustering(round_num, client_data, NUM_CLUSTERS)
#         if cluster_labels is not None:
#             print(f"Clusters formed: {cluster_labels}")

#         # Train the model
#         train(model, priv_dataset, args)

#         # Save checkpoint
#         save_checkpoint(round_num, model, args)
#         print(f"Round {round_num} completed.\n")


# def evaluate_model(model, test_loader, device):
#     """
#     Evaluate the model and compute metrics.
#     """
#     model.eval()
#     y_true = []
#     y_scores = []

#     with torch.no_grad():
#         for data, labels in test_loader:
#             data, labels = data.to(device), labels.to(device)
#             outputs = model(data)
#             probabilities = torch.softmax(outputs, dim=1)[:, 1]
#             y_scores.extend(probabilities.cpu().numpy())
#             y_true.extend(labels.cpu().numpy())

#     # Compute ROC and PR metrics
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)

#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     pr_auc = average_precision_score(y_true, y_scores)

#     return {
#         'fpr': fpr,
#         'tpr': tpr,
#         'roc_auc': roc_auc,
#         'precision': precision,
#         'recall': recall,
#         'pr_auc': pr_auc
#     }


# def plot_metrics(metrics):
#     """
#     Plot ROC and PR curves.
#     """
#     plt.figure(figsize=(10, 5))

#     # ROC Curve
#     plt.subplot(1, 2, 1)
#     plt.plot(metrics['fpr'], metrics['tpr'], label=f"ROC Curve (AUC = {metrics['roc_auc']:.2f})")
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend()

#     # Precision-Recall Curve
#     plt.subplot(1, 2, 2)
#     plt.plot(metrics['recall'], metrics['precision'], label=f"PR Curve (AP = {metrics['pr_auc']:.2f})")
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# def parse_args():
#     """
#     Parse command-line arguments.
#     """
#     parser = ArgumentParser(description='FedHEAL Federated Learning', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0, help='Device ID for experiment')
#     parser.add_argument('--communication_epoch', type=int, default=200, help='Number of communication epochs')
#     parser.add_argument('--local_epoch', type=int, default=10, help='Number of local epochs')
#     parser.add_argument('--parti_num', type=int, default=20, help='Number of participants')
#     parser.add_argument('--seed', type=int, default=0, help='Random seed')
#     parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models(), help='Model name')

#     # Correctly merge DATASET_NAMES keys with 'svhn'
#     available_datasets = list(DATASET_NAMES.keys()) + ['svhn']
#     parser.add_argument('--dataset', type=str, default='fl_digits', choices=available_datasets, help='Dataset')

#     torch.set_num_threads(4)
#     add_management_args(parser)
#     args = parser.parse_args()

#     best = best_args[args.dataset][args.model]
#     for key, value in best.items():
#         setattr(args, key, value)

#     if args.seed is not None:
#         set_random_seed(args.seed)

#     return args



# def main(args=None):
#     """
#     Main entry point.
#     """
#     if args is None:
#         args = parse_args()

#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()

#     if args.dataset == 'svhn':
#         priv_dataset = get_svhn_dataset(args)
#     else:
#         priv_dataset = get_prive_dataset(args)

#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)
#     model = get_model(backbones_list, args, priv_dataset.get_transform())
#     args.arch = model.nets_list[0].name

#     print(f"Training {args.model} on {args.dataset} with {args.parti_num} participants")

#     # Train the model
#     train_with_clustering(model, priv_dataset, args)

#     # Evaluate the model
#     test_loader = priv_dataset.get_test_loader()
#     device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     metrics = evaluate_model(model, test_loader, device)
#     plot_metrics(metrics)


# if __name__ == '__main__':
#     main()





# # File path: main_fl_svhn.py

# import os
# import sys
# import socket
# import torch
# import torch.multiprocessing
# import warnings
# import numpy as np
# import uuid
# import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# import matplotlib.pyplot as plt
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models
# from argparse import ArgumentParser
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from datasets.svhn import get_svhn_dataset  # Importing SVHN-specific functionality
# from models import get_model
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed

# torch.multiprocessing.set_sharing_strategy('file_system')
# warnings.filterwarnings("ignore")

# # Config paths
# conf_path = os.getcwd()
# sys.path.append(conf_path)
# sys.path.append(conf_path + '/datasets')
# sys.path.append(conf_path + '/backbone')
# sys.path.append(conf_path + '/models')

# # Clustering configuration
# NUM_CLUSTERS = 3
# RECLUSTER_INTERVAL = 5
# CHECKPOINT_DIR = "./checkpoints"

# os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# def save_checkpoint(round_num, model, args):
#     checkpoint = {
#         'round_num': round_num,
#         'model_state_dict': model.state_dict(),
#         'args': vars(args)
#     }
#     checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_round_{round_num}.pth")
#     torch.save(checkpoint, checkpoint_path)
#     print(f"Checkpoint saved: {checkpoint_path}")


# def load_checkpoint(model, args):
#     checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint")])
#     if not checkpoint_files:
#         print("No checkpoint found. Starting training from scratch.")
#         return 1

#     latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
#     checkpoint = torch.load(latest_checkpoint)

#     model.load_state_dict(checkpoint['model_state_dict'])
#     for key, value in checkpoint['args'].items():
#         setattr(args, key, value)

#     print(f"Resuming from checkpoint: {latest_checkpoint}")
#     return checkpoint['round_num']


# def calculate_similarity_matrix(client_data):
#     return cosine_similarity(client_data)


# def perform_spectral_clustering(similarity_matrix, num_clusters):
#     clustering = SpectralClustering(
#         n_clusters=num_clusters,
#         affinity='precomputed',
#         random_state=42
#     )
#     return clustering.fit_predict(similarity_matrix)


# def dynamic_clustering(round_num, client_data, num_clusters):
#     if round_num % RECLUSTER_INTERVAL == 0:
#         print(f"Performing clustering at round {round_num}...")
#         similarity_matrix = calculate_similarity_matrix(client_data)
#         cluster_labels = perform_spectral_clustering(similarity_matrix, num_clusters)
#         print(f"Cluster labels: {cluster_labels}")
#         return cluster_labels
#     return None


# def get_client_data(participants):
#     return [np.random.rand(100) for _ in range(participants)]


# def train_with_clustering(model, priv_dataset, args):
#     client_data = get_client_data(args.parti_num)

#     # Resume from last checkpoint if available
#     start_round = load_checkpoint(model, args)

#     for round_num in range(start_round, args.communication_epoch + 1):
#         print(f"Round {round_num} starting...")

#         # Dynamic clustering step
#         cluster_labels = dynamic_clustering(round_num, client_data, NUM_CLUSTERS)
#         if cluster_labels is not None:
#             print(f"Clusters formed: {cluster_labels}")

#         # Standard federated training
#         train(model, priv_dataset, args)

#         # Save checkpoint at the end of each round
#         save_checkpoint(round_num, model, args)

#         print(f"Round {round_num} completed.\n")


# def evaluate_model(model, test_loader, device):
#     model.eval()
#     y_true = []
#     y_scores = []

#     with torch.no_grad():
#         for data, labels in test_loader:
#             data, labels = data.to(device), labels.to(device)
#             outputs = model(data)
#             probabilities = torch.softmax(outputs, dim=1)[:, 1]
#             y_scores.extend(probabilities.cpu().numpy())
#             y_true.extend(labels.cpu().numpy())

#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)

#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     pr_auc = average_precision_score(y_true, y_scores)

#     return {
#         'fpr': fpr,
#         'tpr': tpr,
#         'roc_auc': roc_auc,
#         'precision': precision,
#         'recall': recall,
#         'pr_auc': pr_auc
#     }


# def plot_metrics(metrics):
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(metrics['fpr'], metrics['tpr'], label=f"ROC Curve (AUC = {metrics['roc_auc']:.2f})")
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(metrics['recall'], metrics['precision'], label=f"PR Curve (AP = {metrics['pr_auc']:.2f})")
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# def evaluate_and_plot(model, test_loader, device):
#     metrics = evaluate_model(model, test_loader, device)
#     print(f"ROC AUC: {metrics['roc_auc']:.4f}")
#     print(f"PR AUC: {metrics['pr_auc']:.4f}")
#     plot_metrics(metrics)


# def parse_args():
#     parser = ArgumentParser(description='Federated Learning with Clustering', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0, help='Device Id for Experiment')
#     parser.add_argument('--communication_epoch', type=int, default=200, help='Communication Epochs in Federated Learning')
#     parser.add_argument('--local_epoch', type=int, default=10, help='Local Epochs for each Participant')
#     parser.add_argument('--parti_num', type=int, default=20, help='Number of Participants')
#     parser.add_argument('--seed', type=int, default=0, help='Random seed')
#     parser.add_argument('--rand_dataset', type=int, default=0, help='Random dataset seed')
#     parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models(), help='Model name')
#     parser.add_argument('--dataset', type=str, default='fl_digits', choices=DATASET_NAMES + ['svhn'], help='Dataset to use')
#     parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for Dirichlet sampler')
#     parser.add_argument('--online_ratio', type=float, default=1, help='Ratio of online clients')
#     parser.add_argument('--learning_decay', type=int, default=0, help='Learning rate decay option')
#     parser.add_argument('--averaging', type=str, default='weight', help='Averaging strategy')
#     parser.add_argument('--wHEAL', type=int, default=1, help='CORE of FedHEAL to add HEAL')
#     parser.add_argument('--threshold', type=float, default=0.3, help='Threshold of HEAL')
#     parser.add_argument('--beta', type=float, default=0.4, help='Momentum update beta')

#     torch.set_num_threads(4)
#     add_management_args(parser)
#     args = parser.parse_args()

#     best = best_args[args.dataset][args.model]
#     for key, value in best.items():
#         setattr(args, key, value)

#     if args.seed is not None:
#         set_random_seed(args.seed)
#     return args


# def main(args=None):
#     if args is None:
#         args = parse_args()

#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()

#     if args.dataset == 'svhn':
#         priv_dataset = get_svhn_dataset(args)
#     else:
#         priv_dataset = get_prive_dataset(args)

#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)
#     model = get_model(backbones_list, args, priv_dataset.get_transform())
#     args.arch = model.nets_list[0].name

#     print(f'{args.model}_{args.parti_num}_{args.dataset}_{args.communication_epoch}_{args.local_epoch}')

#     train_with_clustering(model, priv_dataset, args)

#     test_loader = priv_dataset.get_test_loader()
#     device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     evaluate_and_plot(model, test_loader, device)


# if __name__ == '__main__':
#     main()









# import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# import matplotlib.pyplot as plt
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models
# from argparse import ArgumentParser
# @@ -123,6 +125,83 @@ def train_with_clustering(model, priv_dataset, args):

#         print(f"Round {round_num} completed.\n")

# def evaluate_model(model, test_loader, device):
#     """
#     Evaluate the model on the test dataset and calculate ROC-AUC and Precision-Recall metrics.
#     Args:
#         model: Trained global model.
#         test_loader: DataLoader for the test dataset.
#         device: Device to run the evaluation (CPU or GPU).
#     Returns:
#         dict: Contains ROC-AUC, PR-AUC, and other evaluation metrics.
#     """
#     model.eval()
#     y_true = []
#     y_scores = []
    
#     with torch.no_grad():
#         for data, labels in test_loader:
#             data, labels = data.to(device), labels.to(device)
#             outputs = model(data)
#             probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability for the positive class
#             y_scores.extend(probabilities.cpu().numpy())
#             y_true.extend(labels.cpu().numpy())
    
#     # Calculate metrics
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)
    
#     precision, recall, _ = precision_recall_curve(y_true, y_scores)
#     pr_auc = average_precision_score(y_true, y_scores)
    
#     return {
#         'fpr': fpr,
#         'tpr': tpr,
#         'roc_auc': roc_auc,
#         'precision': precision,
#         'recall': recall,
#         'pr_auc': pr_auc
#     }
# def plot_metrics(metrics):
#     """
#     Plot ROC and Precision-Recall curves.
#     Args:
#         metrics (dict): Contains metrics such as fpr, tpr, precision, recall, etc.
#     """
#     # ROC Curve
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(metrics['fpr'], metrics['tpr'], label=f"ROC Curve (AUC = {metrics['roc_auc']:.2f})")
#     plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend()
#     # Precision-Recall Curve
#     plt.subplot(1, 2, 2)
#     plt.plot(metrics['recall'], metrics['precision'], label=f"PR Curve (AP = {metrics['pr_auc']:.2f})")
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
# def evaluate_and_plot(model, test_loader, device):
#     """
#     Evaluate the model and plot AUC-ROC and PR curves.
#     Args:
#         model: Trained global model.
#         test_loader: DataLoader for the test dataset.
#         device: Device to run evaluation (CPU or GPU).
#     """
#     metrics = evaluate_model(model, test_loader, device)
#     print(f"ROC AUC: {metrics['roc_auc']:.4f}")
#     print(f"PR AUC: {metrics['pr_auc']:.4f}")
#     plot_metrics(metrics)
# def parse_args():
#     parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
# @@ -169,13 +248,198 @@ def main(args=None):

#     print('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))

#     # Train the model
#     train_with_clustering(model, priv_dataset, args)

#     # Evaluate the model after training
#     test_loader = priv_dataset.get_test_loader()
#     device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     evaluate_and_plot(model, test_loader, device)
# if __name__ == '__main__':
#     main()




# import os
# import sys
# import socket
# import torch
# import torch.multiprocessing
# import warnings
# import numpy as np
# import uuid
# import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import silhouette_score
# from argparse import ArgumentParser
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models, get_model  # Ensure this is correctly implemented
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed

# torch.multiprocessing.set_sharing_strategy('file_system')
# warnings.filterwarnings("ignore")

# # Config paths
# conf_path = os.getcwd()
# sys.path.extend([conf_path, f"{conf_path}/datasets", f"{conf_path}/backbone", f"{conf_path}/models"])

# # Clustering and checkpointing configuration
# NUM_CLUSTERS = 3
# RECLUSTER_INTERVAL = 5
# CHECKPOINT_DIR = "./checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# # Debugging Logs
# def log_model_parameters(model):
#     params = list(model.parameters())
#     print(f"Model initialized with {len(params)} parameters.")
#     if not params:
#         raise ValueError("The model has no trainable parameters. Ensure it is properly initialized.")
#     else:
#         print("Model has trainable parameters.")

# def save_checkpoint(round_num, model, args):
#     checkpoint = {
#         'round_num': round_num,
#         'model_state_dict': model.state_dict(),
#         'args': vars(args)
#     }
#     checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_round_{round_num}.pth")
#     torch.save(checkpoint, checkpoint_path)
#     print(f"Checkpoint saved: {checkpoint_path}")

# def load_checkpoint(model, args):
#     checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint")])
#     if not checkpoint_files:
#         print("No checkpoint found. Starting from scratch.")
#         return 1
#     latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
#     checkpoint = torch.load(latest_checkpoint)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     for key, value in checkpoint['args'].items():
#         setattr(args, key, value)
#     print(f"Resuming from checkpoint: {latest_checkpoint}")
#     return checkpoint['round_num']

# def calculate_similarity_matrix(client_data):
#     return cosine_similarity(client_data)

# def perform_spectral_clustering(similarity_matrix, num_clusters):
#     clustering = SpectralClustering(
#         n_clusters=num_clusters,
#         affinity='precomputed',
#         random_state=42
#     )
#     return clustering.fit_predict(similarity_matrix)

# def advanced_clustering(client_data):
#     best_score = -1
#     best_labels = None

#     for k in range(2, 10):  # Test multiple cluster sizes
#         similarity_matrix = calculate_similarity_matrix(client_data)
#         labels = perform_spectral_clustering(similarity_matrix, num_clusters=k)
#         score = silhouette_score(similarity_matrix, labels)

#         if score > best_score:
#             best_score = score
#             best_labels = labels

#     print(f"Best clustering score: {best_score}")
#     return best_labels

# def dynamic_clustering(round_num, client_data, num_clusters):
#     if round_num % RECLUSTER_INTERVAL == 0:
#         print(f"Clustering at round {round_num}...")
#         return advanced_clustering(client_data)
#     return None

# def get_client_data(participants):
#     return [np.random.rand(100) for _ in range(participants)]

# def compute_gradients(model, data, learning_rate):
#     if not list(model.parameters()):
#         raise ValueError("Cannot compute gradients: Model has no trainable parameters.")
    
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#     model.train()
#     optimizer.zero_grad()

#     inputs, labels = torch.tensor(data), torch.randint(0, 2, (len(data),))
#     outputs = model(inputs)
#     loss = torch.nn.CrossEntropyLoss()(outputs, labels)
#     loss.backward()

#     gradients = [param.grad.clone() for param in model.parameters()]
#     return gradients

# def aggregate_gradients(model, gradients_list, cluster_labels):
#     num_clusters = max(cluster_labels) + 1
#     aggregated_gradients = []

#     for i, params in enumerate(zip(*gradients_list)):
#         cluster_weights = np.zeros(num_clusters)
#         for label in cluster_labels:
#             cluster_weights[label] += 1
#         cluster_weights /= len(cluster_labels)

#         weighted_avg_grad = sum(weight * grad[i] for weight, grad in zip(cluster_weights, gradients_list))
#         aggregated_gradients.append(weighted_avg_grad)

#     for param, grad in zip(model.parameters(), aggregated_gradients):
#         param.grad = grad

# def train_with_fedheal_enhancements(model, priv_dataset, args):
#     client_data = get_client_data(args.parti_num)
#     start_round = load_checkpoint(model, args)
#     learning_rates = np.full(args.parti_num, args.beta)

#     for round_num in range(start_round, args.communication_epoch + 1):
#         print(f"Round {round_num} starting...")

#         cluster_labels = dynamic_clustering(round_num, client_data, NUM_CLUSTERS)
#         if cluster_labels is not None:
#             print(f"Clusters formed: {cluster_labels}")

#         global_gradients = []
#         for client_id, data in enumerate(client_data):
#             client_performance = np.random.rand()  # Replace with actual performance evaluation
#             if client_performance > args.threshold:
#                 gradients = compute_gradients(model, data, learning_rate=learning_rates[client_id])
#                 global_gradients.append(gradients)
#             else:
#                 print(f"Client {client_id} skipped due to low performance.")

#         if global_gradients:
#             aggregate_gradients(model, global_gradients, cluster_labels)
#         else:
#             print("No client updates were included this round.")

#         save_checkpoint(round_num, model, args)
#         print(f"Round {round_num} completed.\n")

# def parse_args():
#     parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0)
#     parser.add_argument('--communication_epoch', type=int, default=200)
#     parser.add_argument('--local_epoch', type=int, default=10)
#     parser.add_argument('--parti_num', type=int, default=20)
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--rand_dataset', type=int, default=0)
#     parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models())
#     parser.add_argument('--structure', type=str, default='homogeneity')
#     parser.add_argument('--dataset', type=str, default='fl_digits', choices=DATASET_NAMES)
#     parser.add_argument('--alpha', type=float, default=0.5)
#     parser.add_argument('--online_ratio', type=float, default=1)
#     parser.add_argument('--learning_decay', type=int, default=0)
#     parser.add_argument('--averaging', type=str, default='weight')
#     parser.add_argument('--wHEAL', type=int, default=1)
#     parser.add_argument('--threshold', type=float, default=0.3)
#     parser.add_argument('--beta', type=float, default=0.4)
#     torch.set_num_threads(4)
#     add_management_args(parser)
#     args = parser.parse_args()
#     best = best_args[args.dataset][args.model]
#     for key, value in best.items():
#         setattr(args, key, value)
#     if args.seed is not None:
#         set_random_seed(args.seed)
#     return args

# def main(args=None):
#     if args is None:
#         args = parse_args()
#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()
#     priv_dataset = get_prive_dataset(args)
#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)
#     model = get_model(backbones_list, args, priv_dataset.get_transform())
#     log_model_parameters(model)  # Check if the model has trainable parameters
#     train_with_fedheal_enhancements(model, priv_dataset, args)
#     test_loader = priv_dataset.get_test_loader()
#     device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     evaluate_and_plot(model, test_loader, device)

# if __name__ == '__main__':
#     main()











# import os
# import sys
# import socket
# import torch
# import torch.multiprocessing
# import warnings
# import numpy as np
# import uuid
# import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# from argparse import ArgumentParser
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from models import get_model  # Ensure this is correctly implemented
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed

# torch.multiprocessing.set_sharing_strategy('file_system')
# warnings.filterwarnings("ignore")

# # Config paths
# conf_path = os.getcwd()
# sys.path.extend([
#     conf_path, f"{conf_path}/datasets", f"{conf_path}/backbone", f"{conf_path}/models"
# ])

# # Clustering and checkpointing configuration
# NUM_CLUSTERS = 3
# RECLUSTER_INTERVAL = 5
# CHECKPOINT_DIR = "./checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# # Debugging Logs
# def log_model_parameters(model):
#     params = list(model.parameters())
#     print(f"Model initialized with {len(params)} parameters.")
#     if not params:
#         raise ValueError("The model has no trainable parameters. Ensure it is properly initialized.")

# def save_checkpoint(round_num, model, args):
#     checkpoint = {
#         'round_num': round_num,
#         'model_state_dict': model.state_dict(),
#         'args': vars(args)
#     }
#     checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_round_{round_num}.pth")
#     torch.save(checkpoint, checkpoint_path)
#     print(f"Checkpoint saved: {checkpoint_path}")

# def load_checkpoint(model, args):
#     checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint")])
#     if not checkpoint_files:
#         print("No checkpoint found. Starting from scratch.")
#         return 1
#     latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
#     checkpoint = torch.load(latest_checkpoint)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     for key, value in checkpoint['args'].items():
#         setattr(args, key, value)
#     print(f"Resuming from checkpoint: {latest_checkpoint}")
#     return checkpoint['round_num']

# def calculate_similarity_matrix(client_data):
#     return cosine_similarity(client_data)

# def perform_spectral_clustering(similarity_matrix, num_clusters):
#     clustering = SpectralClustering(
#         n_clusters=num_clusters,
#         affinity='precomputed',
#         random_state=42
#     )
#     return clustering.fit_predict(similarity_matrix)

# def advanced_clustering(client_data):
#     best_score = -1
#     best_labels = None

#     for k in range(2, 10):  # Test multiple cluster sizes
#         similarity_matrix = calculate_similarity_matrix(client_data)
#         labels = perform_spectral_clustering(similarity_matrix, num_clusters=k)
#         score = silhouette_score(similarity_matrix, labels)

#         if score > best_score:
#             best_score = score
#             best_labels = labels

#     print(f"Best clustering score: {best_score}")
#     return best_labels

# def dynamic_clustering(round_num, client_data, num_clusters):
#     if round_num % RECLUSTER_INTERVAL == 0:
#         print(f"Clustering at round {round_num}...")
#         return advanced_clustering(client_data)
#     return None

# def get_client_data(participants):
#     return [np.random.rand(100) for _ in range(participants)]

# def compute_gradients(model, data, learning_rate):
#     if not list(model.parameters()):
#         raise ValueError("Cannot compute gradients: Model has no trainable parameters.")
    
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#     model.train()
#     optimizer.zero_grad()

#     inputs, labels = torch.tensor(data), torch.randint(0, 2, (len(data),))
#     outputs = model(inputs)
#     loss = torch.nn.CrossEntropyLoss()(outputs, labels)
#     loss.backward()

#     gradients = [param.grad.clone() for param in model.parameters()]
#     return gradients

# def aggregate_gradients(model, gradients_list, cluster_labels):
#     num_clusters = max(cluster_labels) + 1
#     aggregated_gradients = []

#     for i, params in enumerate(zip(*gradients_list)):
#         cluster_weights = np.zeros(num_clusters)
#         for label in cluster_labels:
#             cluster_weights[label] += 1
#         cluster_weights /= len(cluster_labels)

#         weighted_avg_grad = sum(weight * grad[i] for weight, grad in zip(cluster_weights, gradients_list))
#         aggregated_gradients.append(weighted_avg_grad)

#     for param, grad in zip(model.parameters(), aggregated_gradients):
#         param.grad = grad

# def train_with_fedheal_enhancements(model, priv_dataset, args):
#     client_data = get_client_data(args.parti_num)
#     start_round = load_checkpoint(model, args)
#     learning_rates = np.full(args.parti_num, args.beta)

#     for round_num in range(start_round, args.communication_epoch + 1):
#         print(f"Round {round_num} starting...")

#         cluster_labels = dynamic_clustering(round_num, client_data, NUM_CLUSTERS)
#         if cluster_labels is not None:
#             print(f"Clusters formed: {cluster_labels}")

#         global_gradients = []
#         for client_id, data in enumerate(client_data):
#             client_performance = np.random.rand()  # Replace with actual performance evaluation
#             if client_performance > args.threshold:
#                 gradients = compute_gradients(model, data, learning_rate=learning_rates[client_id])
#                 global_gradients.append(gradients)
#             else:
#                 print(f"Client {client_id} skipped due to low performance.")

#         if global_gradients:
#             aggregate_gradients(model, global_gradients, cluster_labels)
#         else:
#             print("No client updates were included this round.")

#         save_checkpoint(round_num, model, args)
#         print(f"Round {round_num} completed.\n")

# def parse_args():
#     parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0)
#     parser.add_argument('--communication_epoch', type=int, default=200)
#     parser.add_argument('--local_epoch', type=int, default=10)
#     parser.add_argument('--parti_num', type=int, default=20)
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--rand_dataset', type=int, default=0)
#     parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models())
#     parser.add_argument('--structure', type=str, default='homogeneity')
#     parser.add_argument('--dataset', type=str, default='fl_digits', choices=DATASET_NAMES)
#     parser.add_argument('--alpha', type=float, default=0.5)
#     parser.add_argument('--online_ratio', type=float, default=1)
#     parser.add_argument('--learning_decay', type=int, default=0)
#     parser.add_argument('--averaging', type=str, default='weight')
#     parser.add_argument('--wHEAL', type=int, default=1)
#     parser.add_argument('--threshold', type=float, default=0.3)
#     parser.add_argument('--beta', type=float, default=0.4)
#     torch.set_num_threads(4)
#     add_management_args(parser)
#     args = parser.parse_args()
#     best = best_args[args.dataset][args.model]
#     for key, value in best.items():
#         setattr(args, key, value)
#     if args.seed is not None:
#         set_random_seed(args.seed)
#     return args

# def main(args=None):
#     if args is None:
#         args = parse_args()
#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()
#     priv_dataset = get_prive_dataset(args)
#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)
#     model = get_model(backbones_list, args, priv_dataset.get_transform())
#     log_model_parameters(model)
#     train_with_fedheal_enhancements(model, priv_dataset, args)
#     test_loader = priv_dataset.get_test_loader()
#     device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     evaluate_and_plot(model, test_loader, device)

# if __name__ == '__main__':
#     main()










# import os
# import sys
# import socket
# import torch
# import torch.multiprocessing
# import warnings
# import numpy as np
# import uuid
# import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models
# from argparse import ArgumentParser
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from models import get_model
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed

# torch.multiprocessing.set_sharing_strategy('file_system')
# warnings.filterwarnings("ignore")

# # Config paths
# conf_path = os.getcwd()
# sys.path.extend([
#     conf_path, f"{conf_path}/datasets", f"{conf_path}/backbone", f"{conf_path}/models"
# ])

# # Clustering and checkpointing configuration
# NUM_CLUSTERS = 3
# RECLUSTER_INTERVAL = 5
# CHECKPOINT_DIR = "./checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# # Debug and Log Initialization
# def log_model_parameters(model):
#     params = list(model.parameters())
#     print(f"Model initialized with {len(params)} parameters.")
#     if not params:
#         raise ValueError("The model has no trainable parameters. Ensure it is properly initialized.")

# def save_checkpoint(round_num, model, args):
#     checkpoint = {
#         'round_num': round_num,
#         'model_state_dict': model.state_dict(),
#         'args': vars(args)
#     }
#     checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_round_{round_num}.pth")
#     torch.save(checkpoint, checkpoint_path)
#     print(f"Checkpoint saved: {checkpoint_path}")

# def load_checkpoint(model, args):
#     checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint")])
#     if not checkpoint_files:
#         print("No checkpoint found. Starting from scratch.")
#         return 1
#     latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
#     checkpoint = torch.load(latest_checkpoint)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     for key, value in checkpoint['args'].items():
#         setattr(args, key, value)
#     print(f"Resuming from checkpoint: {latest_checkpoint}")
#     return checkpoint['round_num']

# def calculate_similarity_matrix(client_data):
#     return cosine_similarity(client_data)

# def perform_spectral_clustering(similarity_matrix, num_clusters):
#     clustering = SpectralClustering(
#         n_clusters=num_clusters,
#         affinity='precomputed',
#         random_state=42
#     )
#     return clustering.fit_predict(similarity_matrix)

# def advanced_clustering(client_data):
#     best_score = -1
#     best_labels = None

#     for k in range(2, 10):  # Test multiple cluster sizes
#         similarity_matrix = calculate_similarity_matrix(client_data)
#         labels = perform_spectral_clustering(similarity_matrix, num_clusters=k)
#         score = silhouette_score(similarity_matrix, labels)

#         if score > best_score:
#             best_score = score
#             best_labels = labels

#     print(f"Best clustering score: {best_score}")
#     return best_labels

# def dynamic_clustering(round_num, client_data, num_clusters):
#     if round_num % RECLUSTER_INTERVAL == 0:
#         print(f"Clustering at round {round_num}...")
#         return advanced_clustering(client_data)
#     return None

# def get_client_data(participants):
#     return [np.random.rand(100) for _ in range(participants)]

# def compute_gradients(model, data, learning_rate):
#     if not list(model.parameters()):
#         raise ValueError("Cannot compute gradients: Model has no trainable parameters.")
    
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#     model.train()
#     optimizer.zero_grad()

#     inputs, labels = torch.tensor(data), torch.randint(0, 2, (len(data),))
#     outputs = model(inputs)
#     loss = torch.nn.CrossEntropyLoss()(outputs, labels)
#     loss.backward()

#     gradients = [param.grad.clone() for param in model.parameters()]
#     return gradients

# def aggregate_gradients(model, gradients_list, cluster_labels):
#     num_clusters = max(cluster_labels) + 1
#     aggregated_gradients = []

#     for i, params in enumerate(zip(*gradients_list)):
#         cluster_weights = np.zeros(num_clusters)
#         for label in cluster_labels:
#             cluster_weights[label] += 1
#         cluster_weights /= len(cluster_labels)

#         weighted_avg_grad = sum(weight * grad[i] for weight, grad in zip(cluster_weights, gradients_list))
#         aggregated_gradients.append(weighted_avg_grad)

#     for param, grad in zip(model.parameters(), aggregated_gradients):
#         param.grad = grad

# def train_with_fedheal_enhancements(model, priv_dataset, args):
#     client_data = get_client_data(args.parti_num)
#     start_round = load_checkpoint(model, args)
#     learning_rates = np.full(args.parti_num, args.beta)

#     for round_num in range(start_round, args.communication_epoch + 1):
#         print(f"Round {round_num} starting...")

#         cluster_labels = dynamic_clustering(round_num, client_data, NUM_CLUSTERS)
#         if cluster_labels is not None:
#             print(f"Clusters formed: {cluster_labels}")

#         global_gradients = []
#         for client_id, data in enumerate(client_data):
#             client_performance = np.random.rand()  # Replace with actual performance evaluation
#             if client_performance > args.threshold:
#                 gradients = compute_gradients(model, data, learning_rate=learning_rates[client_id])
#                 global_gradients.append(gradients)
#             else:
#                 print(f"Client {client_id} skipped due to low performance.")

#         if global_gradients:
#             aggregate_gradients(model, global_gradients, cluster_labels)
#         else:
#             print("No client updates were included this round.")

#         save_checkpoint(round_num, model, args)
#         print(f"Round {round_num} completed.\n")

# def parse_args():
#     parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0)
#     parser.add_argument('--communication_epoch', type=int, default=200)
#     parser.add_argument('--local_epoch', type=int, default=10)
#     parser.add_argument('--parti_num', type=int, default=20)
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--rand_dataset', type=int, default=0)
#     parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models())
#     parser.add_argument('--structure', type=str, default='homogeneity')
#     parser.add_argument('--dataset', type=str, default='fl_digits', choices=DATASET_NAMES)
#     parser.add_argument('--alpha', type=float, default=0.5)
#     parser.add_argument('--online_ratio', type=float, default=1)
#     parser.add_argument('--learning_decay', type=int, default=0)
#     parser.add_argument('--averaging', type=str, default='weight')
#     parser.add_argument('--wHEAL', type=int, default=1)
#     parser.add_argument('--threshold', type=float, default=0.3)
#     parser.add_argument('--beta', type=float, default=0.4)
#     torch.set_num_threads(4)
#     add_management_args(parser)
#     args = parser.parse_args()
#     best = best_args[args.dataset][args.model]
#     for key, value in best.items():
#         setattr(args, key, value)
#     if args.seed is not None:
#         set_random_seed(args.seed)
#     return args

# def main(args=None):
#     if args is None:
#         args = parse_args()
#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()
#     priv_dataset = get_prive_dataset(args)
#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)
#     model = get_model(backbones_list, args, priv_dataset.get_transform())
#     log_model_parameters(model)
#     train_with_fedheal_enhancements(model, priv_dataset, args)
#     test_loader = priv_dataset.get_test_loader()
#     device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     evaluate_and_plot(model, test_loader, device)

# if __name__ == '__main__':
#     main()








# import os
# import sys
# import socket
# import torch
# import torch.multiprocessing
# import warnings
# import numpy as np
# import uuid
# import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models
# from argparse import ArgumentParser
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from models import get_model
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed

# torch.multiprocessing.set_sharing_strategy('file_system')
# warnings.filterwarnings("ignore")

# # Config paths
# conf_path = os.getcwd()
# sys.path.extend([
#     conf_path, f"{conf_path}/datasets", f"{conf_path}/backbone", f"{conf_path}/models"
# ])

# # Clustering and checkpointing configuration
# NUM_CLUSTERS = 3
# RECLUSTER_INTERVAL = 5
# CHECKPOINT_DIR = "./checkpoints"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# def save_checkpoint(round_num, model, args):
#     checkpoint = {
#         'round_num': round_num,
#         'model_state_dict': model.state_dict(),
#         'args': vars(args)
#     }
#     checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_round_{round_num}.pth")
#     torch.save(checkpoint, checkpoint_path)
#     print(f"Checkpoint saved: {checkpoint_path}")

# def load_checkpoint(model, args):
#     checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint")])
#     if not checkpoint_files:
#         print("No checkpoint found. Starting from scratch.")
#         return 1
#     latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
#     checkpoint = torch.load(latest_checkpoint)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     for key, value in checkpoint['args'].items():
#         setattr(args, key, value)
#     print(f"Resuming from checkpoint: {latest_checkpoint}")
#     return checkpoint['round_num']

# def calculate_similarity_matrix(client_data):
#     return cosine_similarity(client_data)

# def perform_spectral_clustering(similarity_matrix, num_clusters):
#     clustering = SpectralClustering(
#         n_clusters=num_clusters,
#         affinity='precomputed',
#         random_state=42
#     )
#     return clustering.fit_predict(similarity_matrix)

# def advanced_clustering(client_data):
#     """
#     Advanced clustering with silhouette score optimization.
#     """
#     best_score = -1
#     best_labels = None

#     for k in range(2, 10):  # Test multiple cluster sizes
#         similarity_matrix = calculate_similarity_matrix(client_data)
#         labels = perform_spectral_clustering(similarity_matrix, num_clusters=k)
#         score = silhouette_score(similarity_matrix, labels)

#         if score > best_score:
#             best_score = score
#             best_labels = labels

#     print(f"Best clustering score: {best_score}")
#     return best_labels

# def dynamic_clustering(round_num, client_data, num_clusters):
#     if round_num % RECLUSTER_INTERVAL == 0:
#         print(f"Clustering at round {round_num}...")
#         return advanced_clustering(client_data)
#     return None

# def get_client_data(participants):
#     return [np.random.rand(100) for _ in range(participants)]

# def compute_gradients(model, data, learning_rate):
#     """
#     Compute gradients for a client's data.
#     """
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#     model.train()
#     optimizer.zero_grad()

#     # Mock loss computation
#     inputs, labels = torch.tensor(data), torch.randint(0, 2, (len(data),))
#     outputs = model(inputs)
#     loss = torch.nn.CrossEntropyLoss()(outputs, labels)
#     loss.backward()

#     gradients = [param.grad.clone() for param in model.parameters()]
#     return gradients

# def aggregate_gradients(model, gradients_list, cluster_labels):
#     """
#     Fair aggregation of gradients with potential cluster-based weighting.
#     """
#     num_clusters = max(cluster_labels) + 1
#     aggregated_gradients = []

#     for i, params in enumerate(zip(*gradients_list)):
#         cluster_weights = np.zeros(num_clusters)
#         for label in cluster_labels:
#             cluster_weights[label] += 1
#         cluster_weights /= len(cluster_labels)

#         weighted_avg_grad = sum(weight * grad[i] for weight, grad in zip(cluster_weights, gradients_list))
#         aggregated_gradients.append(weighted_avg_grad)

#     for param, grad in zip(model.parameters(), aggregated_gradients):
#         param.grad = grad

# def adaptive_learning_rate_adjustment(learning_rates, round_losses, decay_factor=0.9):
#     """
#     Adjust learning rates adaptively based on loss trends.
#     """
#     for i, loss in enumerate(round_losses):
#         if loss > np.mean(round_losses):  # Penalize poor-performing clients
#             learning_rates[i] *= decay_factor
#         else:
#             learning_rates[i] /= decay_factor
#     return learning_rates

# def evaluate_client_performance(model, client_data):
#     """
#     Mock performance evaluation for each client.
#     """
#     return np.random.rand()

# def train_with_fedheal_enhancements(model, priv_dataset, args):
#     """
#     Enhanced training loop with FedHEAL improvements and advanced techniques
#     for superior accuracy and fairness.
#     """
#     client_data = get_client_data(args.parti_num)
#     start_round = load_checkpoint(model, args)
#     learning_rates = np.full(args.parti_num, args.beta)  # Adaptive learning rates

#     for round_num in range(start_round, args.communication_epoch + 1):
#         print(f"Round {round_num} starting...")

#         # Dynamic clustering and similarity adjustments
#         cluster_labels = dynamic_clustering(round_num, client_data, NUM_CLUSTERS)
#         if cluster_labels is not None:
#             print(f"Clusters formed: {cluster_labels}")

#         # Train and aggregate with selective updates
#         global_gradients = []
#         for client_id, data in enumerate(client_data):
#             client_performance = evaluate_client_performance(model, data)
#             if client_performance > args.threshold:
#                 gradients = compute_gradients(model, data, learning_rate=learning_rates[client_id])
#                 global_gradients.append(gradients)
#             else:
#                 print(f"Client {client_id} skipped due to low performance.")

#         # Fair aggregation with regularization
#         if global_gradients:
#             aggregate_gradients(model, global_gradients, cluster_labels)
#         else:
#             print("No client updates were included this round.")

#         # Save checkpoint after each round
#         save_checkpoint(round_num, model, args)
#         print(f"Round {round_num} completed.\n")

# def parse_args():
#     parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0)
#     parser.add_argument('--communication_epoch', type=int, default=200)
#     parser.add_argument('--local_epoch', type=int, default=10)
#     parser.add_argument('--parti_num', type=int, default=20)
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--rand_dataset', type=int, default=0)
#     parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models())
#     parser.add_argument('--structure', type=str, default='homogeneity')
#     parser.add_argument('--dataset', type=str, default='fl_digits', choices=DATASET_NAMES)
#     parser.add_argument('--alpha', type=float, default=0.5)
#     parser.add_argument('--online_ratio', type=float, default=1)
#     parser.add_argument('--learning_decay', type=int, default=0)
#     parser.add_argument('--averaging', type=str, default='weight')
#     parser.add_argument('--wHEAL', type=int, default=1)
#     parser.add_argument('--threshold', type=float, default=0.3)
#     parser.add_argument('--beta', type=float, default=0.4)
#     torch.set_num_threads(4)
#     add_management_args(parser)
#     args = parser.parse_args()
#     best = best_args[args.dataset][args.model]
#     for key, value in best.items():
#         setattr(args, key, value)
#     if args.seed is not None:
#         set_random_seed(args.seed)
#     return args

# def main(args=None):
#     if args is None:
#         args = parse_args()
#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()
#     priv_dataset = get_prive_dataset(args)
#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)
#     model = get_model(backbones_list, args, priv_dataset.get_transform())
#     args.arch = model.nets_list[0].name
#     train_with_fedheal_enhancements(model, priv_dataset, args)
#     test_loader = priv_dataset.get_test_loader()
#     device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     evaluate_and_plot(model, test_loader, device)

# if __name__ == '__main__':
#     main()






# import os
# import sys
# import socket
# import torch
# import torch.multiprocessing
# import warnings
# import numpy as np
# import uuid
# import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import SpectralClustering
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models
# from argparse import ArgumentParser
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from models import get_model
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed

# torch.multiprocessing.set_sharing_strategy('file_system')
# warnings.filterwarnings("ignore")

# # Config paths
# conf_path = os.getcwd()
# sys.path.append(conf_path)
# sys.path.append(conf_path + '/datasets')
# sys.path.append(conf_path + '/backbone')
# sys.path.append(conf_path + '/models')

# # Clustering configuration
# NUM_CLUSTERS = 3
# RECLUSTER_INTERVAL = 5
# CHECKPOINT_DIR = "./checkpoints"

# # Ensure checkpoint directory exists
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# def save_checkpoint(round_num, model, args):
#     """
#     Save the current training state as a checkpoint.
#     Args:
#         round_num (int): Current communication round.
#         model: Federated learning model.
#         args: Command-line arguments and other metadata.
#     """
#     checkpoint = {
#         'round_num': round_num,
#         'model_state_dict': model.state_dict(),
#         'args': vars(args)
#     }
#     checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_round_{round_num}.pth")
#     torch.save(checkpoint, checkpoint_path)
#     print(f"Checkpoint saved: {checkpoint_path}")

# def load_checkpoint(model, args):
#     """
#     Load the latest checkpoint to resume training.
#     Args:
#         model: Federated learning model.
#         args: Command-line arguments.
#     Returns:
#         int: The communication round to resume from.
#     """
#     checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint")])
#     if not checkpoint_files:
#         print("No checkpoint found. Starting training from scratch.")
#         return 1  # Start from round 1 if no checkpoint exists

#     latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
#     checkpoint = torch.load(latest_checkpoint)

#     model.load_state_dict(checkpoint['model_state_dict'])
#     for key, value in checkpoint['args'].items():
#         setattr(args, key, value)

#     print(f"Resuming from checkpoint: {latest_checkpoint}")
#     return checkpoint['round_num']

# def calculate_similarity_matrix(client_data):
#     return cosine_similarity(client_data)

# def perform_spectral_clustering(similarity_matrix, num_clusters):
#     clustering = SpectralClustering(
#         n_clusters=num_clusters,
#         affinity='precomputed',
#         random_state=42
#     )
#     return clustering.fit_predict(similarity_matrix)

# def dynamic_clustering(round_num, client_data, num_clusters):
#     if round_num % RECLUSTER_INTERVAL == 0:
#         print(f"Performing clustering at round {round_num}...")
#         similarity_matrix = calculate_similarity_matrix(client_data)
#         cluster_labels = perform_spectral_clustering(similarity_matrix, num_clusters)
#         print(f"Cluster labels: {cluster_labels}")
#         return cluster_labels
#     return None

# def get_client_data(participants):
#     return [np.random.rand(100) for _ in range(participants)]

# def train_with_clustering(model, priv_dataset, args):
#     client_data = get_client_data(args.parti_num)

#     # Resume from last checkpoint if available
#     start_round = load_checkpoint(model, args)

#     for round_num in range(start_round, args.communication_epoch + 1):
#         print(f"Round {round_num} starting...")

#         # Dynamic clustering step
#         cluster_labels = dynamic_clustering(round_num, client_data, NUM_CLUSTERS)
#         if cluster_labels is not None:
#             print(f"Clusters formed: {cluster_labels}")

#         # Standard federated training
#         train(model, priv_dataset, args)

#         # Save checkpoint at the end of each round
#         save_checkpoint(round_num, model, args)

#         print(f"Round {round_num} completed.\n")

# def parse_args():
#     parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
#     parser.add_argument('--communication_epoch', type=int, default=200, help='The Communication Epoch in Federated Learning')
#     parser.add_argument('--local_epoch', type=int, default=10, help='The Local Epoch for each Participant')
#     parser.add_argument('--parti_num', type=int, default=20, help='The Number for Participants')
#     parser.add_argument('--seed', type=int, default=0, help='The random seed.')
#     parser.add_argument('--rand_dataset', type=int, default=0, help='The random dataset seed.')
#     parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models(), help='Model name.')
#     parser.add_argument('--structure', type=str, default='homogeneity')
#     parser.add_argument('--dataset', type=str, default='fl_digits', choices=DATASET_NAMES, help='Dataset to use.')
#     parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for Dirichlet sampler.')
#     parser.add_argument('--online_ratio', type=float, default=1, help='Ratio of online clients.')
#     parser.add_argument('--learning_decay', type=int, default=0, help='Learning rate decay option.')
#     parser.add_argument('--averaging', type=str, default='weight', help='Averaging strategy option.')
#     parser.add_argument('--wHEAL', type=int, default=1, help='CORE of FedHEAL to add HEAL.')
#     parser.add_argument('--threshold', type=float, default=0.3, help='Threshold of HEAL.')
#     parser.add_argument('--beta', type=float, default=0.4, help='Momentum update beta.')

#     torch.set_num_threads(4)
#     add_management_args(parser)
#     args = parser.parse_args()

#     best = best_args[args.dataset][args.model]
#     for key, value in best.items():
#         setattr(args, key, value)

#     if args.seed is not None:
#         set_random_seed(args.seed)
#     return args

# def main(args=None):
#     if args is None:
#         args = parse_args()

#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()

#     priv_dataset = get_prive_dataset(args)
#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)
#     model = get_model(backbones_list, args, priv_dataset.get_transform())
#     args.arch = model.nets_list[0].name

#     print('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))

#     train_with_clustering(model, priv_dataset, args)

# if __name__ == '__main__':
#     main()




# import os
# import sys
# import socket
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# import warnings
# import numpy as np
# import uuid
# import datetime

# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import SpectralClustering

# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models
# from argparse import ArgumentParser
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from models import get_model
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed

# warnings.filterwarnings("ignore")

# # Config paths
# conf_path = os.getcwd()
# sys.path.append(conf_path)
# sys.path.append(conf_path + '/datasets')
# sys.path.append(conf_path + '/backbone')
# sys.path.append(conf_path + '/models')

# # Clustering configuration
# NUM_CLUSTERS = 3  # Default number of clusters
# RECLUSTER_INTERVAL = 5  # Perform re-clustering every 5 communication rounds

# def calculate_similarity_matrix(client_data):
#     """
#     Calculate the cosine similarity matrix for given client data.
#     Args:
#         client_data (list of numpy arrays): Data representing client features.
#     Returns:
#         numpy.ndarray: Cosine similarity matrix.
#     """
#     return cosine_similarity(client_data)

# def perform_spectral_clustering(similarity_matrix, num_clusters):
#     """
#     Perform spectral clustering based on the similarity matrix.
#     Args:
#         similarity_matrix (numpy.ndarray): Precomputed similarity matrix.
#         num_clusters (int): Number of clusters.
#     Returns:
#         list: Cluster labels for each client.
#     """
#     clustering = SpectralClustering(
#         n_clusters=num_clusters,
#         affinity='precomputed',
#         random_state=42
#     )
#     return clustering.fit_predict(similarity_matrix)

# def dynamic_clustering(round_num, client_data, num_clusters):
#     """
#     Perform dynamic clustering periodically.
#     Args:
#         round_num (int): Current communication round number.
#         client_data (list of numpy arrays): Data representing client features.
#         num_clusters (int): Number of clusters.
#     Returns:
#         list: Cluster labels if clustering is performed, None otherwise.
#     """
#     if round_num % RECLUSTER_INTERVAL == 0:
#         print(f"Performing clustering at round {round_num}...")
#         similarity_matrix = calculate_similarity_matrix(client_data)
#         cluster_labels = perform_spectral_clustering(similarity_matrix, num_clusters)
#         print(f"Cluster labels: {cluster_labels}")
#         return cluster_labels
#     return None

# def get_client_data(participants):
#     """
#     Retrieve or simulate client data. Replace with actual client feature extraction logic.
#     Args:
#         participants (int): Number of participants.
#     Returns:
#         list of numpy arrays: Mock client feature data.
#     """
#     return [np.random.rand(100) for _ in range(participants)]

# def train_with_clustering(model, priv_dataset, args):
#     """
#     Main training loop with dynamic clustering integration.
#     Args:
#         model: The federated learning model.
#         priv_dataset: Dataset for training.
#         args: Command-line arguments.
#     """
#     client_data = get_client_data(args.parti_num)  # Initialize client data
#     for round_num in range(1, args.communication_epoch + 1):
#         print(f"Round {round_num} starting...")
        
#         # Dynamic clustering step
#         cluster_labels = dynamic_clustering(round_num, client_data, NUM_CLUSTERS)
#         if cluster_labels is not None:
#             print(f"Clusters formed: {cluster_labels}")
        
#         # Standard federated training
#         train(model, priv_dataset, args)

#         print(f"Round {round_num} completed.\n")

# def parse_args():
#     parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
#     parser.add_argument('--communication_epoch', type=int, default=200, help='The Communication Epoch in Federated Learning')
#     parser.add_argument('--local_epoch', type=int, default=10, help='The Local Epoch for each Participant')
#     parser.add_argument('--parti_num', type=int, default=20, help='The Number for Participants')
#     parser.add_argument('--seed', type=int, default=0, help='The random seed.')
#     parser.add_argument('--rand_dataset', type=int, default=0, help='The random dataset seed.')
#     parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models(), help='Model name.')
#     parser.add_argument('--structure', type=str, default='homogeneity')
#     parser.add_argument('--dataset', type=str, default='fl_digits', choices=DATASET_NAMES, help='Dataset to use.')
#     parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for Dirichlet sampler.')
#     parser.add_argument('--online_ratio', type=float, default=1, help='Ratio of online clients.')
#     parser.add_argument('--learning_decay', type=int, default=0, help='Learning rate decay option.')
#     parser.add_argument('--averaging', type=str, default='weight', help='Averaging strategy option.')
#     parser.add_argument('--wHEAL', type=int, default=1, help='CORE of FedHEAL to add HEAL.')
#     parser.add_argument('--threshold', type=float, default=0.3, help='Threshold of HEAL.')
#     parser.add_argument('--beta', type=float, default=0.4, help='Momentum update beta.')

#     # Client configuration
#     parser.add_argument('--mnist', type=int, default=5, help='Number of MNIST clients.')
#     parser.add_argument('--usps', type=int, default=5, help='Number of USPS clients.')
#     parser.add_argument('--svhn', type=int, default=5, help='Number of SVHN clients.')
#     parser.add_argument('--syn', type=int, default=5, help='Number of Syn clients.')
#     parser.add_argument('--caltech', type=int, default=5, help='Number of Caltech clients.')
#     parser.add_argument('--amazon', type=int, default=5, help='Number of Amazon clients.')
#     parser.add_argument('--webcam', type=int, default=5, help='Number of Webcam clients.')
#     parser.add_argument('--dslr', type=int, default=5, help='Number of DSLR clients.')

#     torch.set_num_threads(4)
#     add_management_args(parser)
#     args = parser.parse_args()

#     # Set default best args based on dataset and model
#     best = best_args[args.dataset][args.model]
#     for key, value in best.items():
#         setattr(args, key, value)

#     if args.seed is not None:
#         set_random_seed(args.seed)
#     return args

# def main(args=None):
#     if args is None:
#         args = parse_args()

#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()

#     priv_dataset = get_prive_dataset(args)
#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)
#     model = get_model(backbones_list, args, priv_dataset.get_transform())
#     args.arch = model.nets_list[0].name

#     print('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))

#     # Start training with clustering
#     train_with_clustering(model, priv_dataset, args)

# if __name__ == '__main__':
#     main()





















# import os
# import sys
# import socket
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# import warnings

# warnings.filterwarnings("ignore")

# conf_path = os.getcwd()
# sys.path.append(conf_path)
# sys.path.append(conf_path + '/datasets')
# sys.path.append(conf_path + '/backbone')
# sys.path.append(conf_path + '/models')
# from datasets import Priv_NAMES as DATASET_NAMES
# from models import get_all_models
# from argparse import ArgumentParser
# from utils.args import add_management_args
# from datasets import get_prive_dataset
# from models import get_model
# from utils.training import train
# from utils.best_args import best_args
# from utils.conf import set_random_seed
# # # ************** improvement code ********
# # from models import MyFederatedModel
# # from utils import calculate_consistency, aggregate_updates
# # # ******************************

# import torch
# import uuid
# import datetime



# def parse_args():
#     parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    
#     parser.add_argument('--communication_epoch', type=int, default=200, help='The Communication Epoch in Federated Learning')
#     parser.add_argument('--local_epoch', type=int, default=10, help='The Local Epoch for each Participant')
#     parser.add_argument('--parti_num', type=int, default=20, help='The Number for Participants')
#     parser.add_argument('--seed', type=int, default=0, help='The random seed.')
#     parser.add_argument('--rand_dataset', type=int, default=0, help='The random seed.')

#     parser.add_argument('--model', type=str, default='fedavgheal', 
#                         help='Model name.', choices=get_all_models())
#     parser.add_argument('--structure', type=str, default='homogeneity')
#     parser.add_argument('--dataset', type=str, default='fl_digits', 
#                         choices=DATASET_NAMES, help='Which scenario to perform experiments on.')
#     parser.add_argument('--alpha', type=float, default=0.5, help='alpha of dirichlet sampler.')
#     parser.add_argument('--online_ratio', type=float, default=1, help='The Ratio for Online Clients')
#     parser.add_argument('--learning_decay', type=int, default=0, help='The Option for Learning Rate Decay')
#     parser.add_argument('--averaging', type=str, default='weight', help='The Option for averaging strategy')

#     parser.add_argument('--wHEAL', type=int, default=1, help='The CORE of the FedHEAL decides whether to add HEAL to other FL method')
#     parser.add_argument('--threshold', type=float, default=0.3, help='threshold of HEAL')
#     parser.add_argument('--beta', type=float, default=0.4, help='momentum update beta')
     
#     parser.add_argument('--mnist', type=int, default=5, help='Number of mnist clients')
#     parser.add_argument('--usps', type=int, default=5, help='Number of usps clients')
#     parser.add_argument('--svhn', type=int, default=5, help='Number of svhn clients')
#     parser.add_argument('--syn', type=int, default=5, help='Number of syn clients')
    
#     parser.add_argument('--caltech', type=int, default=5, help='Number of caltech clients')
#     parser.add_argument('--amazon', type=int, default=5, help='Number of amazon clients')
#     parser.add_argument('--webcam', type=int, default=5, help='Number of webcam clients')
#     parser.add_argument('--dslr', type=int, default=5, help='Number of dslr clients')
    
#     torch.set_num_threads(4)
#     add_management_args(parser)
#     args = parser.parse_args()

#     best = best_args[args.dataset][args.model]

#     for key, value in best.items():
#         setattr(args, key, value)

#     if args.seed is not None:
#         set_random_seed(args.seed)
#     return args


# def main(args=None):
#     if args is None:
#         args = parse_args()

#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()

#     priv_dataset = get_prive_dataset(args)

#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)

#     model = get_model(backbones_list, args, priv_dataset.get_transform())
    
#     args.arch = model.nets_list[0].name

#     print('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))

#     train(model, priv_dataset, args)


# if __name__ == '__main__':
#     main()


 
# import os
# import torch

# def parse_args():
#     parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
#     parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
#     parser.add_argument('--communication_epoch', type=int, default=200, help='The Communication Epoch in Federated Learning')
#     parser.add_argument('--local_epoch', type=int, default=10, help='The Local Epoch for each Participant')
#     parser.add_argument('--parti_num', type=int, default=20, help='The Number for Participants')
#     parser.add_argument('--seed', type=int, default=0, help='The random seed.')
#     parser.add_argument('--rand_dataset', type=int, default=0, help='The random dataset seed.')
#     parser.add_argument('--checkpoint', type=str, default=None, help='Path to load checkpoint.')
#     parser.add_argument('--save_dir', type=str, default='checkpoints/', help='Directory to save checkpoints.')

#     # Other arguments as before...
#     torch.set_num_threads(4)
#     add_management_args(parser)
#     args = parser.parse_args()

#     best = best_args[args.dataset][args.model]
#     for key, value in best.items():
#         setattr(args, key, value)

#     if args.seed is not None:
#         set_random_seed(args.seed)

#     return args

# def save_checkpoint(state, save_dir, filename="checkpoint.pth.tar"):
#     os.makedirs(save_dir, exist_ok=True)
#     filepath = os.path.join(save_dir, filename)
#     torch.save(state, filepath)
#     print(f"Checkpoint saved to {filepath}")

# def load_checkpoint(checkpoint_path, model, optimizer=None):
#     if os.path.isfile(checkpoint_path):
#         print(f"Loading checkpoint from {checkpoint_path}")
#         checkpoint = torch.load(checkpoint_path)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         if optimizer and 'optimizer_state_dict' in checkpoint:
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         return checkpoint['epoch']
#     else:
#         print(f"No checkpoint found at {checkpoint_path}")
#         return None

# def main(args=None):
#     if args is None:
#         args = parse_args()

#     args.conf_jobnum = str(uuid.uuid4())
#     args.conf_timestamp = str(datetime.datetime.now())
#     args.conf_host = socket.gethostname()

#     priv_dataset = get_prive_dataset(args)
#     backbones_list = priv_dataset.get_backbone(args.parti_num, None)
#     model = get_model(backbones_list, args, priv_dataset.get_transform())
#     args.arch = model.nets_list[0].name

#     print('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))

#     # Load checkpoint if provided
#     start_epoch = 0
#     if args.checkpoint:
#         start_epoch = load_checkpoint(args.checkpoint, model)

#     # Training
#     for epoch in range(start_epoch, args.communication_epoch):
#         print(f"Starting epoch {epoch + 1}/{args.communication_epoch}")
#         train(model, priv_dataset, args)

#         # Save checkpoint after each epoch
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'model_state_dict': model.state_dict(),
#             'args': vars(args)
#         }, args.save_dir, filename=f"checkpoint_epoch_{epoch + 1}.pth.tar")

# if __name__ == '__main__':
#     main()
