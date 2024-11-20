import os
import sys
import socket
import torch
import torch.multiprocessing
import warnings
import numpy as np
import uuid
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from datasets import Priv_NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_prive_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")

# Config paths
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')

# Clustering configuration
NUM_CLUSTERS = 3
RECLUSTER_INTERVAL = 5
CHECKPOINT_DIR = "./checkpoints"

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(round_num, model, args):
    """
    Save the current training state as a checkpoint.
    Args:
        round_num (int): Current communication round.
        model: Federated learning model.
        args: Command-line arguments and other metadata.
    """
    checkpoint = {
        'round_num': round_num,
        'model_state_dict': model.state_dict(),
        'args': vars(args)
    }
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_round_{round_num}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, args):
    """
    Load the latest checkpoint to resume training.
    Args:
        model: Federated learning model.
        args: Command-line arguments.
    Returns:
        int: The communication round to resume from.
    """
    checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint")])
    if not checkpoint_files:
        print("No checkpoint found. Starting training from scratch.")
        return 1  # Start from round 1 if no checkpoint exists

    latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
    checkpoint = torch.load(latest_checkpoint)

    model.load_state_dict(checkpoint['model_state_dict'])
    for key, value in checkpoint['args'].items():
        setattr(args, key, value)

    print(f"Resuming from checkpoint: {latest_checkpoint}")
    return checkpoint['round_num']

def calculate_similarity_matrix(client_data):
    return cosine_similarity(client_data)

def perform_spectral_clustering(similarity_matrix, num_clusters):
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        random_state=42
    )
    return clustering.fit_predict(similarity_matrix)

def dynamic_clustering(round_num, client_data, num_clusters):
    if round_num % RECLUSTER_INTERVAL == 0:
        print(f"Performing clustering at round {round_num}...")
        similarity_matrix = calculate_similarity_matrix(client_data)
        cluster_labels = perform_spectral_clustering(similarity_matrix, num_clusters)
        print(f"Cluster labels: {cluster_labels}")
        return cluster_labels
    return None

def get_client_data(participants):
    return [np.random.rand(100) for _ in range(participants)]

def train_with_clustering(model, priv_dataset, args):
    client_data = get_client_data(args.parti_num)

    # Resume from last checkpoint if available
    start_round = load_checkpoint(model, args)

    for round_num in range(start_round, args.communication_epoch + 1):
        print(f"Round {round_num} starting...")

        # Dynamic clustering step
        cluster_labels = dynamic_clustering(round_num, client_data, NUM_CLUSTERS)
        if cluster_labels is not None:
            print(f"Clusters formed: {cluster_labels}")

        # Standard federated training
        train(model, priv_dataset, args)

        # Save checkpoint at the end of each round
        save_checkpoint(round_num, model, args)

        print(f"Round {round_num} completed.\n")

def parse_args():
    parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--communication_epoch', type=int, default=200, help='The Communication Epoch in Federated Learning')
    parser.add_argument('--local_epoch', type=int, default=10, help='The Local Epoch for each Participant')
    parser.add_argument('--parti_num', type=int, default=20, help='The Number for Participants')
    parser.add_argument('--seed', type=int, default=0, help='The random seed.')
    parser.add_argument('--rand_dataset', type=int, default=0, help='The random dataset seed.')
    parser.add_argument('--model', type=str, default='fedavgheal', choices=get_all_models(), help='Model name.')
    parser.add_argument('--structure', type=str, default='homogeneity')
    parser.add_argument('--dataset', type=str, default='fl_digits', choices=DATASET_NAMES, help='Dataset to use.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for Dirichlet sampler.')
    parser.add_argument('--online_ratio', type=float, default=1, help='Ratio of online clients.')
    parser.add_argument('--learning_decay', type=int, default=0, help='Learning rate decay option.')
    parser.add_argument('--averaging', type=str, default='weight', help='Averaging strategy option.')
    parser.add_argument('--wHEAL', type=int, default=1, help='CORE of FedHEAL to add HEAL.')
    parser.add_argument('--threshold', type=float, default=0.3, help='Threshold of HEAL.')
    parser.add_argument('--beta', type=float, default=0.4, help='Momentum update beta.')

    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_args()

    best = best_args[args.dataset][args.model]
    for key, value in best.items():
        setattr(args, key, value)

    if args.seed is not None:
        set_random_seed(args.seed)
    return args

def main(args=None):
    if args is None:
        args = parse_args()

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    priv_dataset = get_prive_dataset(args)
    backbones_list = priv_dataset.get_backbone(args.parti_num, None)
    model = get_model(backbones_list, args, priv_dataset.get_transform())
    args.arch = model.nets_list[0].name

    print('{}_{}_{}_{}_{}'.format(args.model, args.parti_num, args.dataset, args.communication_epoch, args.local_epoch))

    train_with_clustering(model, priv_dataset, args)

if __name__ == '__main__':
    main()




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
