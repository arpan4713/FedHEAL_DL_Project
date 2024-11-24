import torchvision.transforms as transforms
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader, random_split

class SVHNDataset:
    def __init__(self, args):
        """
        Initialize the SVHN dataset with necessary transformations.
        """
        self.args = args
        self.batch_size = args.local_epoch
        self.num_workers = 4

        # Data transformations
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalization for SVHN
        ])

        # Load train and test datasets
        self.train_dataset = SVHN(root='./data', split='train', download=True, transform=self.transform)
        self.test_dataset = SVHN(root='./data', split='test', download=True, transform=self.transform)

    def get_train_loader(self):
        """
        Return a DataLoader for the training dataset.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def get_test_loader(self):
        """
        Return a DataLoader for the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_backbone(self, parti_num, transform=None):
        """
        Simulate federated dataset partitioning among participants.
        """
        split_size = len(self.train_dataset) // parti_num
        splits = [split_size] * parti_num
        splits[-1] += len(self.train_dataset) - sum(splits)  # Handle remainder

        subsets = random_split(self.train_dataset, splits)
        return [
            DataLoader(subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            for subset in subsets
        ]

    def get_transform(self):
        """
        Return the transformation used for SVHN.
        """
        return self.transform


def get_svhn_dataset(args):
    """
    Factory method to return the SVHN dataset object.
    """
    return SVHNDataset(args)
