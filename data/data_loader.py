"""
Data loading and preprocessing utilities for RBBD Federated Defense
"""

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
import random
from typing import Tuple, List, Optional


class DatasetLoader:
    """Class for loading and preprocessing datasets"""
    
    def __init__(self, dataset_name: str = 'cifar10', data_root: str = './data'):
        self.dataset_name = dataset_name.lower()
        self.data_root = data_root
        self.num_classes = 10 if dataset_name.lower() == 'cifar10' else 100
        
        # Get dataset-specific parameters
        if self.dataset_name == 'cifar10':
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2023, 0.1994, 0.2010)
            self.dataset_class = datasets.CIFAR10
        elif self.dataset_name == 'cifar100':
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)
            self.dataset_class = datasets.CIFAR100
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def get_transforms(self, is_train: bool = True) -> transforms.Compose:
        """
        Get appropriate transforms for training or testing
        
        Args:
            is_train: Whether transforms are for training (includes augmentation)
            
        Returns:
            Composed transforms
        """
        if is_train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        
        return transform
    
    def load_datasets(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        Load training and test datasets
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        train_transform = self.get_transforms(is_train=True)
        test_transform = self.get_transforms(is_train=False)
        
        train_dataset = self.dataset_class(
            root=self.data_root,
            train=True,
            download=True,
            transform=train_transform
        )
        
        test_dataset = self.dataset_class(
            root=self.data_root,
            train=False,
            download=True,
            transform=test_transform
        )
        
        return train_dataset, test_dataset
    
    def create_validation_set(self, train_dataset: torch.utils.data.Dataset, 
                            val_size: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """
        Create validation set from training data
        
        Args:
            train_dataset: Full training dataset
            val_size: Number of samples for validation
            
        Returns:
            Tuple of (remaining_train_dataset, validation_dataset)
        """
        total_size = len(train_dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        val_dataset = Subset(train_dataset, val_indices)
        remaining_train = Subset(train_dataset, train_indices)
        
        return remaining_train, val_dataset


class BackdoorDatasetCreator:
    """Class for creating backdoored datasets for edge-case attacks"""
    
    def __init__(self, trigger_size: int = 3, target_class: int = 0):
        self.trigger_size = trigger_size
        self.target_class = target_class
    
    def add_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """
        Add trigger pattern to an image
        
        Args:
            image: Input image tensor
            
        Returns:
            Image with trigger added
        """
        triggered_image = image.clone()
        # Add white square trigger in bottom-right corner
        triggered_image[:, -self.trigger_size:, -self.trigger_size:] = 1.0
        return triggered_image
    
    def create_backdoor_dataset(self, base_dataset: torch.utils.data.Dataset,
                              tail_indices: np.ndarray,
                              train_samples: int,
                              test_samples: int) -> Tuple[Optional[TensorDataset], Optional[TensorDataset]]:
        """
        Create backdoored training and test datasets using edge cases
        
        Args:
            base_dataset: Source dataset to create backdoors from
            tail_indices: Indices of edge-case samples
            train_samples: Number of backdoor training samples
            test_samples: Number of backdoor test samples
            
        Returns:
            Tuple of (backdoor_train_dataset, backdoor_test_dataset)
        """
        if len(tail_indices) == 0:
            # If no tail indices, use random samples
            candidates = list(range(len(base_dataset)))
        else:
            candidates = sorted(tail_indices.tolist())
        
        total_needed = train_samples + test_samples
        if len(candidates) < total_needed:
            # If not enough candidates, sample with replacement or use all available
            chosen_indices = np.random.choice(candidates, size=total_needed, replace=True)
        else:
            chosen_indices = np.random.choice(candidates, size=total_needed, replace=False)
        
        # Split into train and test
        split_point = train_samples
        train_indices = chosen_indices[:split_point]
        test_indices = chosen_indices[split_point:]
        
        # Create backdoor training dataset
        train_images, train_labels = [], []
        for idx in train_indices:
            image, _ = base_dataset[idx]
            backdoor_image = self.add_trigger(image)
            train_images.append(backdoor_image)
            train_labels.append(self.target_class)
        
        # Create backdoor test dataset
        test_images, test_labels = [], []
        for idx in test_indices:
            image, _ = base_dataset[idx]
            backdoor_image = self.add_trigger(image)
            test_images.append(backdoor_image)
            test_labels.append(self.target_class)
        
        # Convert to TensorDatasets
        if train_images:
            train_bd = TensorDataset(
                torch.stack(train_images),
                torch.tensor(train_labels, dtype=torch.long)
            )
        else:
            train_bd = None
        
        if test_images:
            test_bd = TensorDataset(
                torch.stack(test_images),
                torch.tensor(test_labels, dtype=torch.long)
            )
        else:
            test_bd = None
        
        return train_bd, test_bd
    
    def create_clean_edge_case_dataset(self, base_dataset: torch.utils.data.Dataset,
                                     tail_indices: np.ndarray) -> Optional[TensorDataset]:
        """
        Create clean edge-case dataset (without triggers) for analysis
        
        Args:
            base_dataset: Source dataset
            tail_indices: Indices of edge-case samples
            
        Returns:
            Clean edge-case dataset
        """
        if len(tail_indices) == 0:
            return None
        
        images, labels = [], []
        for idx in tail_indices:
            image, label = base_dataset[idx]
            images.append(image)
            labels.append(label)
        
        if images:
            return TensorDataset(
                torch.stack(images),
                torch.tensor(labels, dtype=torch.long)
            )
        else:
            return None


class CustomCollate:
    """Custom collate function to handle mixed tensor/int labels"""
    
    def __call__(self, batch):
        """
        Custom collate function ensuring labels are always tensors
        
        Args:
            batch: Batch of (image, label) tuples
            
        Returns:
            Tuple of (stacked_images, label_tensor)
        """
        images, labels = zip(*batch)
        images_stacked = torch.stack(images, 0)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return images_stacked, labels_tensor


def create_data_loaders(dataset: torch.utils.data.Dataset,
                       batch_size: int = 128,
                       shuffle: bool = True,
                       num_workers: int = 0) -> DataLoader:
    """
    Create DataLoader with custom collate function
    
    Args:
        dataset: Dataset to create loader for
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=CustomCollate()
    )


def split_dataset_by_indices(dataset: torch.utils.data.Dataset,
                           indices: List[int]) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Split dataset into two subsets based on indices
    
    Args:
        dataset: Original dataset
        indices: Indices for first subset
        
    Returns:
        Tuple of (subset_with_indices, subset_without_indices)
    """
    all_indices = set(range(len(dataset)))
    remaining_indices = list(all_indices - set(indices))
    
    subset_with = Subset(dataset, indices)
    subset_without = Subset(dataset, remaining_indices)
    
    return subset_with, subset_without


def get_class_distribution(dataset: torch.utils.data.Dataset) -> dict:
    """
    Get class distribution of a dataset
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Dictionary mapping class indices to counts
    """
    class_counts = {}
    
    for i in range(len(dataset)):
        _, label = dataset[i]
        if isinstance(label, torch.Tensor):
            label = label.item()
        
        class_counts[label] = class_counts.get(label, 0) + 1
    
    return class_counts