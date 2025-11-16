"""
Data Loading Module for Meme Propaganda Detection
Handles loading data from Google Drive and creating datasets
"""

import json
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import random

from config import DataConfig


class PropagandaDataLoader:
    """Handles loading and preparing propaganda detection datasets"""

    def __init__(self, data_config: DataConfig):
        self.config = data_config

    def load_json_data(self, split: str = 'train') -> pd.DataFrame:
        """
        Load JSON data for a specific split

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            DataFrame with loaded data
        """
        # Map split to file path
        path_map = {
            'train': self.config.train_json,
            'val': self.config.val_json,
            'test': self.config.test_json
        }

        if split not in path_map:
            raise ValueError(f"Invalid split: {split}. Choose from {list(path_map.keys())}")

        # Try both relative and absolute paths
        json_path = path_map[split]

        # If not absolute, try relative to Drive
        if not os.path.isabs(json_path):
            json_path = self.config.get_full_path(json_path)

        # Fallback: try current directory
        if not os.path.exists(json_path):
            json_path = path_map[split]

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Data file not found: {json_path}")

        print(f"Loading {split} data from: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        print(f"✓ Loaded {len(df)} samples for {split} split")

        # Validate required columns
        required_cols = ['id', 'text', 'image']
        if split != 'test':
            required_cols.append('labels')

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠ Warning: Missing columns: {missing_cols}")

        return df

    def get_image_dir(self, split: str) -> str:
        """Get image directory for a split"""
        dir_map = {
            'train': self.config.train_img_dir,
            'val': self.config.val_img_dir,
            'test': self.config.test_img_dir
        }
        return dir_map.get(split, self.config.train_img_dir)

    def load_all_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load all data splits

        Returns:
            (train_df, val_df, test_df)
        """
        train_df = self.load_json_data('train')
        val_df = self.load_json_data('val')

        try:
            test_df = self.load_json_data('test')
        except FileNotFoundError:
            print("⚠ Test data not found, skipping")
            test_df = None

        return train_df, val_df, test_df


class MemePropagandaDataset(Dataset):
    """
    PyTorch Dataset for meme propaganda detection
    Returns raw data (text, image, labels) - preprocessing done by model
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        label_to_idx: Dict[str, int],
        is_test: bool = False,
        use_caption: bool = True,
        caption_separator: str = " [SEP] "
    ):
        """
        Args:
            df: DataFrame with data
            img_dir: Directory containing images
            label_to_idx: Mapping from label names to indices
            is_test: Whether this is test data (no labels)
            use_caption: Whether to use caption column
            caption_separator: Separator between text and caption
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.label_to_idx = label_to_idx
        self.num_labels = len(label_to_idx)
        self.is_test = is_test
        self.use_caption = use_caption
        self.caption_separator = caption_separator

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns:
            (text, image, labels, sample_id)
        """
        row = self.df.iloc[idx]

        # Get text
        text = row.get('text', "")
        if pd.isna(text):
            text = ""

        # Combine with caption if available
        if self.use_caption and 'caption' in row and pd.notna(row['caption']):
            caption = row['caption']
            combined_text = f"{text}{self.caption_separator}{caption}"
        else:
            combined_text = text

        # Load image
        image = self._load_image(row.get('image', ''))

        # Get labels (multi-hot encoding)
        if self.is_test or 'labels' not in row:
            labels = torch.zeros(self.num_labels)
        else:
            labels = self._encode_labels(row['labels'])

        # Sample ID
        sample_id = row.get('id', idx)

        return combined_text, image, labels, sample_id

    def _load_image(self, image_name: str) -> Image.Image:
        """Load image or return placeholder"""
        if not image_name or pd.isna(image_name):
            return self._create_placeholder_image()

        img_path = self.img_dir / image_name

        try:
            image = Image.open(img_path).convert('RGB')
            return image
        except Exception as e:
            # Return placeholder if image can't be loaded
            if not str(img_path).startswith('path/to/'):
                print(f"⚠ Error loading image {img_path}: {e}")
            return self._create_placeholder_image()

    def _create_placeholder_image(self) -> Image.Image:
        """Create a placeholder image when actual image is not available"""
        colors = ['white', 'lightgray', 'lightblue', 'lightgreen', 'lightyellow']
        color = random.choice(colors)
        return Image.new('RGB', (224, 224), color=color)

    def _encode_labels(self, label_list: List[str]) -> torch.Tensor:
        """
        Convert list of label names to multi-hot vector

        Args:
            label_list: List of technique names

        Returns:
            Multi-hot tensor of shape (num_labels,)
        """
        if not label_list or not isinstance(label_list, list):
            return torch.zeros(self.num_labels)

        labels = torch.zeros(self.num_labels)

        for label_name in label_list:
            if label_name in self.label_to_idx:
                idx = self.label_to_idx[label_name]
                labels[idx] = 1.0

        return labels


def create_label_mappings(techniques_json: str = "propaganda_techniques.json") -> Tuple[Dict, Dict, int]:
    """
    Create label to index mappings from techniques configuration

    Args:
        techniques_json: Path to propaganda techniques JSON

    Returns:
        (label_to_idx, idx_to_label, num_labels)
    """
    with open(techniques_json, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Extract all technique names
    technique_names = [t['name'] for t in config['propaganda_techniques']]

    # Create mappings
    label_to_idx = {name: i for i, name in enumerate(sorted(technique_names))}
    idx_to_label = {i: name for name, i in label_to_idx.items()}
    num_labels = len(label_to_idx)

    return label_to_idx, idx_to_label, num_labels


def create_dataloaders(
    data_config: DataConfig,
    label_to_idx: Dict[str, int],
    batch_size: int = 32,
    use_caption: bool = True,
    num_workers: int = 0
) -> Tuple:
    """
    Create PyTorch DataLoaders for all splits

    Args:
        data_config: Data configuration
        label_to_idx: Label to index mapping
        batch_size: Batch size
        use_caption: Whether to use captions
        num_workers: Number of data loading workers (use 0 for Colab)

    Returns:
        (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
    """
    from torch.utils.data import DataLoader

    # Load data
    loader = PropagandaDataLoader(data_config)
    train_df, val_df, test_df = loader.load_all_splits()

    # Create datasets
    train_dataset = MemePropagandaDataset(
        train_df,
        loader.get_image_dir('train'),
        label_to_idx,
        is_test=False,
        use_caption=use_caption
    )

    val_dataset = MemePropagandaDataset(
        val_df,
        loader.get_image_dir('val'),
        label_to_idx,
        is_test=False,
        use_caption=use_caption
    )

    test_dataset = None
    if test_df is not None:
        test_dataset = MemePropagandaDataset(
            test_df,
            loader.get_image_dir('test'),
            label_to_idx,
            is_test=True,
            use_caption=use_caption
        )

    # Create dataloaders (simple collate, no preprocessing here)
    def simple_collate(batch):
        """Simple collate that just batches the data"""
        texts, images, labels, ids = zip(*batch)
        labels = torch.stack(labels)
        return list(texts), list(images), labels, list(ids)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=simple_collate,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=simple_collate,
        num_workers=num_workers
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=simple_collate,
            num_workers=num_workers
        )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test data loading
    from config import DataConfig

    data_config = DataConfig()
    label_to_idx, idx_to_label, num_labels = create_label_mappings()

    print(f"Number of techniques: {num_labels}")
    print(f"First 5 techniques: {list(label_to_idx.keys())[:5]}")

    # Try loading data
    try:
        loader = PropagandaDataLoader(data_config)
        train_df = loader.load_json_data('train')
        print(f"\nTrain data shape: {train_df.shape}")
        print(f"Columns: {train_df.columns.tolist()}")
    except FileNotFoundError as e:
        print(f"\n⚠ Could not load data: {e}")
        print("This is expected if running outside Colab")
