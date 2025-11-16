"""
Training Module for Neural Stream
Handles training loop, evaluation, and checkpointing
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from config import NeuralConfig
from neural_stream import NeuralMultimodalStream
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (batch_size, num_labels)
            targets: Binary labels (batch_size, num_labels)
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


class NeuralStreamTrainer:
    """Trainer for neural multimodal stream"""

    def __init__(
        self,
        model: NeuralMultimodalStream,
        config: NeuralConfig,
        idx_to_label: Dict[int, str],
        device: str = "cuda"
    ):
        """
        Args:
            model: Neural stream model
            config: Training configuration
            idx_to_label: Index to label mapping
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.idx_to_label = idx_to_label
        self.device = device

        # Loss function
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0)

        # Optimizer (only classifier parameters)
        self.optimizer = Adam(
            self.model.get_trainable_parameters(),
            lr=config.learning_rate
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=2,
            factor=0.5,
            verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }

        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_model_state = None

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train_mode()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for texts, images, labels, _ in pbar:
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            result = self.model.predict(texts, images, return_logits=True)
            logits = result['logits']

            # Compute loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(),
                max_norm=1.0
            )

            # Update weights
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate on validation set

        Args:
            val_loader: Validation data loader
            threshold: Detection threshold

        Returns:
            Dictionary with metrics
        """
        self.model.eval_mode()

        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for texts, images, labels, _ in tqdm(val_loader, desc="Evaluating"):
            labels = labels.to(self.device)

            # Forward pass
            result = self.model.predict(texts, images, return_logits=True)
            logits = result['logits']
            probabilities = result['probabilities']

            # Compute loss
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            # Apply threshold
            predictions = (probabilities > threshold).float()

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute metrics
        f1 = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
        precision = precision_score(all_labels, all_predictions, average='micro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='micro', zero_division=0)

        avg_loss = total_loss / num_batches

        metrics = {
            'loss': avg_loss,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: str = "./checkpoints",
        save_best: bool = True
    ):
        """
        Complete training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model
        """
        print("\n" + "="*60)
        print("STARTING NEURAL STREAM TRAINING")
        print("="*60)

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_metrics = self.evaluate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])

            # Print metrics
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_metrics['f1'])

            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_model_state = self.model.classifier.state_dict().copy()

                if save_best:
                    checkpoint_path = Path(checkpoint_dir) / f"best_model_f1_{val_metrics['f1']:.4f}.pth"
                    self.model.save_classifier(str(checkpoint_path))

                print(f"  ✓ New best F1: {val_metrics['f1']:.4f}")

        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Best Val F1: {self.best_val_f1:.4f}")

        # Restore best model
        if self.best_model_state is not None:
            self.model.classifier.load_state_dict(self.best_model_state)
            print("✓ Best model restored")

    def save_history(self, path: str):
        """Save training history to JSON"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training history saved to {path}")

    def detailed_evaluation(
        self,
        val_loader: DataLoader,
        threshold: float = 0.5
    ) -> str:
        """
        Detailed evaluation with per-class metrics

        Args:
            val_loader: Validation data loader
            threshold: Detection threshold

        Returns:
            Classification report string
        """
        self.model.eval_mode()

        all_predictions = []
        all_labels = []

        for texts, images, labels, _ in tqdm(val_loader, desc="Detailed Eval"):
            labels = labels.to(self.device)

            result = self.model.predict(texts, images)
            probabilities = result['probabilities']
            predictions = (probabilities > threshold).float()

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Get label names
        target_names = [self.idx_to_label[i] for i in range(len(self.idx_to_label))]

        # Generate classification report
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=target_names,
            zero_division=0
        )

        return report


if __name__ == "__main__":
    # Test trainer
    from config import NeuralConfig, DataConfig
    from data_loader import create_label_mappings, create_dataloaders
    from neural_stream import NeuralMultimodalStream

    print("Testing Neural Stream Trainer...")

    # Create configs
    neural_config = NeuralConfig()
    neural_config.batch_size = 4
    neural_config.epochs = 2
    neural_config.device = "cuda" if torch.cuda.is_available() else "cpu"

    data_config = DataConfig()

    # Create label mappings
    label_to_idx, idx_to_label, num_labels = create_label_mappings()

    # Initialize model
    model = NeuralMultimodalStream(neural_config, num_labels, neural_config.device)

    # Initialize trainer
    trainer = NeuralStreamTrainer(model, neural_config, idx_to_label, neural_config.device)

    print("\n✓ Trainer initialized successfully")
    print("  To train, provide data loaders:")
    print("  trainer.train(train_loader, val_loader, num_epochs=10)")
