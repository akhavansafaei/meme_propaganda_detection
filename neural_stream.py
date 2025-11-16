"""
Stream 1: Neural Multimodal Pattern Recognition
CLIP (visual) + RoBERTa (textual) → Soft Evidence (Probabilities)

This stream provides prior probabilities based on pattern recognition.
It does NOT make final decisions - just provides soft evidence for fusion.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from transformers import (
    CLIPModel, CLIPProcessor,
    RobertaModel, RobertaTokenizer
)
from PIL import Image

from config import NeuralConfig


class NeuralMultimodalStream:
    """
    Neural stream that combines CLIP (image) and RoBERTa (text)
    to produce soft probability evidence for each propaganda technique
    """

    def __init__(
        self,
        config: NeuralConfig,
        num_labels: int,
        device: str = "cuda"
    ):
        """
        Args:
            config: Neural configuration
            num_labels: Number of propaganda techniques
            device: Device to run on
        """
        self.config = config
        self.num_labels = num_labels
        self.device = device

        print(f"Initializing Neural Multimodal Stream on {device}...")

        # Load RoBERTa for text encoding
        print("  Loading RoBERTa...")
        self.roberta_model = RobertaModel.from_pretrained(
            config.roberta_model_name
        ).to(device)
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(
            config.roberta_model_name
        )

        # Freeze RoBERTa (we use it as feature extractor)
        for param in self.roberta_model.parameters():
            param.requires_grad = False
        self.roberta_model.eval()

        # Load CLIP for multimodal encoding
        print("  Loading CLIP...")
        self.clip_model = CLIPModel.from_pretrained(
            config.clip_model_name
        ).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            config.clip_model_name
        )

        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        # Classification head (trainable)
        roberta_dim = 768  # RoBERTa hidden size
        clip_dim = 1024    # CLIP combined text+image embedding (512+512)
        combined_dim = roberta_dim + clip_dim  # 1792

        print("  Building classification head...")
        self.classifier = self._build_classifier(combined_dim, num_labels).to(device)

        print(f"✓ Neural stream initialized with {combined_dim}->{num_labels} classifier")

    def _build_classifier(self, input_dim: int, num_labels: int) -> nn.Module:
        """
        Build a simple MLP classifier

        For more complex architecture, you can replace this with
        your multi-head GNN architecture from the notebook
        """
        return nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_labels)
            # No sigmoid here - we apply it in predict()
        )

    @torch.no_grad()
    def extract_roberta_features(self, texts: List[str]) -> torch.Tensor:
        """
        Extract RoBERTa features from text

        Args:
            texts: List of text strings

        Returns:
            Tensor of shape (batch_size, 768)
        """
        encoded = self.roberta_tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        outputs = self.roberta_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token embedding
        features = outputs.last_hidden_state[:, 0, :]
        return features

    @torch.no_grad()
    def extract_clip_features(
        self,
        texts: List[str],
        images: List[Image.Image]
    ) -> torch.Tensor:
        """
        Extract CLIP features from text and images

        Args:
            texts: List of text strings
            images: List of PIL images

        Returns:
            Tensor of shape (batch_size, 1024)
        """
        inputs = self.clip_processor(
            text=texts,
            images=images,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=77
        )

        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        outputs = self.clip_model(**inputs)

        # Concatenate text and image embeddings
        clip_features = torch.cat(
            (outputs.text_embeds, outputs.image_embeds),
            dim=-1
        )
        return clip_features

    def extract_features(
        self,
        texts: List[str],
        images: List[Image.Image]
    ) -> torch.Tensor:
        """
        Extract combined RoBERTa + CLIP features

        Args:
            texts: List of text strings (can include captions)
            images: List of PIL images

        Returns:
            Combined features of shape (batch_size, 1792)
        """
        roberta_features = self.extract_roberta_features(texts)
        clip_features = self.extract_clip_features(texts, images)

        combined = torch.cat((roberta_features, clip_features), dim=-1)
        return combined

    def predict(
        self,
        texts: List[str],
        images: List[Image.Image],
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Get soft probability predictions (prior evidence)

        Args:
            texts: List of text strings
            images: List of PIL images
            return_logits: If True, also return logits

        Returns:
            Dictionary with:
                - 'probabilities': Tensor of shape (batch_size, num_labels)
                - 'logits': Tensor of shape (batch_size, num_labels) [if requested]
        """
        # Extract features
        features = self.extract_features(texts, images)

        # Forward through classifier
        logits = self.classifier(features)

        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits)

        result = {'probabilities': probabilities}

        if return_logits:
            result['logits'] = logits

        return result

    def predict_single(
        self,
        text: str,
        image: Image.Image
    ) -> Dict[str, float]:
        """
        Predict for a single sample

        Args:
            text: Text string
            image: PIL image

        Returns:
            Dictionary mapping technique_index -> probability
        """
        result = self.predict([text], [image])
        probs = result['probabilities'][0]  # Get first sample

        # Convert to dict
        prob_dict = {i: float(probs[i]) for i in range(self.num_labels)}
        return prob_dict

    def train_mode(self):
        """Set classifier to training mode"""
        self.classifier.train()

    def eval_mode(self):
        """Set classifier to evaluation mode"""
        self.classifier.eval()

    def get_trainable_parameters(self):
        """Get trainable parameters (only classifier)"""
        return self.classifier.parameters()

    def save_classifier(self, path: str):
        """Save classifier weights"""
        torch.save(self.classifier.state_dict(), path)
        print(f"✓ Classifier saved to {path}")

    def load_classifier(self, path: str):
        """Load classifier weights"""
        self.classifier.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✓ Classifier loaded from {path}")

    def to(self, device: str):
        """Move model to device"""
        self.device = device
        self.roberta_model = self.roberta_model.to(device)
        self.clip_model = self.clip_model.to(device)
        self.classifier = self.classifier.to(device)
        return self


class NeuralStreamPredictor:
    """
    Wrapper for making predictions with a trained neural stream
    Handles batching and conversion to technique names
    """

    def __init__(
        self,
        neural_stream: NeuralMultimodalStream,
        idx_to_label: Dict[int, str]
    ):
        """
        Args:
            neural_stream: Trained neural stream model
            idx_to_label: Mapping from index to technique name
        """
        self.stream = neural_stream
        self.idx_to_label = idx_to_label

        # Set to eval mode
        self.stream.eval_mode()

    @torch.no_grad()
    def predict(
        self,
        text: str,
        image: Image.Image
    ) -> Dict[str, float]:
        """
        Predict for a single sample

        Args:
            text: Text string (can include caption)
            image: PIL image

        Returns:
            Dictionary mapping technique_name -> probability
        """
        # Get predictions from stream
        prob_dict_idx = self.stream.predict_single(text, image)

        # Convert indices to names
        prob_dict_named = {
            self.idx_to_label[idx]: prob
            for idx, prob in prob_dict_idx.items()
        }

        return prob_dict_named

    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        images: List[Image.Image]
    ) -> List[Dict[str, float]]:
        """
        Predict for a batch of samples

        Args:
            texts: List of text strings
            images: List of PIL images

        Returns:
            List of dictionaries, each mapping technique_name -> probability
        """
        result = self.stream.predict(texts, images)
        probabilities = result['probabilities']  # (batch_size, num_labels)

        # Convert to list of named dicts
        predictions = []
        for i in range(len(texts)):
            prob_dict = {
                self.idx_to_label[j]: float(probabilities[i, j])
                for j in range(self.stream.num_labels)
            }
            predictions.append(prob_dict)

        return predictions


if __name__ == "__main__":
    # Test neural stream
    from config import NeuralConfig, DataConfig
    from data_loader import create_label_mappings
    import torch

    print("Testing Neural Multimodal Stream...")

    # Create config
    config = NeuralConfig()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create label mappings
    label_to_idx, idx_to_label, num_labels = create_label_mappings()

    # Initialize stream
    stream = NeuralMultimodalStream(config, num_labels, config.device)

    # Test with dummy data
    dummy_text = "This is a test meme about politics"
    dummy_image = Image.new('RGB', (224, 224), color='white')

    # Create predictor
    predictor = NeuralStreamPredictor(stream, idx_to_label)

    # Predict
    predictions = predictor.predict(dummy_text, dummy_image)

    print(f"\n✓ Prediction successful!")
    print(f"  Got predictions for {len(predictions)} techniques")
    print(f"  Sample predictions:")
    for i, (tech, prob) in enumerate(list(predictions.items())[:5]):
        print(f"    {tech}: {prob:.4f}")
