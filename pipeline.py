"""
Main Three-Stream Propaganda Detection Pipeline
Integrates Neural, KG, and LLM streams with fusion
"""

import json
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, asdict

from config import ExperimentConfig
from neural_stream import NeuralMultimodalStream, NeuralStreamPredictor
from kg_stream import KnowledgeGraphStream, KGStreamPredictor
from llm_stream import LLMSymbolicStream
from fusion import StreamFusion
from data_loader import create_label_mappings


@dataclass
class PredictionResult:
    """Complete prediction result from three-stream pipeline"""
    # Final results
    final_predictions: Dict[str, float]
    detected_techniques: List[str]

    # Individual stream predictions
    neural_predictions: Dict[str, float]
    kg_predictions: Dict[str, float]
    llm_predictions: Optional[Dict[str, float]]

    # Additional information
    prerequisite_scores: Dict[str, float]
    kg_reasoning: Dict[str, str]
    sample_id: any = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for saving"""
        return asdict(self)


class ThreeStreamPropagandaDetector:
    """
    Complete three-stream architecture for propaganda detection

    Usage:
        # Initialize
        detector = ThreeStreamPropagandaDetector(config)

        # Predict single sample
        result = detector.predict(text, image)

        # Predict batch
        results = detector.predict_batch(texts, images)
    """

    def __init__(self, config: ExperimentConfig):
        """
        Args:
            config: Complete experiment configuration
        """
        self.config = config

        print("\n" + "="*60)
        print("INITIALIZING THREE-STREAM PROPAGANDA DETECTOR")
        print("="*60)

        # Set random seed
        self._set_seed(config.seed)

        # Create label mappings
        self.label_to_idx, self.idx_to_label, self.num_labels = create_label_mappings(
            config.kg.techniques_json
        )
        self.technique_names = list(self.label_to_idx.keys())

        print(f"\nLoaded {self.num_labels} propaganda techniques")

        # Initialize streams
        self._initialize_streams()

        # Initialize fusion
        self._initialize_fusion()

        print("\n" + "="*60)
        print("✓ THREE-STREAM DETECTOR READY")
        print("="*60)

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _initialize_streams(self):
        """Initialize all three streams"""
        print("\n--- Initializing Streams ---")

        # Stream 1: Neural Multimodal
        print("\n[1/3] Neural Multimodal Stream")
        self.neural_stream = NeuralMultimodalStream(
            self.config.neural,
            self.num_labels,
            self.config.neural.device
        )
        self.neural_predictor = NeuralStreamPredictor(
            self.neural_stream,
            self.idx_to_label
        )

        # Stream 2: Knowledge Graph
        print("\n[2/3] Knowledge Graph Stream")
        self.kg_stream = KnowledgeGraphStream(
            self.config.kg,
            self.config.kg.techniques_json
        )
        self.kg_predictor = KGStreamPredictor(self.kg_stream)

        # Stream 3: LLM Symbolic Reasoner
        print("\n[3/3] LLM Symbolic Stream")

        # Load technique definitions for LLM
        with open(self.config.kg.techniques_json, 'r') as f:
            techniques_config = json.load(f)

        technique_defs = {
            t['name']: t['prerequisite']
            for t in techniques_config['propaganda_techniques']
        }

        self.llm_stream = LLMSymbolicStream(
            self.config.llm,
            technique_defs
        )

        print("\n✓ All streams initialized")

    def _initialize_fusion(self):
        """Initialize fusion module"""
        print("\n--- Initializing Fusion ---")
        self.fusion = StreamFusion(self.config.fusion, self.technique_names)

    def predict(
        self,
        text: str,
        image: Image.Image,
        sample_id: any = None,
        return_details: bool = True
    ) -> PredictionResult:
        """
        Predict propaganda techniques for a single sample

        Args:
            text: Meme text (can include caption)
            image: PIL Image
            sample_id: Optional sample identifier
            return_details: Whether to return full details

        Returns:
            PredictionResult with all predictions and reasoning
        """
        # Stream 1: Neural predictions
        neural_preds = self.neural_predictor.predict(text, image)

        # Stream 2: Knowledge graph predictions
        kg_result = self.kg_predictor.predict(
            text=text,
            image=image,
            neural_predictions=neural_preds,
            return_full_output=True
        )
        kg_preds = kg_result['refined_predictions']

        # Stream 3: LLM predictions (if enabled)
        llm_preds = None
        if self.config.llm.use_llm:
            # Create simple image description (placeholder - can be enhanced)
            image_desc = "Image with text overlay"  # TODO: Use actual image captioning

            llm_preds = self.llm_stream.predict(
                text=text,
                image_description=image_desc,
                neural_predictions=neural_preds,
                kg_predictions=kg_preds
            )

        # Fusion
        final_preds = self.fusion.fuse(neural_preds, kg_preds, llm_preds)

        # Apply thresholds to get detected techniques
        detected = self.fusion.apply_thresholds(final_preds)

        # Create result
        result = PredictionResult(
            final_predictions=final_preds,
            detected_techniques=detected,
            neural_predictions=neural_preds,
            kg_predictions=kg_preds,
            llm_predictions=llm_preds,
            prerequisite_scores=kg_result['prerequisite_scores'],
            kg_reasoning=kg_result['reasoning'],
            sample_id=sample_id
        )

        return result

    def predict_batch(
        self,
        texts: List[str],
        images: List[Image.Image],
        sample_ids: Optional[List] = None,
        batch_size: int = 32
    ) -> List[PredictionResult]:
        """
        Predict for a batch of samples

        Args:
            texts: List of text strings
            images: List of PIL images
            sample_ids: Optional list of sample IDs
            batch_size: Batch size for neural stream

        Returns:
            List of PredictionResults
        """
        if sample_ids is None:
            sample_ids = list(range(len(texts)))

        results = []

        # Process in batches for neural stream efficiency
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_images = images[i:i+batch_size]
            batch_ids = sample_ids[i:i+batch_size]

            # Neural predictions (batched)
            neural_preds_batch = self.neural_predictor.predict_batch(
                batch_texts, batch_images
            )

            # Process each sample individually for KG and LLM
            for j, (text, image, neural_preds, sid) in enumerate(
                zip(batch_texts, batch_images, neural_preds_batch, batch_ids)
            ):
                result = self.predict(text, image, sid)
                results.append(result)

        return results

    def save_predictions(
        self,
        results: List[PredictionResult],
        output_path: str,
        save_full_details: bool = True
    ):
        """
        Save predictions to JSON file

        Args:
            results: List of prediction results
            output_path: Path to save JSON
            save_full_details: Whether to save full details or just detections
        """
        if save_full_details:
            data = [result.to_dict() for result in results]
        else:
            # Save only essential information
            data = [
                {
                    'sample_id': result.sample_id,
                    'detected_techniques': result.detected_techniques,
                    'final_predictions': result.final_predictions
                }
                for result in results
            ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Predictions saved to {output_path}")

    def load_neural_checkpoint(self, checkpoint_path: str):
        """Load pretrained neural stream weights"""
        self.neural_stream.load_classifier(checkpoint_path)

    def save_neural_checkpoint(self, checkpoint_path: str):
        """Save neural stream weights"""
        self.neural_stream.save_classifier(checkpoint_path)

    def update_fusion_weights(
        self,
        alpha: float,
        beta: float,
        gamma: float
    ):
        """Update fusion weights"""
        self.config.fusion.alpha = alpha
        self.config.fusion.beta = beta
        self.config.fusion.gamma = gamma
        print(f"Updated fusion weights: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")

    def print_prediction(self, result: PredictionResult, top_k: int = 5):
        """Pretty print a prediction result"""
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)

        if result.sample_id is not None:
            print(f"Sample ID: {result.sample_id}")

        print(f"\n✓ Detected Techniques ({len(result.detected_techniques)}):")
        for technique in result.detected_techniques:
            final_score = result.final_predictions[technique]
            neural_score = result.neural_predictions[technique]
            kg_score = result.kg_predictions[technique]

            print(f"  • {technique}")
            print(f"      Final: {final_score:.3f} | Neural: {neural_score:.3f} | KG: {kg_score:.3f}")

        print(f"\nTop {top_k} Predictions:")
        sorted_preds = sorted(
            result.final_predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        for technique, score in sorted_preds:
            detected = "✓" if technique in result.detected_techniques else " "
            print(f"  {detected} {technique}: {score:.3f}")

        print("="*60)


if __name__ == "__main__":
    # Test pipeline
    from config import get_config_colab_no_llm

    print("Testing Three-Stream Pipeline...")

    # Create config
    config = get_config_colab_no_llm()

    # Initialize detector
    detector = ThreeStreamPropagandaDetector(config)

    # Test with dummy data
    dummy_text = "This politician is a complete liar and corrupt fraud! Everyone knows it!"
    dummy_image = Image.new('RGB', (224, 224), color='white')

    # Predict
    result = detector.predict(dummy_text, dummy_image, sample_id="test_001")

    # Print result
    detector.print_prediction(result)

    print("\n✓ Pipeline test successful!")
