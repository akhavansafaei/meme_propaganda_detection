"""
Stream 2: Knowledge Graph Reasoning Engine
Constraint Satisfaction + Rule-Based Refinement

This stream applies logical reasoning using:
1. Prerequisite constraints (hard)
2. Co-occurrence patterns (soft boost)
3. Mutual exclusion rules (soft penalty)
4. Hierarchical consistency (structural)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from prerequisite_checker import PrerequisiteChecker
from knowledge_graph import KnowledgeGraph, ConstraintSatisfactionSolver
from config import KnowledgeGraphConfig


class KnowledgeGraphStream:
    """
    Enhanced Knowledge Graph reasoning stream with constraint satisfaction
    """

    def __init__(
        self,
        config: KnowledgeGraphConfig,
        techniques_json: str = "propaganda_techniques.json"
    ):
        """
        Args:
            config: Knowledge graph configuration
            techniques_json: Path to techniques configuration
        """
        self.config = config
        self.techniques_json = techniques_json

        print("Initializing Knowledge Graph Stream...")

        # Load prerequisite checker
        print("  Loading prerequisite checker...")
        self.prerequisite_checker = PrerequisiteChecker(techniques_json)

        # Load knowledge graph
        print("  Loading knowledge graph...")
        self.knowledge_graph = KnowledgeGraph(techniques_json)

        # Load constraint solver
        print("  Initializing constraint satisfaction solver...")
        self.constraint_solver = ConstraintSatisfactionSolver(self.knowledge_graph)

        self.technique_names = list(self.knowledge_graph.techniques.keys())

        print(f"✓ Knowledge Graph Stream initialized with {len(self.technique_names)} techniques")

    def predict(
        self,
        text: str,
        image_features: Dict,
        neural_predictions: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Apply knowledge graph reasoning to refine neural predictions

        Args:
            text: Meme text content (can include caption)
            image_features: Extracted image features (entities, symbols, etc.)
            neural_predictions: Raw predictions from neural stream (technique -> prob)

        Returns:
            Dictionary with:
                - 'refined_predictions': Dict[str, float] - Refined probabilities
                - 'prerequisite_scores': Dict[str, float] - Prerequisite satisfaction
                - 'proposed_techniques': Dict[str, float] - Techniques proposed by KG
                - 'rejected_techniques': List[str] - Techniques rejected due to prerequisites
                - 'reasoning': Dict[str, str] - Explanations for each technique
        """
        # Step 1: Check prerequisites
        prerequisite_scores = self.prerequisite_checker.get_prerequisite_score(
            text, image_features
        )

        # Step 2: Apply constraint satisfaction
        refined = self.constraint_solver.solve(
            neural_predictions,
            prerequisite_scores,
            alpha=0.7,  # Weight for neural predictions
            beta=0.3    # Weight for constraint violations
        )

        # Step 3: Propose missing techniques (high recall for rare techniques)
        proposals = self.knowledge_graph.propose_missing_techniques(
            neural_predictions,
            prerequisite_scores,
            min_prerequisite_score=self.config.prerequisite_threshold
        )

        # Step 4: Merge proposals with refined scores
        for technique, proposed_score in proposals.items():
            if proposed_score > refined.get(technique, 0.0):
                refined[technique] = proposed_score

        # Step 5: Identify rejected techniques
        rejected = [
            tech for tech, prereq_score in prerequisite_scores.items()
            if prereq_score < self.config.prerequisite_threshold
            and neural_predictions.get(tech, 0.0) > 0.3  # Neural thought it was present
        ]

        # Step 6: Generate explanations
        detected_techniques = [
            tech for tech, score in refined.items()
            if score >= self.config.detection_threshold
        ]

        reasoning = {}
        for technique in self.technique_names:
            neural_score = neural_predictions.get(technique, 0.0)
            prereq_score = prerequisite_scores.get(technique, 0.0)
            final_score = refined.get(technique, 0.0)

            reasoning[technique] = self.knowledge_graph.get_explanation(
                technique,
                neural_score,
                prereq_score,
                final_score,
                detected_techniques
            )

        return {
            'refined_predictions': refined,
            'prerequisite_scores': prerequisite_scores,
            'proposed_techniques': proposals,
            'rejected_techniques': rejected,
            'reasoning': reasoning
        }

    def get_co_occurrence_matrix(self) -> np.ndarray:
        """Get the co-occurrence matrix"""
        return self.knowledge_graph.co_occurrence_matrix

    def get_mutual_exclusion_matrix(self) -> np.ndarray:
        """Get the mutual exclusion matrix"""
        return self.knowledge_graph.mutual_exclusion_matrix

    def update_co_occurrence_from_data(
        self,
        training_labels: List[List[str]],
        min_support: int = 5,
        pmi_threshold: float = 0.5
    ):
        """
        Learn co-occurrence patterns from training data

        Args:
            training_labels: List of label lists from training data
            min_support: Minimum number of co-occurrences to trust
            pmi_threshold: Minimum PMI to consider co-occurring
        """
        print(f"\nLearning co-occurrence from {len(training_labels)} training samples...")

        # Compute co-occurrence counts
        n_techniques = len(self.technique_names)
        co_counts = np.zeros((n_techniques, n_techniques))
        technique_counts = np.zeros(n_techniques)

        technique_to_idx = {name: i for i, name in enumerate(self.technique_names)}

        for labels in training_labels:
            indices = [technique_to_idx[label] for label in labels if label in technique_to_idx]

            # Count individual occurrences
            for idx in indices:
                technique_counts[idx] += 1

            # Count co-occurrences
            for i in indices:
                for j in indices:
                    if i != j:
                        co_counts[i, j] += 1

        # Compute PMI (Pointwise Mutual Information)
        total_samples = len(training_labels)
        pmi_matrix = np.zeros((n_techniques, n_techniques))

        for i in range(n_techniques):
            for j in range(n_techniques):
                if i == j or co_counts[i, j] < min_support:
                    continue

                # P(i,j) = co-occurrence / total
                p_ij = co_counts[i, j] / total_samples

                # P(i) = occurrence_i / total
                p_i = technique_counts[i] / total_samples
                p_j = technique_counts[j] / total_samples

                # PMI = log(P(i,j) / (P(i) * P(j)))
                if p_i > 0 and p_j > 0:
                    pmi = np.log((p_ij + 1e-10) / ((p_i * p_j) + 1e-10))
                    pmi_matrix[i, j] = max(0, pmi)  # Positive PMI only

        # Normalize to [0, 1]
        if pmi_matrix.max() > 0:
            pmi_matrix = pmi_matrix / pmi_matrix.max()

        # Filter by threshold
        pmi_matrix[pmi_matrix < pmi_threshold] = 0

        # Update knowledge graph
        self.knowledge_graph.co_occurrence_matrix = pmi_matrix

        # Report statistics
        high_co_occurrence = (pmi_matrix > 0.7).sum() // 2  # Divide by 2 for symmetry
        print(f"✓ Learned co-occurrence matrix:")
        print(f"  Total pairs with PMI > {pmi_threshold}: {(pmi_matrix > 0).sum() // 2}")
        print(f"  Strong co-occurrence (PMI > 0.7): {high_co_occurrence}")


class KGStreamPredictor:
    """
    Wrapper for making predictions with KG stream
    """

    def __init__(
        self,
        kg_stream: KnowledgeGraphStream,
        image_feature_extractor=None
    ):
        """
        Args:
            kg_stream: Knowledge graph stream
            image_feature_extractor: Optional image feature extractor
        """
        self.stream = kg_stream
        self.image_feature_extractor = image_feature_extractor

    def predict(
        self,
        text: str,
        image=None,
        neural_predictions: Dict[str, float] = None,
        return_full_output: bool = False
    ) -> Dict[str, any]:
        """
        Predict with knowledge graph reasoning

        Args:
            text: Text content
            image: PIL image (optional, for feature extraction)
            neural_predictions: Neural stream predictions (optional)
            return_full_output: If True, return full reasoning output

        Returns:
            Dictionary with predictions and optionally full reasoning
        """
        # Extract image features if extractor is available
        if self.image_feature_extractor and image:
            image_features = self.image_feature_extractor.extract(image)
        else:
            # Use empty features
            image_features = {
                'entities': [],
                'faces': 0,
                'symbols': [],
                'emotion': None,
                'objects': []
            }

        # If no neural predictions provided, use uniform prior
        if neural_predictions is None:
            neural_predictions = {
                tech: 0.1 for tech in self.stream.technique_names
            }

        # Run KG reasoning
        result = self.stream.predict(text, image_features, neural_predictions)

        if return_full_output:
            return result
        else:
            # Return only refined predictions
            return result['refined_predictions']


# Utility class for extracting image features (placeholder)
class SimpleImageFeatureExtractor:
    """
    Placeholder for image feature extraction
    In production, this would use actual computer vision models
    """

    def extract(self, image) -> Dict:
        """Extract features from image"""
        # This is a placeholder - in production you would use:
        # - Face detection (MTCNN, RetinaFace)
        # - Object detection (YOLO, Faster R-CNN)
        # - OCR (EasyOCR, PaddleOCR)
        # - Symbol/logo detection (custom trained model)
        # - Emotion recognition

        return {
            'entities': [],
            'faces': 0,
            'symbols': [],
            'emotion': None,
            'scene': '',
            'objects': []
        }


if __name__ == "__main__":
    # Test KG stream
    from config import KnowledgeGraphConfig

    print("Testing Knowledge Graph Stream...")

    config = KnowledgeGraphConfig()
    stream = KnowledgeGraphStream(config)

    # Test data
    text = "This politician is a complete liar and fraud!"
    image_features = {
        'entities': [{'text': 'politician', 'type': 'PERSON'}],
        'faces': 1,
        'symbols': [],
        'emotion': 'anger',
        'objects': []
    }

    # Dummy neural predictions
    neural_preds = {
        'Smears': 0.85,
        'Loaded Language': 0.75,
        'Name Calling/Labeling': 0.70,
        'Doubt': 0.40,
        'Appeal to Authority': 0.30,  # Should be rejected (no authority)
    }

    # Add other techniques with low probs
    for tech in stream.technique_names:
        if tech not in neural_preds:
            neural_preds[tech] = 0.1

    # Predict
    result = stream.predict(text, image_features, neural_preds)

    print("\n✓ Prediction successful!")
    print(f"\nTop 5 refined predictions:")
    sorted_preds = sorted(
        result['refined_predictions'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for tech, score in sorted_preds:
        print(f"  {tech}: {score:.3f}")

    print(f"\nRejected techniques: {result['rejected_techniques']}")

    if result['proposed_techniques']:
        print(f"\nProposed techniques:")
        for tech, score in result['proposed_techniques'].items():
            print(f"  {tech}: {score:.3f}")
