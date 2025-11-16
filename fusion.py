"""
Stream Fusion Module
Combines predictions from Neural, KG, and LLM streams
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import FusionConfig


@dataclass
class StreamPredictions:
    """Container for predictions from all streams"""
    neural: Dict[str, float]
    kg: Dict[str, float]
    llm: Optional[Dict[str, float]] = None


class StreamFusion:
    """
    Fuses predictions from multiple reasoning streams
    Supports different fusion strategies
    """

    def __init__(self, config: FusionConfig, technique_names: List[str]):
        """
        Args:
            config: Fusion configuration
            technique_names: List of all technique names
        """
        self.config = config
        self.technique_names = technique_names

        print(f"Stream Fusion initialized:")
        print(f"  Strategy: {config.strategy}")
        print(f"  Weights: α={config.alpha}, β={config.beta}, γ={config.gamma}")
        print(f"  Veto enabled: {config.enable_veto}")

    def fuse(
        self,
        neural_predictions: Dict[str, float],
        kg_predictions: Dict[str, float],
        llm_predictions: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Fuse predictions from all streams

        Args:
            neural_predictions: Neural stream predictions
            kg_predictions: Knowledge graph predictions
            llm_predictions: LLM predictions (optional)

        Returns:
            Fused predictions
        """
        if self.config.strategy == "weighted":
            return self._weighted_fusion(
                neural_predictions, kg_predictions, llm_predictions
            )
        elif self.config.strategy == "voting":
            return self._voting_fusion(
                neural_predictions, kg_predictions, llm_predictions
            )
        elif self.config.strategy == "adaptive":
            return self._adaptive_fusion(
                neural_predictions, kg_predictions, llm_predictions
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {self.config.strategy}")

    def _weighted_fusion(
        self,
        neural_preds: Dict[str, float],
        kg_preds: Dict[str, float],
        llm_preds: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Weighted fusion with optional veto logic

        Formula:
            If veto triggered: final = veto_penalty * neural
            Else: final = α * neural + β * kg + γ * llm
        """
        alpha = self.config.alpha
        beta = self.config.beta
        gamma = self.config.gamma

        # Normalize weights if LLM is not used
        if llm_preds is None:
            total = alpha + beta
            alpha = alpha / total
            beta = beta / total
            gamma = 0.0

        fused = {}

        for technique in self.technique_names:
            neural_score = neural_preds.get(technique, 0.0)
            kg_score = kg_preds.get(technique, 0.0)
            llm_score = llm_preds.get(technique, 0.5) if llm_preds else 0.5

            # Check veto conditions
            if self.config.enable_veto:
                # KG veto: if prerequisites not satisfied (score == 0)
                if kg_score <= self.config.kg_veto_threshold:
                    fused[technique] = self.config.veto_penalty * neural_score
                    continue

                # LLM veto: if LLM strongly rejects
                if llm_preds and llm_score < self.config.llm_veto_threshold:
                    fused[technique] = self.config.veto_penalty * neural_score
                    continue

            # Normal weighted fusion
            if llm_preds:
                score = alpha * neural_score + beta * kg_score + gamma * llm_score
            else:
                score = alpha * neural_score + beta * kg_score

            # Clamp to [0, 1]
            fused[technique] = max(0.0, min(1.0, score))

        return fused

    def _voting_fusion(
        self,
        neural_preds: Dict[str, float],
        kg_preds: Dict[str, float],
        llm_preds: Optional[Dict[str, float]],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Majority voting fusion

        Each stream votes (above threshold = vote)
        Final score = votes / total_streams
        """
        fused = {}
        num_streams = 3 if llm_preds else 2

        for technique in self.technique_names:
            votes = 0

            if neural_preds.get(technique, 0.0) >= threshold:
                votes += 1

            if kg_preds.get(technique, 0.0) >= threshold:
                votes += 1

            if llm_preds and llm_preds.get(technique, 0.5) >= threshold:
                votes += 1

            fused[technique] = votes / num_streams

        return fused

    def _adaptive_fusion(
        self,
        neural_preds: Dict[str, float],
        kg_preds: Dict[str, float],
        llm_preds: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Adaptive fusion: adjust weights based on agreement

        If streams agree → trust neural more (faster)
        If streams disagree → trust KG/LLM more (safer)
        """
        fused = {}

        for technique in self.technique_names:
            neural_score = neural_preds.get(technique, 0.0)
            kg_score = kg_preds.get(technique, 0.0)
            llm_score = llm_preds.get(technique, 0.5) if llm_preds else 0.5

            # Measure disagreement
            scores = [neural_score, kg_score]
            if llm_preds:
                scores.append(llm_score)

            disagreement = np.std(scores)

            # Adjust weights based on disagreement
            if disagreement < 0.2:
                # High agreement → trust neural more
                alpha, beta, gamma = 0.6, 0.3, 0.1
            elif disagreement > 0.4:
                # High disagreement → trust KG/LLM more
                alpha, beta, gamma = 0.2, 0.4, 0.4
            else:
                # Medium disagreement → balanced
                alpha, beta, gamma = 0.4, 0.4, 0.2

            # Normalize if no LLM
            if not llm_preds:
                total = alpha + beta
                alpha = alpha / total
                beta = beta / total
                gamma = 0.0

            # Fuse
            if llm_preds:
                score = alpha * neural_score + beta * kg_score + gamma * llm_score
            else:
                score = alpha * neural_score + beta * kg_score

            fused[technique] = max(0.0, min(1.0, score))

        return fused

    def apply_thresholds(
        self,
        predictions: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Apply thresholds to get final detected techniques

        Args:
            predictions: Fused predictions
            thresholds: Per-technique thresholds (optional)

        Returns:
            List of detected technique names
        """
        if thresholds is None:
            # Use default threshold
            default_threshold = 0.35
            thresholds = {tech: default_threshold for tech in self.technique_names}

        detected = []
        for technique, score in predictions.items():
            threshold = thresholds.get(technique, 0.35)
            if score >= threshold:
                detected.append(technique)

        return detected

    def optimize_weights(
        self,
        validation_data: List[Tuple[Dict, Dict, Dict, List[str]]],
        metric: str = "f1"
    ) -> Tuple[float, float, float]:
        """
        Optimize fusion weights on validation data

        Args:
            validation_data: List of (neural_preds, kg_preds, llm_preds, ground_truth)
            metric: Optimization metric ('f1', 'precision', 'recall')

        Returns:
            Optimal (alpha, beta, gamma)
        """
        print(f"\nOptimizing fusion weights on {len(validation_data)} samples...")

        best_score = 0.0
        best_weights = (self.config.alpha, self.config.beta, self.config.gamma)

        # Grid search over weights
        for alpha in np.arange(0.2, 0.7, 0.1):
            for beta in np.arange(0.2, 0.7, 0.1):
                gamma = 1.0 - alpha - beta

                if gamma < 0 or gamma > 0.6:
                    continue

                # Temporarily set weights
                orig_alpha, orig_beta, orig_gamma = self.config.alpha, self.config.beta, self.config.gamma
                self.config.alpha = alpha
                self.config.beta = beta
                self.config.gamma = gamma

                # Evaluate on validation data
                score = self._evaluate_fusion(validation_data, metric)

                if score > best_score:
                    best_score = score
                    best_weights = (alpha, beta, gamma)

                # Restore weights
                self.config.alpha = orig_alpha
                self.config.beta = orig_beta
                self.config.gamma = orig_gamma

        # Set best weights
        self.config.alpha, self.config.beta, self.config.gamma = best_weights

        print(f"✓ Best weights found: α={best_weights[0]:.2f}, β={best_weights[1]:.2f}, γ={best_weights[2]:.2f}")
        print(f"  {metric.upper()}: {best_score:.4f}")

        return best_weights

    def _evaluate_fusion(
        self,
        validation_data: List[Tuple],
        metric: str
    ) -> float:
        """Evaluate fusion on validation data"""
        from sklearn.metrics import f1_score, precision_score, recall_score

        all_preds = []
        all_labels = []

        for neural_preds, kg_preds, llm_preds, ground_truth in validation_data:
            # Fuse predictions
            fused = self.fuse(neural_preds, kg_preds, llm_preds)

            # Apply thresholds
            detected = self.apply_thresholds(fused)

            # Convert to binary vector
            pred_vector = [1 if tech in detected else 0 for tech in self.technique_names]
            label_vector = [1 if tech in ground_truth else 0 for tech in self.technique_names]

            all_preds.append(pred_vector)
            all_labels.append(label_vector)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metric
        if metric == "f1":
            return f1_score(all_labels, all_preds, average='micro', zero_division=0)
        elif metric == "precision":
            return precision_score(all_labels, all_preds, average='micro', zero_division=0)
        elif metric == "recall":
            return recall_score(all_labels, all_preds, average='micro', zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")


if __name__ == "__main__":
    # Test fusion
    from config import FusionConfig

    print("Testing Stream Fusion...")

    techniques = ['Smears', 'Loaded Language', 'Name Calling/Labeling', 'Doubt']

    config = FusionConfig()
    fusion = StreamFusion(config, techniques)

    # Test predictions
    neural_preds = {'Smears': 0.85, 'Loaded Language': 0.75, 'Name Calling/Labeling': 0.60, 'Doubt': 0.30}
    kg_preds = {'Smears': 0.90, 'Loaded Language': 0.80, 'Name Calling/Labeling': 0.70, 'Doubt': 0.05}
    llm_preds = {'Smears': 0.88, 'Loaded Language': 0.70, 'Name Calling/Labeling': 0.65, 'Doubt': 0.10}

    # Test weighted fusion
    fused = fusion.fuse(neural_preds, kg_preds, llm_preds)

    print("\n✓ Fusion successful!")
    print("\nFused predictions:")
    for tech, score in fused.items():
        print(f"  {tech}: {score:.3f}")

    # Test threshold application
    detected = fusion.apply_thresholds(fused)
    print(f"\nDetected techniques: {detected}")
