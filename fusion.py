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
    Supports both global and per-technique weight optimization
    """

    def __init__(self, config: FusionConfig, technique_names: List[str]):
        """
        Args:
            config: Fusion configuration
            technique_names: List of all technique names
        """
        self.config = config
        self.technique_names = technique_names

        # Per-technique weights (initially None, uses global weights)
        self.per_technique_weights = None

        # Flag to use per-technique weights
        self.use_per_technique_weights = False

        print(f"Stream Fusion initialized:")
        print(f"  Strategy: {config.strategy}")
        print(f"  Weights: α={config.alpha}, β={config.beta}, γ={config.gamma}")
        print(f"  Veto enabled: {config.enable_veto}")
        print(f"  Per-technique weights: {self.use_per_technique_weights}")

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

        Supports both global and per-technique weights
        """
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

            # Get weights for this technique
            if self.use_per_technique_weights and self.per_technique_weights:
                # Use per-technique weights
                weights = self.per_technique_weights.get(
                    technique,
                    (self.config.alpha, self.config.beta, self.config.gamma)
                )
                alpha, beta, gamma = weights
            else:
                # Use global weights
                alpha = self.config.alpha
                beta = self.config.beta
                gamma = self.config.gamma

            # Normalize weights if LLM is not used
            if llm_preds is None:
                total = alpha + beta
                if total > 0:
                    alpha = alpha / total
                    beta = beta / total
                gamma = 0.0

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

    def optimize_per_technique_weights(
        self,
        validation_data: List[Tuple[Dict, Dict, Dict, List[str]]],
        metric: str = "f1",
        search_space: str = "coarse",
        min_samples_per_technique: int = 5
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Optimize fusion weights separately for each technique

        Args:
            validation_data: List of (neural_preds, kg_preds, llm_preds, ground_truth)
            metric: Optimization metric ('f1', 'precision', 'recall')
            search_space: 'coarse' (0.1 step), 'fine' (0.05 step), or 'very_fine' (0.025 step)
            min_samples_per_technique: Minimum samples needed to optimize per technique

        Returns:
            Dictionary mapping technique to optimal (alpha, beta, gamma)
        """
        print("\n" + "="*70)
        print("OPTIMIZING PER-TECHNIQUE FUSION WEIGHTS")
        print("="*70)
        print(f"Validation samples: {len(validation_data)}")
        print(f"Optimization metric: {metric.upper()}")
        print(f"Search space: {search_space}")
        print("")

        # Determine step size based on search space
        if search_space == "coarse":
            step = 0.1
        elif search_space == "fine":
            step = 0.05
        elif search_space == "very_fine":
            step = 0.025
        else:
            step = 0.1

        # Count samples per technique
        technique_counts = {tech: 0 for tech in self.technique_names}
        for _, _, _, ground_truth in validation_data:
            for tech in ground_truth:
                if tech in technique_counts:
                    technique_counts[tech] += 1

        # Initialize per-technique weights
        per_technique_weights = {}

        # Optimize for each technique independently
        for tech_idx, technique in enumerate(self.technique_names, 1):
            count = technique_counts[technique]

            print(f"[{tech_idx}/{len(self.technique_names)}] {technique}")
            print(f"  Samples with this technique: {count}")

            # Skip optimization if not enough samples
            if count < min_samples_per_technique:
                # Use global weights as fallback
                per_technique_weights[technique] = (
                    self.config.alpha,
                    self.config.beta,
                    self.config.gamma
                )
                print(f"  ⚠ Not enough samples, using global weights")
                print(f"  Weights: α={self.config.alpha:.3f}, β={self.config.beta:.3f}, γ={self.config.gamma:.3f}\n")
                continue

            # Filter validation data to only include samples with this technique
            # to optimize for this specific technique's performance
            best_score = 0.0
            best_weights = (self.config.alpha, self.config.beta, self.config.gamma)

            # Grid search
            alpha_range = np.arange(0.1, 0.8, step)
            beta_range = np.arange(0.1, 0.8, step)

            for alpha in alpha_range:
                for beta in beta_range:
                    gamma = 1.0 - alpha - beta

                    # Skip invalid weight combinations
                    if gamma < 0.05 or gamma > 0.7:
                        continue

                    # Temporarily set this technique's weights
                    temp_weights = {technique: (alpha, beta, gamma)}
                    orig_per_tech = self.per_technique_weights
                    orig_use_per_tech = self.use_per_technique_weights

                    self.per_technique_weights = temp_weights
                    self.use_per_technique_weights = True

                    # Evaluate on this technique
                    score = self._evaluate_single_technique(
                        validation_data, technique, metric
                    )

                    # Restore original settings
                    self.per_technique_weights = orig_per_tech
                    self.use_per_technique_weights = orig_use_per_tech

                    if score > best_score:
                        best_score = score
                        best_weights = (alpha, beta, gamma)

            per_technique_weights[technique] = best_weights
            print(f"  ✓ Optimal weights: α={best_weights[0]:.3f}, β={best_weights[1]:.3f}, γ={best_weights[2]:.3f}")
            print(f"  {metric.upper()}: {best_score:.4f}\n")

        # Set optimized weights
        self.per_technique_weights = per_technique_weights
        self.use_per_technique_weights = True

        print("="*70)
        print("✓ PER-TECHNIQUE OPTIMIZATION COMPLETE")
        print("="*70)

        # Show summary
        self._print_weight_summary()

        return per_technique_weights

    def _evaluate_single_technique(
        self,
        validation_data: List[Tuple],
        technique: str,
        metric: str
    ) -> float:
        """
        Evaluate performance for a single technique

        Args:
            validation_data: Validation dataset
            technique: Technique name to evaluate
            metric: Metric to compute

        Returns:
            Score for this technique
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        all_preds = []
        all_labels = []

        for neural_preds, kg_preds, llm_preds, ground_truth in validation_data:
            # Fuse predictions
            fused = self.fuse(neural_preds, kg_preds, llm_preds)

            # Get prediction for this technique
            pred = 1 if fused.get(technique, 0.0) >= 0.35 else 0
            label = 1 if technique in ground_truth else 0

            all_preds.append(pred)
            all_labels.append(label)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metric
        if metric == "f1":
            return f1_score(all_labels, all_preds, average='binary', zero_division=0)
        elif metric == "precision":
            return precision_score(all_labels, all_preds, average='binary', zero_division=0)
        elif metric == "recall":
            return recall_score(all_labels, all_preds, average='binary', zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def set_per_technique_weights(self, weights: Dict[str, Tuple[float, float, float]]):
        """
        Manually set per-technique weights

        Args:
            weights: Dictionary mapping technique name to (alpha, beta, gamma)
        """
        self.per_technique_weights = weights
        self.use_per_technique_weights = True
        print(f"✓ Per-technique weights set for {len(weights)} techniques")

    def get_per_technique_weights(self) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """Get current per-technique weights"""
        return self.per_technique_weights

    def enable_per_technique_weights(self):
        """Enable per-technique weight mode"""
        if self.per_technique_weights is None:
            print("⚠ Warning: No per-technique weights set. Please optimize first.")
            return
        self.use_per_technique_weights = True
        print("✓ Per-technique weights enabled")

    def disable_per_technique_weights(self):
        """Disable per-technique weight mode (use global weights)"""
        self.use_per_technique_weights = False
        print("✓ Global weights enabled")

    def _print_weight_summary(self):
        """Print summary of per-technique weights"""
        if not self.per_technique_weights:
            print("No per-technique weights set")
            return

        print("\nPer-Technique Weight Summary:")
        print("-" * 70)
        print(f"{'Technique':<40} {'α (Neural)':<12} {'β (KG)':<12} {'γ (LLM)':<12}")
        print("-" * 70)

        for technique in self.technique_names:
            if technique in self.per_technique_weights:
                alpha, beta, gamma = self.per_technique_weights[technique]
                print(f"{technique:<40} {alpha:<12.3f} {beta:<12.3f} {gamma:<12.3f}")

        print("-" * 70)

    def save_per_technique_weights(self, filepath: str):
        """Save per-technique weights to JSON file"""
        if not self.per_technique_weights:
            print("⚠ No per-technique weights to save")
            return

        import json
        data = {
            tech: {"alpha": w[0], "beta": w[1], "gamma": w[2]}
            for tech, w in self.per_technique_weights.items()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Per-technique weights saved to {filepath}")

    def load_per_technique_weights(self, filepath: str):
        """Load per-technique weights from JSON file"""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.per_technique_weights = {
            tech: (w["alpha"], w["beta"], w["gamma"])
            for tech, w in data.items()
        }
        self.use_per_technique_weights = True

        print(f"✓ Per-technique weights loaded from {filepath}")
        self._print_weight_summary()


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
