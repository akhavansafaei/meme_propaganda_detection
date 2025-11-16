"""
Knowledge Graph Reasoning Engine for Propaganda Detection

This module implements graph-based constraint satisfaction to refine
neural network predictions using logical relationships between techniques.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import networkx as nx


class KnowledgeGraph:
    """
    Implements knowledge graph reasoning with three types of edges:
    1. Prerequisite edges: Techniques that require certain conditions
    2. Co-occurrence edges: Techniques that often appear together
    3. Mutual exclusion edges: Techniques that rarely coexist
    """

    def __init__(self, techniques_path="propaganda_techniques.json"):
        """Initialize knowledge graph from configuration."""
        with open(techniques_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.techniques = {t['name']: t for t in self.config['propaganda_techniques']}
        self.technique_names = list(self.techniques.keys())

        # Build graph structure
        self.graph = nx.DiGraph()
        self._build_graph()

        # Co-occurrence and mutual exclusion matrices
        self._build_relationship_matrices()

    def _build_graph(self):
        """Build the knowledge graph with hierarchical and dependency edges."""
        # Add all techniques as nodes
        for name, data in self.techniques.items():
            self.graph.add_node(name, **data)

        # Add hierarchical edges (category -> technique)
        for category, techniques in self.config['hierarchical_structure'].items():
            for technique in techniques:
                if technique in self.technique_names:
                    self.graph.add_edge(category, technique, type='hierarchical', weight=1.0)

        # Add prerequisite edges (implied by check_method requirements)
        self._add_prerequisite_edges()

    def _add_prerequisite_edges(self):
        """Add edges representing prerequisite relationships."""
        # Some techniques enable or require others
        # Appeal to Authority often enables Ethos
        # Loaded Language often co-occurs with Name Calling

        prerequisite_rules = {
            'Appeal to Authority': ['Ethos'],
            'Name Calling/Labeling': ['Ethos'],
            'Smears': ['Ethos'],
            'Reductio ad Hitlerum': ['Ethos'],
            'Appeal to (Strong) Emotions': ['Pathos'],
            'Appeal to Fear/Prejudice': ['Pathos'],
            'Flag-waving': ['Pathos'],
            'Transfer': ['Pathos'],
        }

        for technique, prerequisites in prerequisite_rules.items():
            if technique in self.technique_names:
                for prereq in prerequisites:
                    if prereq in self.technique_names or prereq in self.config['hierarchical_structure']:
                        self.graph.add_edge(technique, prereq, type='prerequisite', weight=0.8)

    def _build_relationship_matrices(self):
        """Build co-occurrence and mutual exclusion matrices."""
        n = len(self.technique_names)
        self.co_occurrence_matrix = np.zeros((n, n))
        self.mutual_exclusion_matrix = np.zeros((n, n))

        # Create index mapping
        self.technique_to_idx = {name: i for i, name in enumerate(self.technique_names)}

        # Fill co-occurrence matrix from configuration
        for cluster in self.config['co_occurrence_clusters']:
            for tech1 in cluster:
                for tech2 in cluster:
                    if tech1 != tech2 and tech1 in self.technique_to_idx and tech2 in self.technique_to_idx:
                        i, j = self.technique_to_idx[tech1], self.technique_to_idx[tech2]
                        # High co-occurrence score (0.85 means they appear together 85% of the time)
                        self.co_occurrence_matrix[i][j] = 0.85
                        self.co_occurrence_matrix[j][i] = 0.85

        # Fill mutual exclusion matrix
        for tech1, tech2 in self.config['mutual_exclusions']:
            if tech1 in self.technique_to_idx and tech2 in self.technique_to_idx:
                i, j = self.technique_to_idx[tech1], self.technique_to_idx[tech2]
                # Mutual exclusion (techniques rarely appear together)
                self.mutual_exclusion_matrix[i][j] = 1.0
                self.mutual_exclusion_matrix[j][i] = 1.0

    def apply_prerequisite_filtering(
        self,
        neural_predictions: Dict[str, float],
        prerequisite_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply hard filtering: reject techniques whose prerequisites are not satisfied.

        Args:
            neural_predictions: Technique -> probability from neural model
            prerequisite_scores: Technique -> prerequisite satisfaction (0 or 1)

        Returns:
            Filtered predictions
        """
        filtered = {}

        for technique, prob in neural_predictions.items():
            if prerequisite_scores.get(technique, 0.0) > 0.5:
                # Prerequisite satisfied, keep prediction
                filtered[technique] = prob
            else:
                # Prerequisite not satisfied, heavily penalize
                filtered[technique] = prob * 0.1  # Keep small probability in case of false negative

        return filtered

    def apply_co_occurrence_boost(
        self,
        predictions: Dict[str, float],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Boost probabilities of techniques that often co-occur with high-confidence predictions.

        Args:
            predictions: Technique -> probability
            threshold: Minimum probability to be considered "detected"

        Returns:
            Boosted predictions
        """
        boosted = predictions.copy()

        # Find high-confidence predictions
        detected = {tech for tech, prob in predictions.items() if prob >= threshold}

        for technique in self.technique_names:
            if technique not in detected and technique in predictions:
                # Check if this technique co-occurs with detected ones
                idx = self.technique_to_idx[technique]
                boost_score = 0.0

                for detected_tech in detected:
                    detected_idx = self.technique_to_idx[detected_tech]
                    co_occurrence = self.co_occurrence_matrix[idx][detected_idx]

                    if co_occurrence > 0.7:  # Strong co-occurrence
                        # Boost based on detected technique's confidence and co-occurrence strength
                        boost = predictions[detected_tech] * co_occurrence * 0.3
                        boost_score = max(boost_score, boost)

                # Apply boost
                boosted[technique] = min(predictions[technique] + boost_score, 1.0)

        return boosted

    def apply_mutual_exclusion_penalty(
        self,
        predictions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Penalize techniques that are mutually exclusive with high-confidence predictions.

        Args:
            predictions: Technique -> probability

        Returns:
            Penalized predictions
        """
        penalized = predictions.copy()

        for tech1, prob1 in predictions.items():
            if prob1 < 0.3:  # Skip low-confidence predictions
                continue

            idx1 = self.technique_to_idx[tech1]

            for tech2, prob2 in predictions.items():
                if tech1 == tech2:
                    continue

                idx2 = self.technique_to_idx[tech2]
                exclusion = self.mutual_exclusion_matrix[idx1][idx2]

                if exclusion > 0.8:  # Strong mutual exclusion
                    # Keep the one with higher confidence, penalize the other
                    if prob1 > prob2:
                        penalty = prob1 * exclusion * 0.5
                        penalized[tech2] = max(penalized[tech2] - penalty, 0.0)
                    else:
                        penalty = prob2 * exclusion * 0.5
                        penalized[tech1] = max(penalized[tech1] - penalty, 0.0)

        return penalized

    def enforce_hierarchical_consistency(
        self,
        predictions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Ensure hierarchical consistency: if a fine-grained technique is predicted,
        its parent category should also be predicted.

        Args:
            predictions: Technique -> probability

        Returns:
            Hierarchically consistent predictions
        """
        consistent = predictions.copy()

        # For each high-level category
        for category, child_techniques in self.config['hierarchical_structure'].items():
            # Find maximum probability among children
            child_probs = [predictions.get(tech, 0.0) for tech in child_techniques
                          if tech in predictions]

            if child_probs:
                max_child_prob = max(child_probs)

                # If any child has high probability, ensure parent category is also high
                if category in predictions:
                    # Parent should be at least as confident as most confident child
                    if predictions[category] < max_child_prob:
                        consistent[category] = max_child_prob

        # Reverse: if parent is high but all children are low, boost highest child
        for category, child_techniques in self.config['hierarchical_structure'].items():
            if category in predictions and predictions[category] > 0.7:
                child_probs = {tech: predictions.get(tech, 0.0)
                              for tech in child_techniques if tech in predictions}

                if child_probs:
                    max_child = max(child_probs, key=child_probs.get)
                    max_child_prob = child_probs[max_child]

                    # If parent is confident but no child is confident, boost the highest child
                    if max_child_prob < 0.3:
                        # Boost the most likely child
                        boost_amount = (predictions[category] - max_child_prob) * 0.5
                        consistent[max_child] = min(max_child_prob + boost_amount, 0.9)

        return consistent

    def reason(
        self,
        neural_predictions: Dict[str, float],
        prerequisite_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply full knowledge graph reasoning pipeline.

        Args:
            neural_predictions: Raw predictions from neural model
            prerequisite_scores: Prerequisite satisfaction scores (0-1)

        Returns:
            Refined predictions after knowledge graph reasoning
        """
        # Stage 1: Hard prerequisite filtering
        predictions = self.apply_prerequisite_filtering(neural_predictions, prerequisite_scores)

        # Stage 2: Co-occurrence boosting
        predictions = self.apply_co_occurrence_boost(predictions, threshold=0.5)

        # Stage 3: Mutual exclusion penalty
        predictions = self.apply_mutual_exclusion_penalty(predictions)

        # Stage 4: Hierarchical consistency
        predictions = self.enforce_hierarchical_consistency(predictions)

        return predictions

    def propose_missing_techniques(
        self,
        neural_predictions: Dict[str, float],
        prerequisite_scores: Dict[str, float],
        min_prerequisite_score: float = 0.8
    ) -> Dict[str, float]:
        """
        Propose techniques that the neural model missed but have strong prerequisites.

        This handles cases where neural model has low recall for rare techniques.

        Args:
            neural_predictions: Predictions from neural model
            prerequisite_scores: Prerequisite satisfaction scores
            min_prerequisite_score: Minimum score to propose a technique

        Returns:
            Dictionary of proposed techniques with confidence scores
        """
        proposals = {}

        for technique, prereq_score in prerequisite_scores.items():
            neural_score = neural_predictions.get(technique, 0.0)

            # If prerequisite is strongly satisfied but neural prediction is low
            if prereq_score >= min_prerequisite_score and neural_score < 0.3:
                # Propose this technique with confidence based on prerequisite strength
                proposals[technique] = prereq_score * 0.6  # Max 0.6 confidence

                # Check if co-occurring techniques are also present
                idx = self.technique_to_idx[technique]
                for other_tech, other_prob in neural_predictions.items():
                    if other_prob > 0.5:  # Other technique is confidently detected
                        other_idx = self.technique_to_idx[other_tech]
                        co_occurrence = self.co_occurrence_matrix[idx][other_idx]

                        if co_occurrence > 0.7:
                            # Boost proposal confidence
                            proposals[technique] = min(
                                proposals[technique] + (co_occurrence * 0.2),
                                0.85
                            )

        return proposals

    def get_explanation(
        self,
        technique: str,
        neural_score: float,
        prerequisite_score: float,
        final_score: float,
        detected_techniques: List[str]
    ) -> str:
        """
        Generate human-readable explanation for why a technique was detected or rejected.

        Args:
            technique: Name of the technique
            neural_score: Original neural network score
            prerequisite_score: Prerequisite satisfaction score
            final_score: Final score after KG reasoning
            detected_techniques: List of other detected techniques

        Returns:
            Explanation string
        """
        explanation_parts = []

        # Prerequisite check
        if prerequisite_score < 0.5:
            explanation_parts.append(
                f"Prerequisites not satisfied (score: {prerequisite_score:.2f})"
            )
        else:
            explanation_parts.append(
                f"Prerequisites satisfied (score: {prerequisite_score:.2f})"
            )

        # Neural confidence
        explanation_parts.append(f"Neural model confidence: {neural_score:.2f}")

        # Co-occurrence boost/penalty
        idx = self.technique_to_idx[technique]
        co_occurring = []
        for other in detected_techniques:
            if other != technique:
                other_idx = self.technique_to_idx[other]
                co_score = self.co_occurrence_matrix[idx][other_idx]
                if co_score > 0.7:
                    co_occurring.append((other, co_score))

        if co_occurring:
            co_occurring_names = ', '.join([f"{name} ({score:.2f})" for name, score in co_occurring])
            explanation_parts.append(f"Co-occurs with: {co_occurring_names}")

        # Final decision
        if final_score > 0.5:
            explanation_parts.append(f"✓ DETECTED (final score: {final_score:.2f})")
        else:
            explanation_parts.append(f"✗ NOT DETECTED (final score: {final_score:.2f})")

        return " | ".join(explanation_parts)


class ConstraintSatisfactionSolver:
    """
    Implements constraint satisfaction for propaganda technique detection.
    Finds the optimal set of techniques that maximizes agreement with neural
    predictions while satisfying logical constraints.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        """Initialize with a knowledge graph."""
        self.kg = knowledge_graph

    def solve(
        self,
        neural_predictions: Dict[str, float],
        prerequisite_scores: Dict[str, float],
        alpha: float = 0.7,
        beta: float = 0.3
    ) -> Dict[str, float]:
        """
        Solve constraint satisfaction problem using iterative refinement.

        Args:
            neural_predictions: Neural model predictions
            prerequisite_scores: Prerequisite satisfaction scores
            alpha: Weight for neural predictions (deviation penalty)
            beta: Weight for constraint violations

        Returns:
            Optimized predictions satisfying constraints
        """
        # Start with neural predictions
        current = neural_predictions.copy()

        # Iteratively refine over multiple passes
        max_iterations = 5
        for iteration in range(max_iterations):
            previous = current.copy()

            # Apply KG reasoning
            current = self.kg.reason(current, prerequisite_scores)

            # Check convergence
            max_change = max(abs(current.get(tech, 0) - previous.get(tech, 0))
                           for tech in self.kg.technique_names)

            if max_change < 0.01:  # Converged
                break

        return current
