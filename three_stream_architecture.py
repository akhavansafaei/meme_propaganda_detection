"""
Three-Stream Architecture for Propaganda Detection

Combines:
1. Neural Multimodal Stream (CLIP + RoBERTa): Pattern recognition
2. Knowledge Graph Stream: Rule-based reasoning with constraints
3. LLM Symbolic Reasoner Stream: High-level semantic verification

Each stream provides complementary evidence that is fused for final prediction.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from prerequisite_checker import PrerequisiteChecker
from knowledge_graph import KnowledgeGraph, ConstraintSatisfactionSolver


class ThreeStreamArchitecture:
    """
    Implements the three-stream architecture for propaganda technique detection.
    """

    def __init__(
        self,
        techniques_path: str = "propaganda_techniques.json",
        use_llm: bool = True,
        llm_client = None
    ):
        """
        Initialize the three-stream architecture.

        Args:
            techniques_path: Path to propaganda techniques configuration
            use_llm: Whether to use LLM verification stream
            llm_client: LLM client for verification (e.g., OpenAI, Anthropic)
        """
        # Load configuration
        with open(techniques_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.technique_names = [t['name'] for t in self.config['propaganda_techniques']]

        # Initialize components
        self.prerequisite_checker = PrerequisiteChecker(techniques_path)
        self.knowledge_graph = KnowledgeGraph(techniques_path)
        self.constraint_solver = ConstraintSatisfactionSolver(self.knowledge_graph)

        self.use_llm = use_llm
        self.llm_client = llm_client

        # Fusion weights (can be learned from validation set)
        self.fusion_weights = {
            'alpha': 0.4,  # Neural stream weight
            'beta': 0.4,   # Knowledge graph weight
            'gamma': 0.2   # LLM weight
        }

    def predict(
        self,
        text: str,
        image_features: Dict,
        neural_predictions: Dict[str, float],
        return_explanations: bool = False
    ) -> Dict[str, any]:
        """
        Run full three-stream prediction pipeline.

        Args:
            text: Meme text content (caption + OCR)
            image_features: Extracted image features (entities, symbols, etc.)
            neural_predictions: Raw predictions from neural model (technique -> prob)
            return_explanations: Whether to return detailed explanations

        Returns:
            Dictionary containing:
                - 'final_predictions': Final probabilities after fusion
                - 'stream_predictions': Predictions from each stream
                - 'detected_techniques': List of detected techniques
                - 'explanations': Explanations for each technique (if requested)
        """
        # Stream 1: Neural Multimodal Stream (already computed)
        neural_scores = neural_predictions

        # Stream 2: Knowledge Graph Reasoning
        kg_scores = self._run_knowledge_graph_stream(
            text, image_features, neural_scores
        )

        # Stream 3: LLM Symbolic Reasoning (selective)
        llm_scores = None
        if self.use_llm and self.llm_client:
            llm_scores = self._run_llm_stream(
                text, image_features, neural_scores, kg_scores
            )

        # Fusion
        final_scores = self._fuse_streams(
            neural_scores, kg_scores, llm_scores
        )

        # Thresholding and final detection
        detected_techniques = self._apply_thresholds(final_scores)

        result = {
            'final_predictions': final_scores,
            'detected_techniques': detected_techniques,
            'stream_predictions': {
                'neural': neural_scores,
                'knowledge_graph': kg_scores,
                'llm': llm_scores
            }
        }

        if return_explanations:
            result['explanations'] = self._generate_explanations(
                text, image_features, neural_scores, kg_scores,
                llm_scores, final_scores, detected_techniques
            )

        return result

    def _run_knowledge_graph_stream(
        self,
        text: str,
        image_features: Dict,
        neural_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Run knowledge graph reasoning stream.

        This stream:
        1. Checks prerequisites for each technique
        2. Filters out techniques without prerequisites
        3. Applies co-occurrence boosting
        4. Enforces mutual exclusion
        5. Ensures hierarchical consistency
        """
        # Check prerequisites
        prerequisite_scores = self.prerequisite_checker.get_prerequisite_score(
            text, image_features
        )

        # Apply knowledge graph reasoning
        kg_refined = self.knowledge_graph.reason(
            neural_scores, prerequisite_scores
        )

        # Propose missing techniques (high recall for rare techniques)
        proposals = self.knowledge_graph.propose_missing_techniques(
            neural_scores, prerequisite_scores, min_prerequisite_score=0.8
        )

        # Merge proposals with refined scores
        for technique, proposed_score in proposals.items():
            if proposed_score > kg_refined.get(technique, 0.0):
                kg_refined[technique] = proposed_score

        return kg_refined

    def _run_llm_stream(
        self,
        text: str,
        image_features: Dict,
        neural_scores: Dict[str, float],
        kg_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Run LLM verification stream (selective invocation).

        LLM is called only for:
        1. Ambiguous cases (neural and KG disagree)
        2. Rare techniques
        3. Medium confidence predictions (0.3-0.7)
        """
        llm_scores = {}

        # Decide which techniques need LLM verification
        techniques_to_verify = self._select_techniques_for_llm(
            neural_scores, kg_scores
        )

        if not techniques_to_verify:
            # No LLM verification needed, return empty dict
            return {tech: 0.0 for tech in self.technique_names}

        # Call LLM for verification
        llm_responses = self._call_llm_verification(
            text, image_features, techniques_to_verify
        )

        # Parse LLM responses into scores
        for technique in self.technique_names:
            if technique in llm_responses:
                llm_scores[technique] = llm_responses[technique]['score']
            else:
                llm_scores[technique] = 0.0

        return llm_scores

    def _select_techniques_for_llm(
        self,
        neural_scores: Dict[str, float],
        kg_scores: Dict[str, float],
        max_techniques: int = 10
    ) -> List[str]:
        """
        Select which techniques need LLM verification.

        Criteria:
        1. High disagreement between neural and KG
        2. Medium confidence (0.3-0.7)
        3. Rare techniques with any positive signal
        """
        candidates = []

        # Define rare techniques (low support in training data)
        rare_techniques = [
            'Whataboutism', 'Red Herring', 'Bandwagon', 'Repetition',
            'Misrepresentation (Straw Man)', 'Reductio ad Hitlerum',
            'Thought-Terminating Cliche', 'Obfuscation/Vagueness'
        ]

        for technique in self.technique_names:
            neural_score = neural_scores.get(technique, 0.0)
            kg_score = kg_scores.get(technique, 0.0)

            # Criterion 1: High disagreement
            disagreement = abs(neural_score - kg_score)
            if disagreement > 0.3:
                candidates.append((technique, disagreement, 'disagreement'))

            # Criterion 2: Medium confidence
            avg_score = (neural_score + kg_score) / 2
            if 0.3 <= avg_score <= 0.7:
                candidates.append((technique, avg_score, 'medium_confidence'))

            # Criterion 3: Rare technique with positive signal
            if technique in rare_techniques and (neural_score > 0.15 or kg_score > 0.15):
                candidates.append((technique, max(neural_score, kg_score), 'rare'))

        # Remove duplicates and sort by priority
        unique_candidates = {}
        for tech, score, reason in candidates:
            if tech not in unique_candidates or score > unique_candidates[tech][0]:
                unique_candidates[tech] = (score, reason)

        # Sort by score and take top K
        sorted_candidates = sorted(
            unique_candidates.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:max_techniques]

        return [tech for tech, _ in sorted_candidates]

    def _call_llm_verification(
        self,
        text: str,
        image_features: Dict,
        techniques_to_verify: List[str]
    ) -> Dict[str, Dict]:
        """
        Call LLM to verify specific techniques.

        Returns:
            Dictionary mapping technique to verification result:
            {
                'technique_name': {
                    'score': 0.0-1.0,
                    'supported': True/False,
                    'reasoning': 'explanation...',
                    'evidence': ['evidence1', 'evidence2']
                }
            }
        """
        if not self.llm_client:
            # No LLM client available, return neutral scores
            return {tech: {'score': 0.5, 'supported': None, 'reasoning': 'No LLM available'}
                   for tech in techniques_to_verify}

        # Build prompt for LLM
        prompt = self._build_llm_prompt(text, image_features, techniques_to_verify)

        # Call LLM (implementation depends on LLM client)
        try:
            response = self._invoke_llm(prompt)
            parsed_response = self._parse_llm_response(response, techniques_to_verify)
            return parsed_response
        except Exception as e:
            print(f"LLM verification failed: {e}")
            return {tech: {'score': 0.5, 'supported': None, 'reasoning': f'Error: {e}'}
                   for tech in techniques_to_verify}

    def _build_llm_prompt(
        self,
        text: str,
        image_features: Dict,
        techniques_to_verify: List[str]
    ) -> str:
        """Build structured prompt for LLM verification."""
        # Get technique definitions
        technique_defs = []
        for tech_name in techniques_to_verify:
            tech_data = self.prerequisite_checker.techniques[tech_name]
            definition = tech_data.get('prerequisite', 'No definition available')
            technique_defs.append(f"- **{tech_name}**: {definition}")

        technique_defs_str = "\n".join(technique_defs)

        # Build image description
        image_desc = self._describe_image_features(image_features)

        prompt = f"""You are an expert in propaganda technique detection. Analyze the following meme content and determine which propaganda techniques are present.

**Meme Text:**
{text}

**Visual Content:**
{image_desc}

**Techniques to Verify:**
{technique_defs_str}

For each technique, provide:
1. A score from 0.0 to 1.0 indicating confidence that the technique is present
2. Whether you support the detection (true/false)
3. Your reasoning
4. Specific evidence from the text or image

Respond in JSON format:
{{
  "technique_name_1": {{
    "score": 0.0-1.0,
    "supported": true/false,
    "reasoning": "explanation...",
    "evidence": ["evidence1", "evidence2"]
  }},
  ...
}}
"""
        return prompt

    def _describe_image_features(self, image_features: Dict) -> str:
        """Convert image features dict to human-readable description."""
        parts = []

        if 'scene' in image_features:
            parts.append(f"Scene: {image_features['scene']}")

        if 'entities' in image_features and image_features['entities']:
            entities_str = ', '.join([e.get('text', '') for e in image_features['entities']])
            parts.append(f"Entities detected: {entities_str}")

        if 'symbols' in image_features and image_features['symbols']:
            symbols_str = ', '.join(image_features['symbols'])
            parts.append(f"Symbols: {symbols_str}")

        if 'faces' in image_features:
            parts.append(f"Number of faces: {image_features['faces']}")

        if 'emotion' in image_features:
            parts.append(f"Overall emotion: {image_features['emotion']}")

        if 'objects' in image_features and image_features['objects']:
            objects_str = ', '.join(image_features['objects'][:5])  # Top 5
            parts.append(f"Key objects: {objects_str}")

        return "\n".join(parts) if parts else "No significant visual features detected."

    def _invoke_llm(self, prompt: str) -> str:
        """
        Invoke LLM with the given prompt.

        This is a placeholder - actual implementation depends on LLM client.
        For OpenAI: use openai.ChatCompletion.create()
        For Anthropic Claude: use anthropic.messages.create()
        """
        if not self.llm_client:
            return "{}"

        # Example for OpenAI (uncomment and modify as needed):
        # response = self.llm_client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}],
        #     response_format={"type": "json_object"}
        # )
        # return response.choices[0].message.content

        # Example for Anthropic Claude:
        # response = self.llm_client.messages.create(
        #     model="claude-3-opus-20240229",
        #     max_tokens=2048,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response.content[0].text

        # Placeholder: return empty JSON
        return "{}"

    def _parse_llm_response(
        self,
        response: str,
        techniques_to_verify: List[str]
    ) -> Dict[str, Dict]:
        """Parse LLM JSON response."""
        try:
            parsed = json.loads(response)
            return parsed
        except json.JSONDecodeError:
            # Fallback: return neutral scores
            return {tech: {'score': 0.5, 'supported': None, 'reasoning': 'Parse error'}
                   for tech in techniques_to_verify}

    def _fuse_streams(
        self,
        neural_scores: Dict[str, float],
        kg_scores: Dict[str, float],
        llm_scores: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Fuse predictions from all streams using weighted fusion with veto logic.

        Strategy: LLM and KG have veto power for illogical predictions.
        """
        final_scores = {}
        alpha = self.fusion_weights['alpha']
        beta = self.fusion_weights['beta']
        gamma = self.fusion_weights['gamma']

        for technique in self.technique_names:
            neural_score = neural_scores.get(technique, 0.0)
            kg_score = kg_scores.get(technique, 0.0)
            llm_score = llm_scores.get(technique, 0.5) if llm_scores else 0.5

            # Veto logic
            # If LLM strongly rejects (score < 0.2), heavily penalize
            if llm_scores and llm_score < 0.2:
                final_score = 0.1 * neural_score
            # If KG rejects due to missing prerequisites (score == 0), penalize
            elif kg_score == 0.0:
                final_score = 0.1 * neural_score
            # Otherwise, weighted fusion
            else:
                if llm_scores:
                    final_score = alpha * neural_score + beta * kg_score + gamma * llm_score
                else:
                    # No LLM, redistribute weights
                    final_score = 0.5 * neural_score + 0.5 * kg_score

            final_scores[technique] = max(0.0, min(1.0, final_score))

        return final_scores

    def _apply_thresholds(
        self,
        final_scores: Dict[str, float]
    ) -> List[str]:
        """
        Apply per-technique thresholds to get final detections.

        Thresholds are optimized per technique on validation set.
        For now, use technique-specific heuristic thresholds.
        """
        # Default threshold
        default_threshold = 0.35

        # Technique-specific thresholds
        # Lower for rare techniques (boost recall)
        # Higher for common techniques (boost precision)
        thresholds = {
            # Rare techniques - lower threshold
            'Whataboutism': 0.20,
            'Red Herring': 0.20,
            'Bandwagon': 0.20,
            'Repetition': 0.15,
            'Reductio ad Hitlerum': 0.25,
            'Thought-Terminating Cliche': 0.25,

            # Common techniques - higher threshold
            'Loaded Language': 0.45,
            'Appeal to (Strong) Emotions': 0.40,
            'Name Calling/Labeling': 0.40,

            # Medium threshold for others
            'default': default_threshold
        }

        detected = []
        for technique, score in final_scores.items():
            threshold = thresholds.get(technique, thresholds['default'])
            if score >= threshold:
                detected.append(technique)

        return detected

    def _generate_explanations(
        self,
        text: str,
        image_features: Dict,
        neural_scores: Dict[str, float],
        kg_scores: Dict[str, float],
        llm_scores: Optional[Dict[str, float]],
        final_scores: Dict[str, float],
        detected_techniques: List[str]
    ) -> Dict[str, str]:
        """Generate human-readable explanations for each technique."""
        explanations = {}

        # Get prerequisite scores
        prerequisite_scores = self.prerequisite_checker.get_prerequisite_score(
            text, image_features
        )

        for technique in self.technique_names:
            neural_score = neural_scores.get(technique, 0.0)
            kg_score = kg_scores.get(technique, 0.0)
            llm_score = llm_scores.get(technique, 0.5) if llm_scores else None
            prereq_score = prerequisite_scores.get(technique, 0.0)
            final_score = final_scores.get(technique, 0.0)

            explanation = self.knowledge_graph.get_explanation(
                technique, neural_score, prereq_score, final_score, detected_techniques
            )

            # Add LLM reasoning if available
            if llm_scores and technique in llm_scores:
                explanation += f" | LLM: {llm_score:.2f}"

            explanations[technique] = explanation

        return explanations

    def optimize_fusion_weights(
        self,
        validation_data: List[Dict],
        metric: str = 'f1'
    ):
        """
        Optimize fusion weights on validation data.

        Args:
            validation_data: List of samples with ground truth
            metric: Optimization metric ('f1', 'precision', 'recall')

        This would implement grid search or gradient-based optimization
        to find optimal alpha, beta, gamma weights.
        """
        # TODO: Implement weight optimization
        # For now, use default weights
        pass


# Utility functions for integration

def extract_image_features_placeholder(image_path: str) -> Dict:
    """
    Placeholder for image feature extraction.

    In actual implementation, this would:
    1. Use object detection (Faster R-CNN, YOLO) for objects
    2. Use face detection for faces
    3. Use symbol/logo detection for symbols
    4. Use scene classification for scene description
    5. Use visual emotion recognition
    6. Use OCR for text in image

    Returns:
        Dictionary with extracted features
    """
    return {
        'entities': [],
        'faces': 0,
        'symbols': [],
        'emotion': None,
        'scene': '',
        'objects': []
    }


def get_neural_predictions_placeholder(text: str, image_features: Dict) -> Dict[str, float]:
    """
    Placeholder for neural network predictions.

    In actual implementation, this would:
    1. Encode text with RoBERTa
    2. Encode image with CLIP
    3. Fuse multimodal representations
    4. Pass through classification head
    5. Return probabilities for each technique

    Returns:
        Dictionary mapping technique name to probability
    """
    # Return uniform low probabilities as placeholder
    technique_names = [
        'Appeal to (Strong) Emotions', 'Appeal to Authority',
        'Appeal to Fear/Prejudice', 'Bandwagon', 'Black-and-White Fallacy',
        'Causal Oversimplification', 'Doubt', 'Exaggeration/Minimisation',
        'Flag-waving', 'Glittering Generalities', 'Loaded Language',
        'Misrepresentation (Straw Man)', 'Name Calling/Labeling',
        'Obfuscation/Vagueness', 'Red Herring', 'Reductio ad Hitlerum',
        'Repetition', 'Slogans', 'Smears', 'Thought-Terminating Cliche',
        'Transfer', 'Whataboutism'
    ]
    return {tech: 0.1 for tech in technique_names}
