"""
Stream 3: LLM Symbolic Reasoner
Verification and Explanation using Large Language Models

This stream provides:
1. Verification of ambiguous predictions
2. Structured reasoning for each technique
3. Evidence extraction from content
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import LLMConfig


@dataclass
class LLMVerification:
    """Result of LLM verification for a technique"""
    technique: str
    score: float  # 0.0 to 1.0
    supported: bool
    reasoning: str
    evidence: List[str]
    missing_evidence: List[str]


class LLMSymbolicStream:
    """
    LLM-based verification stream for propaganda techniques
    Uses structured prompting to get reasoning and evidence
    """

    def __init__(
        self,
        config: LLMConfig,
        technique_definitions: Dict[str, str]
    ):
        """
        Args:
            config: LLM configuration
            technique_definitions: Dict mapping technique name to definition
        """
        self.config = config
        self.technique_definitions = technique_definitions

        if config.use_llm:
            self._initialize_llm_client()
            print(f"✓ LLM Stream initialized with {config.provider}/{config.model_name}")
        else:
            self.llm_client = None
            print("⚠ LLM Stream disabled (use_llm=False)")

    def _initialize_llm_client(self):
        """Initialize LLM client based on provider"""
        if self.config.provider == "openai":
            self._init_openai()
        elif self.config.provider == "anthropic":
            self._init_anthropic()
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.provider}")

    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            import openai
            openai.api_key = self.config.api_key
            self.llm_client = openai
            print("  OpenAI client initialized")
        except ImportError:
            print("⚠ OpenAI package not installed. Run: pip install openai")
            self.llm_client = None

    def _init_anthropic(self):
        """Initialize Anthropic Claude client"""
        try:
            import anthropic
            self.llm_client = anthropic.Anthropic(api_key=self.config.api_key)
            print("  Anthropic client initialized")
        except ImportError:
            print("⚠ Anthropic package not installed. Run: pip install anthropic")
            self.llm_client = None

    def select_techniques_for_verification(
        self,
        neural_predictions: Dict[str, float],
        kg_predictions: Dict[str, float]
    ) -> List[str]:
        """
        Select which techniques need LLM verification

        Criteria:
        1. High disagreement between neural and KG
        2. Medium confidence (uncertainty)
        3. Rare techniques with any positive signal

        Args:
            neural_predictions: Neural stream predictions
            kg_predictions: KG stream predictions

        Returns:
            List of technique names to verify
        """
        candidates = []
        rare_techniques = set(self.config.rare_techniques)

        for technique in self.technique_definitions.keys():
            neural_score = neural_predictions.get(technique, 0.0)
            kg_score = kg_predictions.get(technique, 0.0)

            # Criterion 1: High disagreement
            disagreement = abs(neural_score - kg_score)
            if disagreement > self.config.disagreement_threshold:
                candidates.append((technique, disagreement, 'disagreement'))

            # Criterion 2: Medium confidence (ambiguous)
            avg_score = (neural_score + kg_score) / 2
            min_conf, max_conf = self.config.medium_confidence_range
            if min_conf <= avg_score <= max_conf:
                candidates.append((technique, avg_score, 'medium_confidence'))

            # Criterion 3: Rare technique with positive signal
            if technique in rare_techniques and (neural_score > 0.15 or kg_score > 0.15):
                candidates.append((technique, max(neural_score, kg_score), 'rare'))

        # Remove duplicates, keep highest score
        unique_candidates = {}
        for tech, score, reason in candidates:
            if tech not in unique_candidates or score > unique_candidates[tech][0]:
                unique_candidates[tech] = (score, reason)

        # Sort by score and take top K
        sorted_candidates = sorted(
            unique_candidates.items(),
            key=lambda x: x[1][0],
            reverse=True
        )[:self.config.max_techniques_to_verify]

        selected_techniques = [tech for tech, _ in sorted_candidates]

        if selected_techniques and self.config.use_llm:
            print(f"  Selected {len(selected_techniques)} techniques for LLM verification")

        return selected_techniques

    def verify_techniques(
        self,
        text: str,
        image_description: str,
        techniques_to_verify: List[str],
        neural_predictions: Dict[str, float],
        kg_predictions: Dict[str, float]
    ) -> Dict[str, LLMVerification]:
        """
        Verify techniques using LLM

        Args:
            text: Meme text content
            image_description: Description of visual content
            techniques_to_verify: List of techniques to verify
            neural_predictions: Neural stream top predictions
            kg_predictions: KG stream top predictions

        Returns:
            Dictionary mapping technique to verification result
        """
        if not self.config.use_llm or not self.llm_client:
            # Return neutral verifications
            return {
                tech: LLMVerification(
                    technique=tech,
                    score=0.5,
                    supported=None,
                    reasoning="LLM verification disabled",
                    evidence=[],
                    missing_evidence=[]
                )
                for tech in techniques_to_verify
            }

        # Build prompt
        prompt = self._build_verification_prompt(
            text,
            image_description,
            techniques_to_verify,
            neural_predictions,
            kg_predictions
        )

        # Call LLM
        try:
            response = self._call_llm(prompt)
            verifications = self._parse_llm_response(response, techniques_to_verify)
            return verifications
        except Exception as e:
            print(f"⚠ LLM verification failed: {e}")
            # Return neutral verifications as fallback
            return {
                tech: LLMVerification(
                    technique=tech,
                    score=0.5,
                    supported=None,
                    reasoning=f"Error: {str(e)}",
                    evidence=[],
                    missing_evidence=[]
                )
                for tech in techniques_to_verify
            }

    def _build_verification_prompt(
        self,
        text: str,
        image_description: str,
        techniques: List[str],
        neural_preds: Dict[str, float],
        kg_preds: Dict[str, float]
    ) -> str:
        """Build structured prompt for LLM verification"""

        # Get top predictions from each stream
        top_neural = sorted(neural_preds.items(), key=lambda x: x[1], reverse=True)[:3]
        top_kg = sorted(kg_preds.items(), key=lambda x: x[1], reverse=True)[:3]

        # Build technique definitions
        tech_defs = []
        for tech in techniques:
            definition = self.technique_definitions.get(tech, "No definition available")
            tech_defs.append(f"**{tech}**: {definition}")

        tech_defs_str = "\n".join(tech_defs)

        # Build neural predictions
        neural_str = "\n".join([f"- {tech}: {score:.2f}" for tech, score in top_neural])

        # Build KG predictions
        kg_str = "\n".join([f"- {tech}: {score:.2f}" for tech, score in top_kg])

        prompt = f"""You are an expert in propaganda technique detection. Analyze the following meme content and verify which propaganda techniques are present.

**Meme Text:**
{text}

**Visual Content:**
{image_description}

**Neural Model Top Predictions:**
{neural_str}

**Knowledge Graph Top Predictions:**
{kg_str}

**Techniques to Verify:**
{tech_defs_str}

For each technique listed above, provide a structured verification:

1. **Score** (0.0 to 1.0): Your confidence that this technique is present
2. **Supported** (true/false): Whether you support detecting this technique
3. **Reasoning**: Explain WHY you believe it is present or absent
4. **Evidence**: Quote specific words/phrases from the text that support the technique
5. **Missing Evidence**: What evidence would be needed if the technique is NOT present

Be strict and precise. Only confirm a technique if there is clear evidence based on the definition.

Respond in this exact JSON format:
{{
  "technique_name_1": {{
    "score": 0.0-1.0,
    "supported": true/false,
    "reasoning": "...",
    "evidence": ["quote1", "quote2"],
    "missing_evidence": ["what's missing"]
  }},
  "technique_name_2": {{
    ...
  }}
}}
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API"""
        if self.config.provider == "openai":
            return self._call_openai(prompt)
        elif self.config.provider == "anthropic":
            return self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = self.llm_client.ChatCompletion.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"}  # Force JSON output
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API"""
        message = self.llm_client.messages.create(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def _parse_llm_response(
        self,
        response: str,
        techniques: List[str]
    ) -> Dict[str, LLMVerification]:
        """Parse LLM JSON response into verification objects"""
        try:
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)

            parsed = json.loads(response)

            verifications = {}
            for tech in techniques:
                if tech in parsed:
                    data = parsed[tech]
                    verifications[tech] = LLMVerification(
                        technique=tech,
                        score=float(data.get('score', 0.5)),
                        supported=bool(data.get('supported', False)),
                        reasoning=str(data.get('reasoning', '')),
                        evidence=data.get('evidence', []),
                        missing_evidence=data.get('missing_evidence', [])
                    )
                else:
                    # Technique not in response, use neutral
                    verifications[tech] = LLMVerification(
                        technique=tech,
                        score=0.5,
                        supported=None,
                        reasoning="Not verified by LLM",
                        evidence=[],
                        missing_evidence=[]
                    )

            return verifications

        except json.JSONDecodeError as e:
            print(f"⚠ Failed to parse LLM response as JSON: {e}")
            print(f"Response: {response[:200]}...")
            # Return neutral verifications
            return {
                tech: LLMVerification(
                    technique=tech,
                    score=0.5,
                    supported=None,
                    reasoning="Parse error",
                    evidence=[],
                    missing_evidence=[]
                )
                for tech in techniques
            }

    def predict(
        self,
        text: str,
        image_description: str,
        neural_predictions: Dict[str, float],
        kg_predictions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Get LLM predictions (scores only)

        Args:
            text: Meme text
            image_description: Visual description
            neural_predictions: Neural stream predictions
            kg_predictions: KG stream predictions

        Returns:
            Dictionary mapping technique -> LLM score
        """
        # Select techniques to verify
        techniques_to_verify = self.select_techniques_for_verification(
            neural_predictions, kg_predictions
        )

        if not techniques_to_verify or not self.config.use_llm:
            # Return neutral scores for all techniques
            return {tech: 0.5 for tech in self.technique_definitions.keys()}

        # Verify selected techniques
        verifications = self.verify_techniques(
            text,
            image_description,
            techniques_to_verify,
            neural_predictions,
            kg_predictions
        )

        # Convert to score dictionary
        scores = {}
        for tech in self.technique_definitions.keys():
            if tech in verifications:
                scores[tech] = verifications[tech].score
            else:
                scores[tech] = 0.5  # Neutral for non-verified techniques

        return scores


if __name__ == "__main__":
    # Test LLM stream
    from config import LLMConfig
    import json

    print("Testing LLM Symbolic Stream...")

    # Load technique definitions
    with open("propaganda_techniques.json", 'r') as f:
        config_data = json.load(f)

    technique_defs = {
        t['name']: t['prerequisite']
        for t in config_data['propaganda_techniques']
    }

    # Create config (LLM disabled for testing)
    config = LLMConfig()
    config.use_llm = False  # Don't actually call LLM in test

    stream = LLMSymbolicStream(config, technique_defs)

    # Test data
    neural_preds = {
        'Smears': 0.85,
        'Loaded Language': 0.75,
        'Whataboutism': 0.40,
    }

    kg_preds = {
        'Smears': 0.90,
        'Loaded Language': 0.80,
        'Whataboutism': 0.10,  # Disagrees with neural
    }

    # Select techniques
    to_verify = stream.select_techniques_for_verification(neural_preds, kg_preds)

    print(f"\n✓ Selected {len(to_verify)} techniques for verification:")
    for tech in to_verify:
        print(f"  - {tech}")

    # Get predictions (will be neutral since LLM is disabled)
    predictions = stream.predict(
        "Test text",
        "Test image",
        neural_preds,
        kg_preds
    )

    print(f"\n✓ LLM predictions generated (neutral scores since disabled)")
