"""
Prerequisite Checker Module for Propaganda Technique Detection

This module implements rule-based prerequisite checking for each propaganda technique.
Before a technique can be classified, its prerequisites must be satisfied.
"""

import re
import json
from typing import Dict, List, Set, Tuple
from collections import Counter
import numpy as np


class PrerequisiteChecker:
    """
    Checks if prerequisites for each propaganda technique are satisfied
    in the given content (text + image features).
    """

    def __init__(self, techniques_path="propaganda_techniques.json"):
        """Initialize with propaganda techniques configuration."""
        with open(techniques_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.techniques = {t['name']: t for t in self.config['propaganda_techniques']}
        self._load_lexicons()

    def _load_lexicons(self):
        """Load lexicons for different detection patterns."""
        # Emotion lexicon
        self.emotion_words = {
            'anger': ['outraged', 'furious', 'enraged', 'angry', 'mad', 'livid', 'hate'],
            'fear': ['terrified', 'afraid', 'scared', 'frightened', 'horrified', 'panic', 'dread'],
            'pride': ['proud', 'glory', 'honor', 'dignity', 'noble', 'heroic'],
            'sadness': ['devastated', 'heartbroken', 'tragic', 'mourning', 'grief']
        }

        # Fear/threat lexicon
        self.fear_lexicon = [
            'danger', 'threat', 'destroy', 'invasion', 'attack', 'eliminate',
            'violence', 'terror', 'crisis', 'catastrophe', 'disaster', 'ruin'
        ]

        # Uncertainty lexicon for Doubt
        self.uncertainty_lexicon = [
            'allegedly', 'supposedly', 'claim', 'so-called', 'maybe', 'perhaps',
            'doubt', 'question', 'uncertain', 'unclear', 'unconfirmed', 'rumor'
        ]

        # Extreme/intensity words
        self.intensity_words = {
            'extreme_positive': ['best', 'greatest', 'perfect', 'amazing', 'incredible', 'always'],
            'extreme_negative': ['worst', 'terrible', 'awful', 'disaster', 'never', 'horrible']
        }

        # Loaded language
        self.loaded_positive = [
            'hero', 'angel', 'savior', 'patriot', 'freedom fighter', 'defender'
        ]
        self.loaded_negative = [
            'terrorist', 'monster', 'enemy', 'traitor', 'criminal', 'thug',
            'radical', 'extremist', 'puppet', 'corrupt', 'evil'
        ]

        # Authority markers
        self.authority_markers = [
            'dr.', 'dr', 'prof.', 'professor', 'expert', 'scientist', 'researcher',
            'doctor', 'study', 'research', 'university', 'institute'
        ]

        # Collective/bandwagon markers
        self.collectivist_phrases = [
            'everyone', 'everybody', 'nobody', 'all of us', 'no one',
            'join the crowd', 'don\'t be left behind', 'millions of people'
        ]

        # Comparative patterns for Whataboutism
        self.comparative_patterns = [
            r'but what about',
            r'what about',
            r'while you',
            r'compared to',
            r'on the other hand',
            r'but you',
            r'however you'
        ]

        # Causal markers
        self.causal_markers = [
            'because', 'therefore', 'thus', 'hence', 'as a result',
            'leads to', 'causes', 'due to', 'consequently'
        ]

        # Binary/dichotomy markers
        self.binary_patterns = [
            r'either .+ or',
            r'with us or against us',
            r'only two (options|choices)',
            r'no middle ground',
            r'must choose'
        ]

        # Thought-terminating cliches
        self.cliche_phrases = [
            'it is what it is', 'boys will be boys', 'that\'s life',
            'end of story', 'period', 'case closed', 'it\'s always been this way'
        ]

        # Name calling labels
        self.negative_labels = [
            'idiot', 'fool', 'moron', 'stupid', 'dumb', 'radical',
            'extremist', 'puppet', 'sheep', 'snowflake', 'thug'
        ]

    def check_all_prerequisites(self, text: str, image_features: Dict = None) -> Dict[str, bool]:
        """
        Check prerequisites for all techniques.

        Args:
            text: The text content (meme caption + OCR text)
            image_features: Dictionary containing image analysis results
                - 'entities': List of detected entities (people, objects, symbols)
                - 'faces': Number of faces detected
                - 'symbols': List of detected symbols (flags, logos, etc.)
                - 'emotion': Detected visual emotion
                - 'scene': Scene description

        Returns:
            Dictionary mapping technique name to boolean (prerequisite satisfied)
        """
        if image_features is None:
            image_features = {}

        text_lower = text.lower()
        results = {}

        # Check each technique
        for technique_name, technique_data in self.techniques.items():
            method_name = f"_check_{technique_name.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}"

            # Call specific checker method if it exists
            if hasattr(self, method_name):
                checker = getattr(self, method_name)
                results[technique_name] = checker(text, text_lower, image_features)
            else:
                # Default: check for keywords
                results[technique_name] = self._check_keywords(
                    text_lower,
                    technique_data.get('keywords', [])
                )

        return results

    def _check_keywords(self, text_lower: str, keywords: List[str]) -> bool:
        """Generic keyword checker."""
        if not keywords:
            return True  # No specific keywords required
        return any(keyword.lower() in text_lower for keyword in keywords)

    # Specific checkers for each technique

    def _check_appeal_to_strong_emotions(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for emotionally charged content."""
        # Text-based emotion detection
        emotion_count = 0
        for emotion_type, words in self.emotion_words.items():
            if any(word in text_lower for word in words):
                emotion_count += 1

        # Check for visual emotion
        if img_features.get('emotion') in ['anger', 'fear', 'sadness']:
            emotion_count += 2

        # Check for exclamation marks (intensity)
        if text.count('!') >= 2:
            emotion_count += 1

        # Check for all caps words (shouting)
        caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 2)
        if caps_words >= 2:
            emotion_count += 1

        return emotion_count >= 2

    def _check_appeal_to_authority(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for authority figures or credentials."""
        # Check for authority markers
        has_authority_marker = any(marker in text_lower for marker in self.authority_markers)

        # Check for entities that might be authorities
        entities = img_features.get('entities', [])
        has_person_entity = any(e.get('type') == 'PERSON' for e in entities)
        has_org_entity = any(e.get('type') == 'ORG' for e in entities)

        # Check for citation patterns
        has_citation = bool(re.search(r'according to|says|states|claims', text_lower))

        return has_authority_marker or (has_citation and (has_person_entity or has_org_entity))

    def _check_appeal_to_fear_prejudice(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for fear-inducing or prejudiced content."""
        # Fear lexicon check
        fear_words = sum(1 for word in self.fear_lexicon if word in text_lower)

        # Check for group references (they, them, those people)
        has_group_ref = bool(re.search(r'\b(they|them|those people|these people)\b', text_lower))

        # Check for threatening visual content
        has_threatening_visual = img_features.get('emotion') == 'fear' or \
                                'weapon' in img_features.get('objects', [])

        return (fear_words >= 2) or (fear_words >= 1 and has_group_ref) or has_threatening_visual

    def _check_bandwagon(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for collective behavior appeals."""
        # Check for collectivist phrases
        has_collective = any(phrase in text_lower for phrase in self.collectivist_phrases)

        # Check for crowd imagery
        has_crowd = 'crowd' in img_features.get('scene', '').lower() or \
                   img_features.get('faces', 0) > 5

        # Check for popularity claims
        has_popularity = bool(re.search(r'(most|many) people', text_lower))

        return has_collective or has_crowd or has_popularity

    def _check_black_and_white_fallacy(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for binary/dichotomous framing."""
        # Check for binary patterns
        has_binary = any(re.search(pattern, text_lower) for pattern in self.binary_patterns)

        # Check for either/or structure
        has_either_or = 'either' in text_lower and 'or' in text_lower

        return has_binary or has_either_or

    def _check_causal_oversimplification(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for oversimplified causal claims."""
        # Check for causal markers
        has_causal = any(marker in text_lower for marker in self.causal_markers)

        # Check for simple causal structure (X causes Y)
        has_simple_causal = bool(re.search(r'\w+ (causes|leads to|results in) \w+', text_lower))

        return has_causal or has_simple_causal

    def _check_doubt(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for doubt-casting language."""
        # Check uncertainty lexicon
        uncertainty_count = sum(1 for word in self.uncertainty_lexicon if word in text_lower)

        # Check for questioning of credibility
        has_credibility_attack = bool(re.search(r'(really|actually) (believe|think)', text_lower))

        # Check for quotation marks around words (scare quotes)
        has_scare_quotes = text.count('"') >= 2 or text.count("'") >= 4

        return uncertainty_count >= 2 or has_credibility_attack or has_scare_quotes

    def _check_exaggeration_minimisation(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for extreme language."""
        # Check intensity words
        intensity_count = 0
        for category, words in self.intensity_words.items():
            if any(word in text_lower for word in words):
                intensity_count += 1

        # Check for superlatives
        has_superlative = bool(re.search(r'\b(most|least|best|worst)\b', text_lower))

        # Check for absolute terms
        has_absolute = bool(re.search(r'\b(always|never|all|none|every|no one)\b', text_lower))

        return intensity_count >= 1 or has_superlative or has_absolute

    def _check_flag_waving(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for nationalist/patriotic appeals."""
        # Check for patriotic keywords
        patriotic_words = ['patriot', 'country', 'nation', 'homeland', 'freedom', 'liberty']
        has_patriotic = any(word in text_lower for word in patriotic_words)

        # Check for flag or national symbols in image
        symbols = img_features.get('symbols', [])
        has_flag = 'flag' in symbols or 'national emblem' in symbols

        # Check for military imagery
        has_military = 'military' in img_features.get('scene', '') or \
                      'uniform' in img_features.get('objects', [])

        return has_patriotic or has_flag or has_military

    def _check_glittering_generalities(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for vague virtue words."""
        virtue_words = ['freedom', 'justice', 'honor', 'truth', 'democracy',
                       'progress', 'change', 'hope', 'peace', 'liberty']

        # Count virtue words
        virtue_count = sum(1 for word in virtue_words if word in text_lower)

        # Check if the text is vague (no specific details)
        word_count = len(text.split())
        has_numbers = bool(re.search(r'\d+', text))
        has_specifics = bool(re.search(r'(specifically|for example|such as)', text_lower))

        is_vague = word_count < 30 and not has_numbers and not has_specifics

        return virtue_count >= 1 and is_vague

    def _check_loaded_language(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for emotionally charged language."""
        # Count loaded words
        loaded_count = sum(1 for word in self.loaded_positive + self.loaded_negative
                          if word in text_lower)

        # Check for dehumanizing language (comparing people to animals/objects)
        dehumanizing = ['rat', 'pig', 'dog', 'vermin', 'parasite', 'virus']
        has_dehumanizing = any(word in text_lower for word in dehumanizing)

        return loaded_count >= 2 or has_dehumanizing

    def _check_misrepresentation_straw_man(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for straw man arguments."""
        # Check for reference to others' positions
        has_position_ref = bool(re.search(r'(they|you) (want|think|believe|say)', text_lower))

        # Check for exaggerated restatement
        has_exaggeration = bool(re.search(r'so (you\'re saying|you think)', text_lower))

        # Check for distortion markers
        has_distortion = bool(re.search(r'(basically|essentially) (saying|claiming)', text_lower))

        return has_position_ref and (has_exaggeration or has_distortion)

    def _check_name_calling_labeling(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for name-calling."""
        # Check for negative labels
        has_negative_label = any(label in text_lower for label in self.negative_labels)

        # Check for entity + negative adjective pattern
        entities = img_features.get('entities', [])
        has_entity = len(entities) > 0

        return has_negative_label or (has_entity and any(word in text_lower for word in self.loaded_negative))

    def _check_obfuscation_vagueness(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for intentional vagueness."""
        # Check for vague references
        vague_refs = ['some people', 'they say', 'sources', 'it is said', 'many believe']
        has_vague_ref = any(ref in text_lower for ref in vague_refs)

        # Check for undefined pronouns
        pronoun_count = len(re.findall(r'\b(it|they|them|those|these)\b', text_lower))
        noun_count = len(re.findall(r'\b[A-Z][a-z]+\b', text))

        has_unclear_pronouns = pronoun_count > noun_count

        return has_vague_ref or has_unclear_pronouns

    def _check_red_herring(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for topic deflection."""
        # Check for topic shift markers
        shift_markers = ['but', 'however', 'what about', 'meanwhile', 'speaking of', 'by the way']
        has_shift = any(marker in text_lower for marker in shift_markers)

        # Check for unrelated comparison
        has_comparison = 'but what about' in text_lower or 'what about' in text_lower

        return has_shift and (has_comparison or text.count('.') >= 2)

    def _check_reductio_ad_hitlerum(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for Hitler/Nazi comparisons."""
        nazi_terms = ['hitler', 'nazi', 'fascist', 'holocaust', 'third reich', 'ss', 'gestapo']

        has_nazi_ref = any(term in text_lower for term in nazi_terms)

        # Check for swastika or Nazi symbols in image
        symbols = img_features.get('symbols', [])
        has_nazi_symbol = 'swastika' in symbols

        return has_nazi_ref or has_nazi_symbol

    def _check_repetition(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for repetition of words/phrases."""
        words = text_lower.split()

        # Count word frequencies
        word_freq = Counter(words)

        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        content_words = {word: count for word, count in word_freq.items()
                        if word not in stopwords and len(word) > 3}

        # Check if any word is repeated 3+ times
        has_repetition = any(count >= 3 for count in content_words.values())

        # Check for phrase repetition (2-3 word sequences)
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        bigram_freq = Counter(bigrams)
        has_phrase_repetition = any(count >= 2 for count in bigram_freq.values())

        return has_repetition or has_phrase_repetition

    def _check_slogans(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for slogan-like phrases."""
        # Check length (slogans are short)
        word_count = len(text.split())
        is_short = word_count <= 10

        # Check for catchy patterns (alliteration, rhyme)
        words = text_lower.split()
        if len(words) >= 2:
            # Simple alliteration check
            first_letters = [w[0] for w in words if len(w) > 0]
            has_alliteration = any(first_letters.count(letter) >= 2 for letter in set(first_letters))
        else:
            has_alliteration = False

        # Check for imperative mood (commands)
        imperative_verbs = ['make', 'stop', 'fight', 'join', 'save', 'protect', 'vote']
        has_imperative = any(text_lower.startswith(verb) for verb in imperative_verbs)

        # Check if it's memorable (short + punchy)
        is_punchy = is_short and (text.count('!') >= 1 or has_imperative)

        return is_short and (has_alliteration or is_punchy)

    def _check_smears(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for personal attacks on credibility."""
        # Check for attack words
        attack_words = ['corrupt', 'liar', 'fraud', 'cheater', 'dishonest', 'criminal']
        has_attack = any(word in text_lower for word in attack_words)

        # Check for entity (person/group being attacked)
        entities = img_features.get('entities', [])
        has_target = len(entities) > 0

        # Check for character attack patterns
        has_character_attack = bool(re.search(r'(always|never) (been|was) (corrupt|dishonest|a liar)', text_lower))

        return (has_attack and has_target) or has_character_attack

    def _check_thought_terminating_cliche(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for thought-terminating cliches."""
        return any(cliche in text_lower for cliche in self.cliche_phrases)

    def _check_transfer(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for symbolic transfer."""
        # This technique requires visual symbolism
        symbols = img_features.get('symbols', [])

        # Check for common transfer symbols
        transfer_symbols = ['flag', 'religious symbol', 'logo', 'leader portrait',
                           'cross', 'star', 'emblem']

        has_symbol = any(symbol in symbols for symbol in transfer_symbols)

        # Check if symbol is associated with emotional text
        has_emotional_text = self._check_appeal_to_strong_emotions(text, text_lower, img_features)

        return has_symbol or (len(symbols) > 0 and has_emotional_text)

    def _check_whataboutism(self, text: str, text_lower: str, img_features: Dict) -> bool:
        """Check for whataboutism."""
        # Check for comparative patterns
        has_comparative = any(re.search(pattern, text_lower)
                            for pattern in self.comparative_patterns)

        # Check for deflection structure (criticism + counter-criticism)
        has_deflection = bool(re.search(r'but (you|they)', text_lower))

        # Check for two topics being compared
        has_two_topics = text.count(',') >= 1 or text.count('but') >= 1

        return has_comparative and (has_deflection or has_two_topics)

    def get_prerequisite_score(self, text: str, image_features: Dict = None) -> Dict[str, float]:
        """
        Get a score (0-1) indicating how strongly prerequisites are satisfied.

        Returns:
            Dictionary mapping technique name to score (0.0 to 1.0)
        """
        results = self.check_all_prerequisites(text, image_features)

        # Convert boolean to float scores
        scores = {}
        for technique, satisfied in results.items():
            # Binary for now, but could be extended to soft scores
            scores[technique] = 1.0 if satisfied else 0.0

        return scores
