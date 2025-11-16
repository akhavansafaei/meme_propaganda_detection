"""
Configuration Module for Three-Stream Propaganda Detection
Manages all hyperparameters and paths for Colab + Google Drive setup
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class DataConfig:
    """Data loading configuration"""
    # Google Drive paths - UPDATE THESE
    drive_root: str = "/content/drive/MyDrive"
    project_dir: str = "propaganda_detection"

    # Data paths
    train_json: str = "dataset_with_rationales_subtask2a_final (1).json"
    val_json: str = "validation_caption.json"
    test_json: str = "dev_processed.json"

    # Image directories
    train_img_dir: str = "/content/train_images/train_images"
    val_img_dir: str = "/content/validation_images/validation_images"
    test_img_dir: str = "/content/dev_images/dev_images"

    # Outputs
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    logs_dir: str = "logs"

    def get_full_path(self, relative_path: str) -> str:
        """Get full path relative to Drive root"""
        return os.path.join(self.drive_root, self.project_dir, relative_path)


@dataclass
class NeuralConfig:
    """Neural model (Stream 1) configuration"""
    # Model names
    clip_model_name: str = "openai/clip-vit-base-patch32"
    roberta_model_name: str = "roberta-base"

    # Training
    batch_size: int = 32
    learning_rate: float = 2e-3
    epochs: int = 12

    # Device
    device: str = "cuda"  # Will auto-detect

    # Features
    use_caption: bool = True
    caption_separator: str = " [SEP] "

    # Outputs
    output_probabilities: bool = True  # Return soft predictions (not binary)


@dataclass
class KnowledgeGraphConfig:
    """Knowledge Graph (Stream 2) configuration"""
    # Paths
    techniques_json: str = "propaganda_techniques.json"

    # Constraint satisfaction
    max_iterations: int = 5
    convergence_threshold: float = 0.01

    # Weights for different reasoning types
    prerequisite_weight: float = 1.0  # Hard constraints
    co_occurrence_weight: float = 0.3  # Boost strength
    mutual_exclusion_weight: float = 0.5  # Penalty strength
    hierarchical_weight: float = 0.4  # Hierarchical consistency

    # Co-occurrence learning
    min_co_occurrence_support: int = 5  # Minimum samples to trust co-occurrence
    co_occurrence_threshold: float = 0.7  # Minimum PMI to consider co-occurring

    # Thresholds
    prerequisite_threshold: float = 0.5  # Minimum score to pass prerequisite
    detection_threshold: float = 0.35  # Default detection threshold


@dataclass
class LLMConfig:
    """LLM Reasoner (Stream 3) configuration"""
    # LLM settings
    use_llm: bool = False  # Enable/disable LLM stream
    provider: str = "openai"  # "openai" or "anthropic"
    model_name: str = "gpt-4"  # or "claude-3-opus-20240229"
    api_key: Optional[str] = None  # Set this!

    # Verification settings
    max_techniques_to_verify: int = 10  # Limit LLM calls
    temperature: float = 0.0  # Deterministic
    max_tokens: int = 2048

    # Selection criteria for LLM verification
    disagreement_threshold: float = 0.3  # Neural vs KG disagreement
    medium_confidence_range: tuple = (0.3, 0.7)  # Uncertainty range
    rare_techniques: List[str] = field(default_factory=lambda: [
        'Whataboutism', 'Red Herring', 'Bandwagon', 'Repetition',
        'Misrepresentation (Straw Man)', 'Reductio ad Hitlerum',
        'Thought-Terminating Cliche', 'Obfuscation/Vagueness'
    ])


@dataclass
class FusionConfig:
    """Stream fusion configuration"""
    # Fusion strategy: "weighted", "voting", "adaptive"
    strategy: str = "weighted"

    # Weights for weighted fusion
    alpha: float = 0.4  # Neural stream weight
    beta: float = 0.4   # Knowledge graph weight
    gamma: float = 0.2  # LLM weight (if enabled)

    # For adaptive fusion
    learn_weights: bool = True  # Learn from validation data
    weight_optimization_metric: str = "hierarchical_f1"

    # Veto logic
    enable_veto: bool = True
    kg_veto_threshold: float = 0.0  # If KG gives 0, heavily penalize
    llm_veto_threshold: float = 0.2  # If LLM strongly rejects
    veto_penalty: float = 0.1  # Multiply prediction by this


@dataclass
class ExperimentConfig:
    """Overall experiment configuration"""
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    kg: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # General settings
    seed: int = 42
    verbose: bool = True

    # Experiment tracking
    experiment_name: str = "three_stream_baseline"
    save_predictions: bool = True
    save_explanations: bool = True

    def __post_init__(self):
        """Auto-detect device and validate paths"""
        import torch
        if self.neural.device == "cuda" and not torch.cuda.is_available():
            self.neural.device = "cpu"
            if self.verbose:
                print("⚠ CUDA not available, using CPU")

        # Create output directories
        for dir_name in [self.data.checkpoint_dir, self.data.results_dir, self.data.logs_dir]:
            os.makedirs(dir_name, exist_ok=True)

    def to_dict(self) -> Dict:
        """Convert config to dictionary for saving"""
        from dataclasses import asdict
        return asdict(self)

    def save(self, path: str):
        """Save configuration to JSON"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        # TODO: Proper deserialization with nested dataclasses
        return cls(**config_dict)


# Pre-defined configurations for different scenarios

def get_config_colab_full() -> ExperimentConfig:
    """Full three-stream configuration for Colab"""
    config = ExperimentConfig()
    config.experiment_name = "three_stream_full"
    config.llm.use_llm = True  # Enable LLM
    return config


def get_config_colab_no_llm() -> ExperimentConfig:
    """Neural + KG only (no LLM) for Colab"""
    config = ExperimentConfig()
    config.experiment_name = "neural_kg_only"
    config.llm.use_llm = False
    config.fusion.alpha = 0.5
    config.fusion.beta = 0.5
    config.fusion.gamma = 0.0
    return config


def get_config_quick_test() -> ExperimentConfig:
    """Quick test configuration (small data, fast)"""
    config = ExperimentConfig()
    config.experiment_name = "quick_test"
    config.neural.batch_size = 8
    config.neural.epochs = 2
    config.llm.use_llm = False
    config.kg.max_iterations = 2
    return config


if __name__ == "__main__":
    # Test configuration
    config = get_config_colab_no_llm()
    print("Configuration loaded successfully:")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Device: {config.neural.device}")
    print(f"  Use LLM: {config.llm.use_llm}")
    print(f"  Fusion weights: α={config.fusion.alpha}, β={config.fusion.beta}, γ={config.fusion.gamma}")
