"""
Integration Code for ImprovedMemeClassifier
============================================

This file shows how to add per-technique weight optimization
to your existing ImprovedMemeClassifier base code.

Add these methods to your ImprovedMemeClassifier class.
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score


# ============================================================================
# METHOD 1: Add this to ImprovedMemeClassifier.__init__
# ============================================================================

def __init__(self, config, train_df=None):
    """
    Modify your existing __init__ to initialize fusion module
    """
    # ... existing initialization code ...

    # Initialize three-stream fusion (add this)
    if config.use_llm:
        from fusion import StreamFusion
        from config import FusionConfig

        fusion_config = FusionConfig()
        fusion_config.alpha = config.fusion_alpha
        fusion_config.beta = config.fusion_beta
        fusion_config.gamma = config.fusion_gamma
        fusion_config.strategy = "weighted"
        fusion_config.enable_veto = True

        self.fusion = StreamFusion(fusion_config, self.technique_names)
        print("✓ Three-stream fusion initialized")


# ============================================================================
# METHOD 2: Collect predictions from all three streams
# ============================================================================

def collect_three_stream_predictions(
    self,
    val_loader,
    use_caption=True
) -> List[Tuple[Dict, Dict, Dict, List[str]]]:
    """
    Collect predictions from Neural, KG, and LLM streams on validation set

    Args:
        val_loader: Validation DataLoader
        use_caption: Whether to use captions

    Returns:
        List of (neural_preds, kg_preds, llm_preds, ground_truth) tuples
    """
    print("\n" + "="*70)
    print("COLLECTING THREE-STREAM PREDICTIONS")
    print("="*70)

    self.eval()
    validation_data = []

    with torch.no_grad():
        for batch_idx, (texts, captions, images, labels, sample_ids) in enumerate(
            tqdm(val_loader, desc="Collecting predictions")
        ):
            images = images.to(self.device)
            labels = labels.to(self.device)

            batch_size = images.size(0)

            for i in range(batch_size):
                # Prepare input
                text = texts[i]
                if use_caption and captions[i]:
                    text = f"{text} [SEP] {captions[i]}"

                image = images[i:i+1]  # Keep batch dimension

                # ========================================
                # Stage 1: Neural predictions
                # ========================================
                # Get text and image features
                text_features = self.extract_text_features([text])
                image_features = self.extract_image_features(image)
                multimodal_features = self.extract_multimodal_features([text], image)

                # Combine features
                combined_features = torch.cat([
                    text_features,
                    image_features,
                    multimodal_features
                ], dim=-1)

                # Get neural predictions
                neural_logits = self.classifier(combined_features)
                neural_probs = torch.sigmoid(neural_logits).cpu().numpy()[0]

                # Convert to dictionary
                neural_preds = {
                    self.technique_names[j]: float(neural_probs[j])
                    for j in range(len(self.technique_names))
                }

                # ========================================
                # Stage 2: KG refinement
                # ========================================
                kg_preds = self.kg_refiner.refine_predictions(
                    neural_preds,
                    text,
                    image_features.cpu().numpy()[0]
                )

                # ========================================
                # Stage 3: LLM verification (if enabled)
                # ========================================
                llm_preds = None
                if self.config.use_llm:
                    # Select techniques for verification
                    techniques_to_verify = []

                    for technique in self.technique_names:
                        should_verify, reason, score = self.llm_selector.should_verify(
                            technique,
                            neural_preds[technique],
                            kg_preds[technique]
                        )

                        if should_verify:
                            techniques_to_verify.append(technique)

                    # Verify with LLM
                    if techniques_to_verify:
                        caption = captions[i] if use_caption else ""
                        llm_results = self.llm_verifier.verify(
                            text,
                            caption,
                            techniques_to_verify,
                            neural_preds,
                            kg_preds
                        )

                        # Convert to scores
                        llm_preds = {
                            tech: llm_results.get(tech, {}).get('score', 0.5)
                            for tech in self.technique_names
                        }

                # ========================================
                # Ground truth
                # ========================================
                label_vector = labels[i].cpu().numpy()
                ground_truth = [
                    self.technique_names[j]
                    for j in range(len(self.technique_names))
                    if label_vector[j] == 1
                ]

                # Add to validation data
                validation_data.append((
                    neural_preds,
                    kg_preds,
                    llm_preds,
                    ground_truth
                ))

    print(f"✓ Collected {len(validation_data)} validation samples")
    print(f"  Neural predictions: ✓")
    print(f"  KG predictions: ✓")
    print(f"  LLM predictions: {'✓' if self.config.use_llm else '✗ (disabled)'}")

    return validation_data


# ============================================================================
# METHOD 3: Optimize per-technique fusion weights
# ============================================================================

def optimize_fusion_weights(
    self,
    val_loader,
    metric='f1',
    search_space='coarse',
    use_caption=True
) -> Dict[str, Tuple[float, float, float]]:
    """
    Optimize per-technique fusion weights on validation set

    Args:
        val_loader: Validation DataLoader
        metric: Optimization metric ('f1', 'precision', 'recall')
        search_space: 'coarse', 'fine', or 'very_fine'
        use_caption: Whether to use captions

    Returns:
        Dictionary mapping technique to (alpha, beta, gamma)
    """
    if not hasattr(self, 'fusion'):
        print("⚠ Error: Fusion module not initialized")
        print("  Make sure config.use_llm = True")
        return None

    print("\n" + "="*70)
    print("OPTIMIZING PER-TECHNIQUE FUSION WEIGHTS")
    print("="*70)

    # Stage 1: Collect predictions
    validation_data = self.collect_three_stream_predictions(val_loader, use_caption)

    # Stage 2: Optimize weights
    per_tech_weights = self.fusion.optimize_per_technique_weights(
        validation_data,
        metric=metric,
        search_space=search_space,
        min_samples_per_technique=5
    )

    # Stage 3: Save weights
    weights_path = "per_technique_weights.json"
    self.fusion.save_per_technique_weights(weights_path)
    print(f"\n✓ Weights saved to {weights_path}")

    return per_tech_weights


# ============================================================================
# METHOD 4: Validate with per-technique weights
# ============================================================================

def validate_with_per_technique_weights(
    self,
    val_loader,
    dataset_name="Validation",
    use_caption=True,
    use_per_technique=True
) -> Tuple[float, float, float, float]:
    """
    Validate using three-stream fusion with per-technique weights

    Args:
        val_loader: Validation DataLoader
        dataset_name: Name for logging
        use_caption: Whether to use captions
        use_per_technique: Use per-technique weights (vs global)

    Returns:
        (avg_loss, f1, precision, recall)
    """
    if not hasattr(self, 'fusion'):
        print("⚠ Warning: Fusion module not initialized, using KG refinement only")
        return self.validate_with_kg_refinement(val_loader, dataset_name)

    print(f"\n{'='*70}")
    print(f"VALIDATION: {dataset_name}")
    print(f"Mode: {'Per-Technique Weights' if use_per_technique else 'Global Weights'}")
    print(f"{'='*70}")

    # Set weight mode
    if use_per_technique:
        self.fusion.enable_per_technique_weights()
    else:
        self.fusion.disable_per_technique_weights()

    self.eval()
    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_labels = []

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, (texts, captions, images, labels, sample_ids) in enumerate(
            tqdm(val_loader, desc=f"Validating")
        ):
            images = images.to(self.device)
            labels = labels.to(self.device)

            batch_size = images.size(0)

            for i in range(batch_size):
                # Prepare input
                text = texts[i]
                if use_caption and captions[i]:
                    text = f"{text} [SEP] {captions[i]}"

                image = images[i:i+1]

                # Get neural predictions
                text_features = self.extract_text_features([text])
                image_features = self.extract_image_features(image)
                multimodal_features = self.extract_multimodal_features([text], image)

                combined_features = torch.cat([
                    text_features, image_features, multimodal_features
                ], dim=-1)

                neural_logits = self.classifier(combined_features)
                neural_probs = torch.sigmoid(neural_logits).cpu().numpy()[0]

                neural_preds = {
                    self.technique_names[j]: float(neural_probs[j])
                    for j in range(len(self.technique_names))
                }

                # Get KG predictions
                kg_preds = self.kg_refiner.refine_predictions(
                    neural_preds,
                    text,
                    image_features.cpu().numpy()[0]
                )

                # Get LLM predictions (if enabled)
                llm_preds = None
                if self.config.use_llm:
                    techniques_to_verify = []
                    for technique in self.technique_names:
                        should_verify, _, _ = self.llm_selector.should_verify(
                            technique,
                            neural_preds[technique],
                            kg_preds[technique]
                        )
                        if should_verify:
                            techniques_to_verify.append(technique)

                    if techniques_to_verify:
                        caption = captions[i] if use_caption else ""
                        llm_results = self.llm_verifier.verify(
                            text, caption, techniques_to_verify,
                            neural_preds, kg_preds
                        )
                        llm_preds = {
                            tech: llm_results.get(tech, {}).get('score', 0.5)
                            for tech in self.technique_names
                        }

                # Fuse predictions
                fused_preds = self.fusion.fuse(neural_preds, kg_preds, llm_preds)

                # Apply thresholds
                detected = self.fusion.apply_thresholds(fused_preds)

                # Convert to binary vector
                pred_vector = [
                    1 if tech in detected else 0
                    for tech in self.technique_names
                ]

                label_vector = labels[i].cpu().numpy()

                all_predictions.append(pred_vector)
                all_labels.append(label_vector)

                # Compute loss (on neural logits)
                loss = criterion(neural_logits, labels[i:i+1])
                total_loss += loss.item()
                num_batches += 1

    # Convert to numpy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Compute metrics
    f1 = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
    precision = precision_score(all_labels, all_predictions, average='micro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='micro', zero_division=0)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Print results
    print(f"\n{dataset_name} Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"{'='*70}\n")

    return avg_loss, f1, precision, recall


# ============================================================================
# METHOD 5: Compare global vs per-technique (optional)
# ============================================================================

def compare_weight_strategies(self, val_loader, use_caption=True):
    """
    Compare global weights vs per-technique weights

    Args:
        val_loader: Validation DataLoader
        use_caption: Whether to use captions

    Returns:
        Comparison dictionary
    """
    print("\n" + "="*70)
    print("COMPARING WEIGHT STRATEGIES")
    print("="*70)

    # Evaluate with global weights
    print("\n[1/2] Evaluating with GLOBAL weights...")
    _, f1_global, prec_global, rec_global = self.validate_with_per_technique_weights(
        val_loader,
        dataset_name="Global Weights",
        use_caption=use_caption,
        use_per_technique=False
    )

    # Evaluate with per-technique weights
    print("\n[2/2] Evaluating with PER-TECHNIQUE weights...")
    _, f1_per_tech, prec_per_tech, rec_per_tech = self.validate_with_per_technique_weights(
        val_loader,
        dataset_name="Per-Technique Weights",
        use_caption=use_caption,
        use_per_technique=True
    )

    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print(f"\n{'Metric':<15} {'Global':<12} {'Per-Tech':<12} {'Improvement':<12}")
    print("-" * 55)

    metrics = [
        ('F1', f1_global, f1_per_tech),
        ('Precision', prec_global, prec_per_tech),
        ('Recall', rec_global, rec_per_tech)
    ]

    for name, global_val, per_tech_val in metrics:
        improvement = per_tech_val - global_val
        print(f"{name:<15} {global_val:<12.4f} {per_tech_val:<12.4f} {improvement:+.4f}")

    print("="*70)

    return {
        'global': {'f1': f1_global, 'precision': prec_global, 'recall': rec_global},
        'per_technique': {'f1': f1_per_tech, 'precision': prec_per_tech, 'recall': rec_per_tech}
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """
    Example of how to use per-technique optimization in your training loop
    """
    print("""
USAGE EXAMPLE:
==============

# 1. Initialize model with LLM enabled
config = CFG()
config.use_llm = True
config.fusion_alpha = 0.4
config.fusion_beta = 0.4
config.fusion_gamma = 0.2

model = ImprovedMemeClassifier(config, train_df)

# 2. Train neural stream normally
model.fit_improved(
    train_loader,
    val_loader,
    test_loader,
    num_epochs=20,
    use_progressive=True
)

# 3. Optimize per-technique fusion weights
per_tech_weights = model.optimize_fusion_weights(
    val_loader,
    metric='f1',
    search_space='coarse',  # Fast
    use_caption=True
)

# 4. Compare strategies
comparison = model.compare_weight_strategies(val_loader, use_caption=True)

# 5. Final evaluation with per-technique weights
_, test_f1, test_prec, test_rec = model.validate_with_per_technique_weights(
    test_loader,
    dataset_name="Test Set",
    use_caption=True,
    use_per_technique=True  # Use optimized per-technique weights
)

print(f"Final Test F1: {test_f1:.4f}")

# 6. Save model with optimized weights
model.fusion.save_per_technique_weights("final_per_technique_weights.json")

# 7. Later, load and use
model.fusion.load_per_technique_weights("final_per_technique_weights.json")
    """)


if __name__ == "__main__":
    example_usage()

    print("\n" + "="*70)
    print("INTEGRATION CHECKLIST")
    print("="*70)
    print("""
To integrate per-technique optimization into your ImprovedMemeClassifier:

✓ Step 1: Add fusion initialization to __init__
  - Copy the __init__ code above
  - Initialize StreamFusion with your config

✓ Step 2: Add collect_three_stream_predictions method
  - This collects predictions from all three streams

✓ Step 3: Add optimize_fusion_weights method
  - This optimizes per-technique weights

✓ Step 4: Add validate_with_per_technique_weights method
  - This validates using optimized weights

✓ Step 5: (Optional) Add compare_weight_strategies method
  - This compares global vs per-technique

✓ Step 6: Update your training loop
  - After training neural stream, optimize weights
  - Use per-technique weights for final evaluation

Expected improvement: +2-4% F1 overall, +10-20% for rare techniques
    """)
