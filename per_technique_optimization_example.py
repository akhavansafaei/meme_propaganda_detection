"""
Per-Technique Weight Optimization Example
==========================================

This script demonstrates how to optimize fusion weights per technique
instead of using global weights.

Expected improvement: +2-4% F1 overall, +10-20% for rare techniques
"""

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from config import get_config_colab_no_llm, ExperimentConfig
from pipeline import ThreeStreamPropagandaDetector
from data_loader import create_label_mappings, create_dataloaders
from trainer import NeuralStreamTrainer


def collect_validation_predictions(detector, val_loader):
    """
    Collect predictions from all three streams on validation set

    Args:
        detector: ThreeStreamPropagandaDetector instance
        val_loader: Validation data loader

    Returns:
        List of (neural_preds, kg_preds, llm_preds, ground_truth) tuples
    """
    print("\n" + "="*70)
    print("COLLECTING VALIDATION PREDICTIONS")
    print("="*70)

    validation_data = []

    for texts, images, labels, sample_ids in tqdm(val_loader, desc="Processing batches"):
        # Process each sample in batch
        batch_size = len(texts)

        for i in range(batch_size):
            text = texts[i]
            image = images[i]
            label_vector = labels[i].cpu().numpy()

            # Convert label vector to list of technique names
            ground_truth = [
                detector.technique_names[j]
                for j, val in enumerate(label_vector)
                if val == 1
            ]

            # Get neural predictions
            neural_preds = detector.neural_predictor.predict(text, image)

            # Get KG predictions
            kg_result = detector.kg_predictor.predict(
                text=text,
                image=image,
                neural_predictions=neural_preds,
                return_full_output=True
            )
            kg_preds = kg_result['refined_predictions']

            # Get LLM predictions (if enabled)
            llm_preds = None
            if detector.config.llm.use_llm:
                image_desc = "Image with text overlay"  # Placeholder
                llm_preds = detector.llm_stream.predict(
                    text=text,
                    image_description=image_desc,
                    neural_predictions=neural_preds,
                    kg_predictions=kg_preds
                )

            # Add to validation data
            validation_data.append((
                neural_preds,
                kg_preds,
                llm_preds,
                ground_truth
            ))

    print(f"✓ Collected {len(validation_data)} validation samples")
    return validation_data


def evaluate_with_weights(detector, val_loader, use_per_technique=False):
    """
    Evaluate detector on validation set

    Args:
        detector: ThreeStreamPropagandaDetector instance
        val_loader: Validation data loader
        use_per_technique: Whether to use per-technique weights

    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    # Set weight mode
    if use_per_technique:
        detector.fusion.enable_per_technique_weights()
    else:
        detector.fusion.disable_per_technique_weights()

    all_predictions = []
    all_labels = []

    for texts, images, labels, _ in tqdm(val_loader, desc="Evaluating"):
        batch_size = len(texts)

        for i in range(batch_size):
            # Get prediction
            result = detector.predict(texts[i], images[i])

            # Convert to binary vector
            pred_vector = [
                1 if tech in result.detected_techniques else 0
                for tech in detector.technique_names
            ]

            label_vector = labels[i].cpu().numpy()

            all_predictions.append(pred_vector)
            all_labels.append(label_vector)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Compute metrics
    f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_predictions, average='micro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='micro', zero_division=0)

    # Per-technique F1
    per_technique_f1 = {}
    for idx, technique in enumerate(detector.technique_names):
        tech_labels = all_labels[:, idx]
        tech_preds = all_predictions[:, idx]
        tech_f1 = f1_score(tech_labels, tech_preds, average='binary', zero_division=0)
        per_technique_f1[technique] = tech_f1

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
        'per_technique_f1': per_technique_f1
    }


def compare_global_vs_per_technique(detector, val_loader, validation_data):
    """
    Compare performance with global vs per-technique weights

    Args:
        detector: ThreeStreamPropagandaDetector instance
        val_loader: Validation data loader
        validation_data: Collected validation predictions

    Returns:
        Comparison results
    """
    print("\n" + "="*70)
    print("COMPARING GLOBAL VS PER-TECHNIQUE WEIGHTS")
    print("="*70)

    # Evaluate with global weights
    print("\n[1/2] Evaluating with GLOBAL weights...")
    metrics_global = evaluate_with_weights(detector, val_loader, use_per_technique=False)

    # Optimize per-technique weights
    print("\n[2/2] Optimizing and evaluating with PER-TECHNIQUE weights...")
    detector.fusion.optimize_per_technique_weights(
        validation_data,
        metric='f1',
        search_space='coarse'
    )

    # Evaluate with per-technique weights
    metrics_per_tech = evaluate_with_weights(detector, val_loader, use_per_technique=True)

    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print(f"\nOverall Metrics:")
    print(f"{'Metric':<20} {'Global':<12} {'Per-Tech':<12} {'Improvement':<12}")
    print("-" * 60)

    for metric_name in ['f1_micro', 'f1_macro', 'precision', 'recall']:
        global_val = metrics_global[metric_name]
        per_tech_val = metrics_per_tech[metric_name]
        improvement = per_tech_val - global_val

        print(f"{metric_name:<20} {global_val:<12.4f} {per_tech_val:<12.4f} {improvement:+.4f}")

    # Per-technique comparison
    print(f"\n\nPer-Technique F1 Comparison:")
    print(f"{'Technique':<40} {'Global':<10} {'Per-Tech':<10} {'Change':<10}")
    print("-" * 70)

    improvements = []
    for technique in detector.technique_names:
        global_f1 = metrics_global['per_technique_f1'][technique]
        per_tech_f1 = metrics_per_tech['per_technique_f1'][technique]
        change = per_tech_f1 - global_f1

        improvements.append((technique, change))

        print(f"{technique:<40} {global_f1:<10.4f} {per_tech_f1:<10.4f} {change:+.4f}")

    # Show top improvements
    print("\n\nTop 5 Improved Techniques:")
    print("-" * 50)
    improvements.sort(key=lambda x: x[1], reverse=True)
    for technique, improvement in improvements[:5]:
        print(f"  {technique:<35} {improvement:+.4f}")

    print("\n" + "="*70)

    return {
        'global': metrics_global,
        'per_technique': metrics_per_tech
    }


def main():
    """
    Main workflow for per-technique weight optimization
    """
    print("\n" + "="*70)
    print("PER-TECHNIQUE WEIGHT OPTIMIZATION")
    print("="*70)

    # ========================================
    # Step 1: Setup
    # ========================================
    print("\n[Step 1/6] Setup")

    # Create config
    config = get_config_colab_no_llm()

    # Update paths (modify these for your setup)
    config.data.train_json = "dataset_with_rationales_subtask2a_final (1).json"
    config.data.val_json = "validation_caption.json"
    config.data.train_img_dir = "/content/train_images/train_images"
    config.data.val_img_dir = "/content/validation_images/validation_images"

    # Training settings
    config.neural.batch_size = 32
    config.neural.epochs = 12

    print("✓ Configuration ready")

    # ========================================
    # Step 2: Load Data
    # ========================================
    print("\n[Step 2/6] Load Data")

    label_to_idx, idx_to_label, num_labels = create_label_mappings(
        config.kg.techniques_json
    )

    train_loader, val_loader, _, _, _, _ = create_dataloaders(
        config.data,
        label_to_idx,
        batch_size=config.neural.batch_size,
        use_caption=config.neural.use_caption,
        num_workers=0
    )

    print(f"✓ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # ========================================
    # Step 3: Initialize Detector
    # ========================================
    print("\n[Step 3/6] Initialize Detector")

    detector = ThreeStreamPropagandaDetector(config)

    # ========================================
    # Step 4: Train Neural Stream (or load checkpoint)
    # ========================================
    print("\n[Step 4/6] Train Neural Stream")

    # Option A: Train from scratch
    # trainer = NeuralStreamTrainer(detector.neural_stream, config.neural, idx_to_label)
    # trainer.train(train_loader, val_loader, num_epochs=config.neural.epochs)

    # Option B: Load pretrained checkpoint
    # detector.load_neural_checkpoint("best_model.pth")

    # For this example, we'll skip training
    print("⚠ Skipping training (use pretrained or train separately)")

    # ========================================
    # Step 5: Collect Validation Predictions
    # ========================================
    print("\n[Step 5/6] Collect Validation Predictions")

    validation_data = collect_validation_predictions(detector, val_loader)

    # ========================================
    # Step 6: Optimize and Compare
    # ========================================
    print("\n[Step 6/6] Optimize Per-Technique Weights and Compare")

    results = compare_global_vs_per_technique(detector, val_loader, validation_data)

    # ========================================
    # Save Results
    # ========================================
    print("\n[Saving] Saving optimized weights...")

    # Save per-technique weights
    weights_path = Path(config.data.checkpoint_dir) / "per_technique_weights.json"
    detector.fusion.save_per_technique_weights(str(weights_path))

    # Save comparison results
    import json
    results_path = Path(config.data.results_dir) / "global_vs_per_technique_comparison.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump({
            'global_f1_micro': results['global']['f1_micro'],
            'per_technique_f1_micro': results['per_technique']['f1_micro'],
            'improvement': results['per_technique']['f1_micro'] - results['global']['f1_micro']
        }, f, indent=2)

    print(f"✓ Weights saved to {weights_path}")
    print(f"✓ Results saved to {results_path}")

    print("\n" + "="*70)
    print("✓ PER-TECHNIQUE OPTIMIZATION COMPLETE!")
    print("="*70)

    # Print final summary
    improvement = results['per_technique']['f1_micro'] - results['global']['f1_micro']
    print(f"\nFinal F1 Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")

    if improvement > 0:
        print("✓ Per-technique weights improve performance!")
    else:
        print("⚠ Per-technique weights did not improve performance")
        print("  This can happen if validation set is too small")


if __name__ == "__main__":
    """
    Run this script to:
    1. Train neural stream (or load pretrained)
    2. Collect predictions from all three streams
    3. Optimize per-technique fusion weights
    4. Compare global vs per-technique performance
    5. Save optimized weights
    """

    # Check if running in Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Running in Google Colab")

        # Navigate to project directory
        import os
        import sys
        project_dir = "/content/drive/MyDrive/propaganda_detection"
        os.chdir(project_dir)
        sys.path.insert(0, project_dir)
    except:
        print("Running locally")

    # Run main workflow
    main()

    print("\n" + "="*70)
    print("USAGE INSTRUCTIONS")
    print("="*70)
    print("""
Next steps:

1. Use optimized weights for prediction:
   ```python
   detector.fusion.load_per_technique_weights("per_technique_weights.json")
   result = detector.predict(text, image)
   ```

2. Evaluate on test set:
   ```python
   test_results = detector.predict_batch(test_texts, test_images)
   ```

3. Toggle between global and per-technique:
   ```python
   # Per-technique weights
   detector.fusion.enable_per_technique_weights()

   # Global weights
   detector.fusion.disable_per_technique_weights()
   ```

4. Analyze which techniques improved:
   - Check the "Top 5 Improved Techniques" in the output above
   - Visual techniques should show higher neural weight (α)
   - Rule-based techniques should show higher KG weight (β)
   - Complex techniques should show higher LLM weight (γ)
    """)
