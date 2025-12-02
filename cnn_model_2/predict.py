"""
Test Prediction Script

Generate predictions for test set using trained models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

from inference import RealTimePredictor, BatchInference, create_submission
from utils import validate_predictions, load_and_prepare_test_data, ProgressTracker


def predict_test_set(test_metadata_path: str,
                    video_base_path: str,
                    enter_model_path: str,
                    exit_model_path: str,
                    output_path: str):
    """
    Generate predictions for entire test set
    
    Args:
        test_metadata_path: Path to test metadata CSV
        video_base_path: Base directory containing test videos
        enter_model_path: Path to trained entrance model
        exit_model_path: Path to trained exit model
        output_path: Path to save predictions
    """
    print("="*60)
    print("TEST SET PREDICTION")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load test metadata
    print("\nLoading test metadata...")
    test_metadata = load_and_prepare_test_data(test_metadata_path)
    print(f"Total test segments: {len(test_metadata)}")
    
    # Count test periods
    n_periods = test_metadata['test_period_id'].nunique()
    print(f"Number of test periods: {n_periods}")
    
    # Initialize batch inference
    print("\nInitializing predictor...")
    batch_predictor = BatchInference(enter_model_path, exit_model_path)
    
    # Run predictions
    print("\nRunning predictions...")
    predictions = batch_predictor.predict_test_set(
        test_metadata,
        video_base_path,
        output_path
    )
    
    # Validate predictions
    if len(predictions) > 0:
        print("\nValidating predictions...")
        if validate_predictions(predictions):
            print("✓ Predictions are valid")
        else:
            print("✗ Warning: Predictions may have issues")
        
        print(f"\nPrediction summary:")
        print(f"Total predictions: {len(predictions)}")
        print(f"\nEntrance rating distribution:")
        print(predictions['congestion_enter_rating'].value_counts())
        print(f"\nExit rating distribution:")
        print(predictions['congestion_exit_rating'].value_counts())
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description='Generate test set predictions')
    parser.add_argument('--test-metadata', type=str, required=True,
                       help='Path to test metadata CSV')
    parser.add_argument('--video-dir', type=str, required=True,
                       help='Base directory containing test videos')
    parser.add_argument('--enter-model', type=str, required=True,
                       help='Path to trained entrance model')
    parser.add_argument('--exit-model', type=str, required=True,
                       help='Path to trained exit model')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Path to save predictions')
    parser.add_argument('--submission-template', type=str, default=None,
                       help='Path to submission template (optional)')
    
    args = parser.parse_args()
    
    # Run predictions
    predict_test_set(
        args.test_metadata,
        args.video_dir,
        args.enter_model,
        args.exit_model,
        args.output
    )
    
    # Create submission file if template provided
    if args.submission_template:
        print("\nCreating submission file...")
        predictions = pd.read_csv(args.output)
        submission_path = Path(args.output).parent / 'submission.csv'
        
        create_submission(
            predictions,
            args.submission_template,
            str(submission_path)
        )


if __name__ == "__main__":
    # Example usage for testing
    if False:  # Set to True for direct testing
        print("Running in test mode...")
        
        # Create dummy predictions
        predictions = pd.DataFrame({
            'test_period_id': ['period_1'] * 5,
            'minute': [18, 19, 20, 21, 22],
            'congestion_enter_rating': ['free flowing'] * 5,
            'congestion_exit_rating': ['light delay'] * 5
        })
        
        predictions.to_csv('test_predictions.csv', index=False)
        print("Test predictions created")
    else:
        main()