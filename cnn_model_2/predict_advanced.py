"""
Advanced Prediction Script - YOLO + CNN Features from GCS

Generate predictions using advanced features with automatic batch processing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

from gcs_batch_processor import GCSBatchProcessor
from model_training import CongestionPredictor
from utils import validate_predictions


class AdvancedPredictor:
    """Advanced predictor with batch processing from GCS"""
    
    def __init__(self,
                 enter_model_path: str,
                 exit_model_path: str,
                 bucket_name: str = 'brb-traffic',
                 batch_size: int = 10,
                 use_yolo: bool = True,
                 use_cnn: bool = True):
        """
        Initialize advanced predictor
        
        Args:
            enter_model_path: Path to entrance model
            exit_model_path: Path to exit model
            bucket_name: GCS bucket name
            batch_size: Batch size for processing
            use_yolo: Use YOLO detection
            use_cnn: Use CNN features
        """
        # Load models
        print("Loading models...")
        self.enter_predictor = CongestionPredictor()
        self.enter_predictor.load(enter_model_path)
        
        self.exit_predictor = CongestionPredictor()
        self.exit_predictor.load(exit_model_path)
        
        print("✓ Models loaded")
        
        # Initialize GCS processor
        print("\nInitializing GCS batch processor...")
        self.gcs_processor = GCSBatchProcessor(
            bucket_name=bucket_name,
            batch_size=batch_size,
            use_yolo=use_yolo,
            use_cnn=use_cnn
        )
        print("✓ Processor initialized")
    
    def predict_test_set(self,
                        test_metadata_path: str,
                        output_path: str) -> pd.DataFrame:
        """
        Generate predictions for test set
        
        Args:
            test_metadata_path: Path to test metadata CSV
            output_path: Path to save predictions
            
        Returns:
            DataFrame with predictions
        """
        print("="*80)
        print("ADVANCED TEST SET PREDICTION")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load test metadata
        print("\nLoading test metadata...")
        test_metadata = pd.read_csv(test_metadata_path)
        print(f"Total test segments: {len(test_metadata)}")
        
        # Group by test period (15-min input → 5-min prediction)
        if 'test_period_id' in test_metadata.columns:
            test_periods = test_metadata.groupby('test_period_id')
            n_periods = test_metadata['test_period_id'].nunique()
        else:
            # If no test_period_id, process all segments
            test_periods = [(None, test_metadata)]
            n_periods = 1
        
        print(f"Number of test periods: {n_periods}")
        
        # Extract features from all test videos
        print("\n" + "="*80)
        print("EXTRACTING FEATURES FROM TEST VIDEOS")
        print("="*80)
        
        # Get all unique video files
        all_videos = []
        for camera in range(1, 5):
            col_name = f'cam{camera}_filename'
            if col_name in test_metadata.columns:
                all_videos.extend(test_metadata[col_name].unique().tolist())
        
        all_videos = list(set(all_videos))
        print(f"Total unique videos: {len(all_videos)}")
        
        # Process videos in batches and extract features
        video_features = {}
        
        for i in range(0, len(all_videos), self.gcs_processor.batch_size):
            batch = all_videos[i:i + self.gcs_processor.batch_size]
            batch_num = i // self.gcs_processor.batch_size + 1
            total_batches = len(all_videos) // self.gcs_processor.batch_size + 1
            
            print(f"\nBatch {batch_num}/{total_batches}")
            
            # Download
            print("  Downloading...")
            local_paths = self.gcs_processor.download_batch(batch, show_progress=True)
            
            if not local_paths:
                print("  No files downloaded, skipping batch")
                continue
            
            # Process
            print("  Processing...")
            batch_df = self.gcs_processor.process_batch(local_paths)
            
            # Store features by video name
            for _, row in batch_df.iterrows():
                video_features[row['video_name']] = row.to_dict()
            
            # Cleanup
            print("  Cleaning up...")
            self.gcs_processor.cleanup_batch(local_paths)
        
        print(f"\n✓ Extracted features from {len(video_features)} videos")
        
        # Generate predictions for each test period
        print("\n" + "="*80)
        print("GENERATING PREDICTIONS")
        print("="*80)
        
        all_predictions = []
        
        for period_id, period_data in test_periods:
            print(f"\nProcessing test period: {period_id if period_id else 'all'}")
            
            # Get input window (minutes 1-15)
            if 'minute' in period_data.columns:
                input_data = period_data[period_data['minute'] <= 15].sort_values('minute')
                
                if len(input_data) != 15:
                    print(f"  Warning: Expected 15 input minutes, got {len(input_data)}")
            else:
                input_data = period_data
            
            # Combine features from all cameras for each segment
            segment_features = []
            
            for _, row in input_data.iterrows():
                seg_features = {}
                
                # Add features from each camera
                for camera in range(1, 5):
                    col_name = f'cam{camera}_filename'
                    if col_name in row:
                        video_name = row[col_name]
                        
                        if video_name in video_features:
                            cam_feats = video_features[video_name]
                            # Add with camera prefix
                            for key, value in cam_feats.items():
                                if key not in ['video_path', 'video_name']:
                                    seg_features[f'cam{camera}_{key}'] = value
                
                if seg_features:
                    segment_features.append(seg_features)
            
            if not segment_features:
                print(f"  No features available for period {period_id}")
                continue
            
            # Use last segment (minute 15) for prediction
            last_segment_features = pd.DataFrame([segment_features[-1]])
            
            # Make predictions for minutes 18-22
            for minute in range(18, 23):
                # Prepare features
                X_enter, _ = self.enter_predictor.prepare_features(
                    last_segment_features,
                    target_col=None,
                    fit_scaler=False
                )
                
                X_exit, _ = self.exit_predictor.prepare_features(
                    last_segment_features,
                    target_col=None,
                    fit_scaler=False
                )
                
                # Predict
                enter_pred = self.enter_predictor.predict(X_enter)[0]
                exit_pred = self.exit_predictor.predict(X_exit)[0]
                
                # Get confidence
                enter_proba = self.enter_predictor.predict_proba(X_enter)[0]
                exit_proba = self.exit_predictor.predict_proba(X_exit)[0]
                
                prediction = {
                    'test_period_id': period_id if period_id else 'test',
                    'minute': minute,
                    'congestion_enter_rating': enter_pred,
                    'congestion_exit_rating': exit_pred,
                    'enter_confidence': max(enter_proba),
                    'exit_confidence': max(exit_proba)
                }
                
                all_predictions.append(prediction)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame(all_predictions)
        
        # Validate
        if len(predictions_df) > 0:
            print("\n" + "="*80)
            print("VALIDATION")
            print("="*80)
            
            if validate_predictions(predictions_df):
                print("✓ Predictions are valid")
            else:
                print("✗ Warning: Predictions may have issues")
            
            # Summary
            print(f"\nPrediction Summary:")
            print(f"  Total predictions: {len(predictions_df)}")
            print(f"\n  Entrance rating distribution:")
            for rating, count in predictions_df['congestion_enter_rating'].value_counts().items():
                print(f"    {rating}: {count}")
            print(f"\n  Exit rating distribution:")
            for rating, count in predictions_df['congestion_exit_rating'].value_counts().items():
                print(f"    {rating}: {count}")
            
            # Save
            predictions_df.to_csv(output_path, index=False)
            print(f"\n✓ Predictions saved to {output_path}")
        
        # Cleanup
        print("\n" + "="*80)
        print("CLEANUP")
        print("="*80)
        self.gcs_processor.cleanup_temp_dir()
        
        print("\n" + "="*80)
        print("PREDICTION COMPLETE")
        print("="*80)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return predictions_df


def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions using advanced features (YOLO + CNN)'
    )
    parser.add_argument('--test-metadata', type=str, required=True,
                       help='Path to test metadata CSV')
    parser.add_argument('--enter-model', type=str, required=True,
                       help='Path to trained entrance model')
    parser.add_argument('--exit-model', type=str, required=True,
                       help='Path to trained exit model')
    parser.add_argument('--output', type=str, default='predictions_advanced.csv',
                       help='Path to save predictions')
    parser.add_argument('--bucket', type=str, default='brb-traffic',
                       choices=['brb-traffic', 'brb-traffic-full'],
                       help='GCS bucket name')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for processing')
    parser.add_argument('--no-yolo', action='store_true',
                       help='Disable YOLO detection')
    parser.add_argument('--no-cnn', action='store_true',
                       help='Disable CNN features')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = AdvancedPredictor(
        enter_model_path=args.enter_model,
        exit_model_path=args.exit_model,
        bucket_name=args.bucket,
        batch_size=args.batch_size,
        use_yolo=not args.no_yolo,
        use_cnn=not args.no_cnn
    )
    
    # Generate predictions
    predictor.predict_test_set(
        test_metadata_path=args.test_metadata,
        output_path=args.output
    )


if __name__ == "__main__":
    main()