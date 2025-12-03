"""
ANN Model Training Script for Traffic Congestion Prediction
Trains feedforward neural network from extracted video features.
"""
import pandas as pd
from pathlib import Path
from ann_training import train_ann_pipeline

def main():
    # load features
    features_df = pd.read_csv('extracted_features.csv')
    
    print(f"\nLoaded {len(features_df)} samples with {len(features_df.columns)} columns")
    print(f"\nTarget distribution (entrance):")
    print(features_df['congestion_enter_rating'].value_counts())
    print(f"\nTarget distribution (exit):")
    print(features_df['congestion_exit_rating'].value_counts())
    
    output_path = Path('models')
    output_path.mkdir(parents=True, exist_ok=True)
    
    features_enter = features_df.copy()
    features_exit = features_df.copy()
    
    # since just 10-video features for now, change later
    hidden_dims = [32, 16]
    epochs = 100
    
    print(f"\nHidden layers: {hidden_dims}")
    print(f"Max epochs: {epochs}")
    
    enter_model_ann = train_ann_pipeline(
        features_enter,
        target_col='congestion_enter_rating',
        model_save_path=str(output_path / 'entrance_model_ann.pkl'),
        hidden_dims=hidden_dims,
        epochs=epochs
    )
    
    exit_model_ann = train_ann_pipeline(
        features_exit,
        target_col='congestion_exit_rating',
        model_save_path=str(output_path / 'exit_model_ann.pkl'),
        hidden_dims=hidden_dims,
        epochs=epochs
    )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print("\nGenerated files:")
    print("  - models/entrance_model_ann.pkl")
    print("  - models/exit_model_ann.pkl")
    
if __name__ == "__main__":
    main()