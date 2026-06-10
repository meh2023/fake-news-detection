"""Train the fake news detection model."""

from src.train_model import train

if __name__ == "__main__":
    print("Training fake news detection model...")
    metrics = train()

    print(f"\nAccuracy: {metrics['accuracy']:.2%}")
    print(f"Training samples: {metrics['train_samples']}")
    print(f"Test samples: {metrics['test_samples']}")
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("\nModel saved to models/fake_news_model.joblib")
