"""Train the ticket classifier and save it to models/classifier.joblib.

Reproducible: fixed random seed, fixed stratified split, versioned CSV input.

Usage:  python scripts/train_model.py [--data PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.services.classifier import TicketClassifier  # noqa: E402


def main() -> None:
    defaults = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=defaults.training_data_path)
    parser.add_argument("--out", type=Path, default=defaults.model_path)
    args = parser.parse_args()

    classifier = TicketClassifier.train(args.data, defaults.classifier_min_confidence)
    classifier.save(args.out)

    holdout = classifier.metrics["holdout"]
    print(f"Trained on {classifier.metrics['n_training_rows']} rows from {args.data}")
    print(
        f"Hold-out ({holdout['n_test']} rows): accuracy={holdout['accuracy']:.3f} "
        f"macro-F1={holdout['macro_f1']:.3f}"
    )
    print(f"Saved model to {args.out}")
    print(classifier.metrics["note"])


if __name__ == "__main__":
    main()
