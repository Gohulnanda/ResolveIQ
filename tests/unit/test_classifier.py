import pandas as pd
import pytest

from app.services.classifier import (
    CATEGORIES,
    TicketClassifier,
    load_dataset,
    train_and_evaluate,
)


@pytest.fixture(scope="module")
def classifier(container):
    return container.classifier


def test_predicts_clear_network_ticket(classifier):
    prediction = classifier.predict(
        "Cannot connect to VPN", "The VPN client shows a timeout error after I enter my credentials."
    )
    assert prediction.category == "Network"
    assert not prediction.low_confidence
    assert prediction.alternatives[0][0] == "Network"
    assert len(prediction.alternatives) == 3


def test_nonsense_is_low_confidence_and_routed_to_other(classifier):
    prediction = classifier.predict("asdf qwer zxcv", "lorem ipsum dolor sit amet")
    assert prediction.low_confidence
    assert prediction.category == "Other"
    assert any("Low confidence" in w for w in prediction.warnings)


def test_short_text_gets_warning(classifier):
    prediction = classifier.predict("VPN down", "cannot connect")
    assert any("short" in w.lower() for w in prediction.warnings)


def test_metrics_are_present_and_labelled_synthetic(classifier):
    holdout = classifier.metrics["holdout"]
    assert holdout["labels"] == list(CATEGORIES)
    assert len(holdout["confusion_matrix"]) == len(CATEGORIES)
    assert "synthetic" in classifier.metrics["note"].lower()
    assert classifier.holdout_macro_f1 == holdout["macro_f1"]


def test_training_is_deterministic(settings):
    df = load_dataset(settings.training_data_path)
    _, first = train_and_evaluate(df)
    _, second = train_and_evaluate(df)
    assert first["holdout"] == second["holdout"]


def test_save_load_roundtrip(classifier, tmp_path):
    path = tmp_path / "m.joblib"
    classifier.save(path)
    loaded = TicketClassifier.load(path, 0.3)
    a = classifier.predict("Printer offline", "The printer shows offline although it is powered on.")
    b = loaded.predict("Printer offline", "The printer shows offline although it is powered on.")
    assert (a.category, a.confidence) == (b.category, b.confidence)


def test_load_or_train_trains_when_missing_and_recovers_from_corruption(settings, tmp_path):
    path = tmp_path / "sub" / "m.joblib"
    created = TicketClassifier.load_or_train(path, settings.training_data_path, 0.3)
    assert path.exists() and created.predict("Forgot password", "I forgot my password and I am locked out").category
    path.write_bytes(b"not a model")
    recovered = TicketClassifier.load_or_train(path, settings.training_data_path, 0.3)
    assert recovered.metrics["holdout"]["n_test"] > 0


def test_load_dataset_validation(tmp_path):
    bad_columns = tmp_path / "a.csv"
    pd.DataFrame({"title": ["x"]}).to_csv(bad_columns, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(bad_columns)

    bad_category = tmp_path / "b.csv"
    pd.DataFrame({"title": ["x"], "description": ["y"], "category": ["Weather"]}).to_csv(
        bad_category, index=False
    )
    with pytest.raises(ValueError, match="unknown categories"):
        load_dataset(bad_category)

    empty = tmp_path / "c.csv"
    pd.DataFrame({"title": [None], "description": [None], "category": [None]}).to_csv(empty, index=False)
    with pytest.raises(ValueError, match="no usable rows"):
        load_dataset(empty)
