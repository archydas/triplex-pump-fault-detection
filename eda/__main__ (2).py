from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

from data_acquisition.utils import get_save_data
from data_processing.utils import get_save_train_test_data
import model_evaluation.model_eval_config as config


def main():

    # ----------------------
    # Load Data
    # ----------------------
    dataset = get_save_data()
    train_test_data = get_save_train_test_data(dataset)

    X_test = np.array(train_test_data.X_test)
    y_test = np.array(train_test_data.y_test)

    # ----------------------
    # Load Trained Model
    # ----------------------
    model_path = config.OUTPUT_DATA_DIR / Path("best_pump_model.h5")
    model = load_model(model_path)

    # ----------------------
    # Evaluate
    # ----------------------
    preds = np.argmax(model.predict(X_test), axis=-1)

    acc = accuracy_score(y_test, preds)
    print(f"\nPump Fault Model Test Accuracy: {acc:.4f}\n")

    # ----------------------
    # Label Mapping
    # ----------------------
    class_map = {
        0: "healthy",
        1: "seal_leak",
        2: "blocked_inlet",
        3: "bearing_wear",
        4: "valve_leak",
        5: "plunger_wear",
        6: "seal_leak + bearing_wear"
    }

    results = pd.DataFrame([y_test, preds]).T
    results.columns = ["Actual", "Prediction"]

    results["Actual"] = results["Actual"].map(class_map)
    results["Prediction"] = results["Prediction"].map(class_map)

    print(results.head(30))  # preview
    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    main()
