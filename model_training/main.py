from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging

from data_acquisition.utils import get_save_data
from data_processing.utils import get_save_train_test_data
import model_training.model_training_config as config

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint

logging.basicConfig(level=logging.INFO)


def main():

    config.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading raw pump data...")
    dataset = get_save_data()

    logging.info("Processing dataset and creating train/test split...")
    train_test_data = get_save_train_test_data(dataset)

    # Prepare arrays
    X_train = np.array(train_test_data.X_train)
    X_test = np.array(train_test_data.X_test)

    y_train = np.array(train_test_data.y_train)
    y_test = np.array(train_test_data.y_test)

    logging.info(f"Training samples: {X_train.shape}")
    logging.info(f"Testing samples: {X_test.shape}")

    # ------------------------------
    # Model Definition
    # ------------------------------
    model = Sequential()
    model.add(
        LSTM(
            config.LSTM_UNITS,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
    model.add(Dense(config.OUTPUT_SIZE, activation=config.ACTIVATION))

    print(model.summary())

    # ------------------------------
    # Training
    # ------------------------------
    model_path = config.OUTPUT_DATA_DIR / Path('best_pump_model.h5')

    chk = ModelCheckpoint(
        filepath=model_path,
        monitor=config.MONITOR,
        save_best_only=True,
        mode='auto',
        verbose=1
    )

    model.compile(
        loss=config.LOSS_FUNCTION,
        optimizer=config.OPTIMIZER,
        metrics=['accuracy']
    )

    hist = model.fit(
        X_train,
        y_train,
        epochs=config.EPOCHS,
        batch_size=int(X_train.shape[0] / 4),
        callbacks=[chk],
        validation_split=config.VAL_SPLIT,
        verbose=1
    )

    logging.info("Training complete.")

    # ------------------------------
    # Save Accuracy Plot
    # ------------------------------
    _, axs = plt.subplots(nrows=1, figsize=(11, 9))
    file_location = config.OUTPUT_DATA_DIR / Path('model_accuracy.png')
    plt.rcParams['font.size'] = '14'

    for label in (axs.get_xticklabels() + axs.get_yticklabels()):
        label.set_fontsize(14)

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])

    axs.set_title('Pump Fault LSTM Model Accuracy')
    axs.set_ylabel('Accuracy')
    axs.set_xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(file_location)

    logging.info(f"Model saved → {model_path}")
    logging.info("Training accuracy plot saved.")


if __name__ == "__main__":
    main()
