import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


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

    X_train = np.array(train_test_data.X_train)
    X_test = np.array(train_test_data.X_test)

    y_train = np.array(train_test_data.y_train)
    y_test = np.array(train_test_data.y_test)

    logging.info(f"Training samples: {X_train.shape}")
    logging.info(f"Testing samples: {X_test.shape}")

    model = Sequential()
    from tensorflow.keras.layers import Dropout

    model.add(LSTM(config.LSTM_UNITS,
               return_sequences=True,
               input_shape=(X_train.shape[1], X_train.shape[2])))

    model.add(Dropout(0.3))

    model.add(LSTM(64))

    model.add(Dense(config.OUTPUT_SIZE, activation=config.ACTIVATION))

    print(model.summary())

    model_path = config.OUTPUT_DATA_DIR / Path('best_pump_model.h5')

    chk = ModelCheckpoint(filepath=model_path,
                          monitor=config.MONITOR,
                          save_best_only=True,
                          mode='auto',
                          verbose=1)

    model.compile(loss=config.LOSS_FUNCTION,
                  optimizer=config.OPTIMIZER,
                  metrics=['accuracy'])

#     hist = model.fit(
#     X_train,
#     y_train,
#     epochs=config.EPOCHS,
#     batch_size=max(16, int(X_train.shape[0] / 4)),
#     callbacks=[chk],
#     validation_split=config.VAL_SPLIT,
#     verbose=1
# )
    from tensorflow.keras.callbacks import EarlyStopping

    es = EarlyStopping(
          monitor="val_loss",
          patience=8,
          restore_best_weights=True
      )

    hist = model.fit(
          X_train,
          y_train,
          epochs=config.EPOCHS,
          batch_size=16,
          callbacks=[chk, es],
          validation_data=(X_test, y_test),   # <<< IMPORTANT
          verbose=1
      )




    logging.info("Training complete.")

    _, axs = plt.subplots(figsize=(11, 9))
    file_location = config.OUTPUT_DATA_DIR / Path('model_accuracy.png')
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    axs.set_title("Pump Fault LSTM Accuracy")
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.savefig(file_location)

    logging.info(f"Model saved → {model_path}")
    logging.info("Accuracy plot saved.")


if __name__ == "__main__":
    main()

