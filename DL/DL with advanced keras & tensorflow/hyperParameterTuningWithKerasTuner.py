
import keras_tuner as kt, tensorflow as tf, os, warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 2.  Load / split / scale MNIST
(x_all, y_all), _ = mnist.load_data()
x_all = x_all.reshape(-1, 784).astype("float32") / 255.0

# 60 % train  /  20 % val  /  20 % test
split1 = int(0.6 * len(x_all))
split2 = int(0.8 * len(x_all))
x_train, y_train = x_all[:split1], y_all[:split1]
x_val, y_val     = x_all[split1:split2], y_all[split1:split2]
x_test, y_test   = x_all[split2:], y_all[split2:]

# 3.  Model-building function
def build_model(hp):
    model = Sequential([
        Flatten(input_shape=(784,)),
        Dense(units=hp.Int("units", 32, 512, step=32), activation="relu"),
        Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer=Adam(hp.Float("learning_rate", 1e-4, 1e-2, sampling="LOG")),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 4.  Keras-Tuner RandomSearch
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=2,
    directory="kt_dir",
    project_name="mnist_tune"
)

tuner.search_space_summary()

# 5.  Run search (5 epochs per trial)
tuner.search(x_train, y_train,
             epochs=5,
             validation_data=(x_val, y_val),
             verbose=0)

tuner.results_summary()

# 6.  Train final model with best hp
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\nBest units: {best_hp.get('units')}  |  Best lr: {best_hp.get('learning_rate'):.5f}")

final_model = tuner.hypermodel.build(best_hp)
final_model.fit(x_train, y_train,
                epochs=10,
                validation_split=0.2,
                verbose=0)

# 7.  Evaluate on untouched test set
test_loss, test_acc = final_model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")