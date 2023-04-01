import ray
from ray import tune
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import h5py
import wandb
from ray.tune.integration.wandb import WandbLoggerCallback

config = {
    "arch": tune.grid_search([[64, 32], [128, 64, 32]]),
    "batchnorm": tune.choice([True, False]),
    "drop": tune.choice([True, False]),
    "lossf": tune.choice(["mean_squared_error", "log_cosh"]),
    "activation": tune.choice(["PReLU", "ReLU"]),
    "drop_vals": tune.grid_search([0.2, 0.3, 0.4]),
    "learning_rate": tune.loguniform(1e-7, 1e-3)
}

def train_model(config):
    model = Sequential()
    model.add(layers.BatchNormalization())

    for j, item in enumerate(config["arch"]):
        model.add(layers.Dense(item))
        if config["batchnorm"] is True:
            batchnorm_str = "_bn_"
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(config["activation"]))
        if config["drop"] is True:
            dropout_str = "_do_"
            model.add(layers.Dropout(config["drop_vals"][j]))
    model.add(layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]), loss=config["lossf"])

    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    wandb.log({"val_loss": model.evaluate(x_val, y_val)})
    wandb.finish()

ray.init()
analysis = tune.run(
    train_model,
    config=config,
    num_samples=10,
    local_dir="../results",
    stop={"training_iteration": 1}
)
print("Best config: ", analysis.get_best_config(metric="val_loss"))