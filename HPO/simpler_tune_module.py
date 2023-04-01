import wandb
from wandb.keras import WandbCallback
from ray import tune
import ray
from ray.tune.integration.wandb import WandbLoggerCallback
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback
#from ray.tune.suggest import ConcurrencyLimiter, grid_search
from ray.air.config import RunConfig
import ray.air as air
import optuna




class TrainNN(tune.Trainable):
    def setup(self, config):
        self.config = config
        self.x_train = self.config["x_train"]
        self.y_train = self.config["y_train"]
        self.x_val = self.config["x_val"]
        self.y_val = self.config["y_val"]
        self.x_train_weights = self.config["x_train_weights"]
        self.x_val_weights = self.config["x_val_weights"]
        self.model = self.build_model()
        
    
    def step(self):
        result = self._train()
        return result

    def build_model(self):
        model = keras.Sequential()
        model.add(layers.BatchNormalization())

        for j in range(self.config["n_layers"]):
            model.add(layers.Dense(self.config["n_nodes"]))
            if self.config["batchnorm"] is True:
                #batchnorm_str = "_bn_"
                model.add(layers.BatchNormalization())
            model.add(layers.Activation(self.config["activation"]))
            model.add(layers.Dropout(self.config["drop_vals"]))
        model.add(layers.Dense(1))

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config["learning_rate"]
            ),
            loss=self.config["lossf"],
        )
        return model

    def _train(self):
        history = self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_val, self.y_val, self.x_val_weights),
            sample_weight=self.x_train_weights,
            epochs=10,
        #    callbacks=[WandbCallback()],
#            callbacks = [TuneReportCallback({"mean_accuracy": "accuracy"})],
            verbose=0,
        )
        val_loss = history.history["val_loss"][-1]
        #    self._save({"val_loss": val_loss})
        return {"val_loss": val_loss}
    
    def set_data(self, x_train, y_train, x_val, y_val, x_train_weights, x_val_weights):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_train_weights = x_train_weights
        self.x_val_weights = x_val_weights


def run_train(x_train, y_train, x_val, y_val, x_train_weights, x_val_weights):
    config = {
        "n_layers": tune.grid_search([4,8,12,16,20,24]),
        "n_nodes": tune.grid_search([16,32,64,128]),
        "batchnorm": tune.choice([True, False]),
        "drop": tune.choice([True, False]),
        "lossf": tune.choice(["mean_squared_error", "log_cosh"]),
        "activation": tune.choice(["PReLU", "ReLU"]),
        "drop_vals": tune.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5]),
        "learning_rate": tune.choice([1e-3, 1e-4, 1e-5, 1e-6]),
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_train_weights": x_train_weights,
        "x_val_weights": x_val_weights,
    }
    ray.init(num_cpus=1, num_gpus=0, _memory=1 * 1024 ** 3)
    print(ray.available_resources())
    resources_per_trial = {
        "cpu": 1,
        "memory": 1 * 1024 ** 3  # 2GB of memory
    }
    #analysis = tune.run(
    #    TrainNN,
    #    config=config,
    #    stop={"training_iteration": 2},
    #    verbose=1,
    #    resources_per_trial=resources_per_trial,
    #    num_samples = 1,
    #)
    analysis = tune.Tuner(
        tune.with_resources(
        TrainNN, resources={"cpu": 2}
    ),
        #TrainNN,
        run_config=RunConfig(
            name="test_tune",
            stop={"training_iteration": 4 },
            #checkpoint_config=air.CheckpointConfig(
            #    checkpoint_frequency=perturbation_interval,
            #    #checkpoint_score_attribute="mean_accuracy",
            #    num_to_keep=2,
            #),
        ),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(
        max_t=10, grace_period=1, reduction_factor=2
    ),
            metric="mean_squared_error",
            mode="min",
            num_samples=2,
        ),
        param_space=config,
    )
    analysis.fit()
    print("Best config: ", analysis.get_best_config(metric="mse", mode="min"))
    return analysis.dataframe()
