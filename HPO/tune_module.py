import wandb
from wandb.keras import WandbCallback
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter, grid_search
from ray.air.config import RunConfig
import ray.air as air
import optuna

#class MyTuner(tune.Tuner):
class MyTrainable(tune.Trainable):
    #def __init__(self, x_train, y_train, x_val, y_val, x_train_weights, x_val_weights, project_name):
    #    #super().__init__(metric="val_loss", mode="min")
    #    self.x_train = x_train
    #    self.y_train = y_train
    #    self.x_val = x_val
    #    self.y_val = y_val
    #    self.x_train_weights = x_train_weights
    #    self.x_val_weights = x_val_weights
    #    self.project_name = project_name

    def _setup(self):
        self.config = config
        self.model = self.build_model()

    def _train(self):
        history = self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_val, self.y_val, self.x_val_weights),
            sample_weight=self.x_train_weights,
            epochs=10,
            callbacks=[WandbCallback()],
            verbose=0,
        )
        val_loss = history.history["val_loss"][-1]
    #    self._save({"val_loss": val_loss})
        return {"val_loss": val_loss}

    #def _save(self, checkpoint_dir):
    #    self.model.save(checkpoint_dir)

    #def _restore(self, checkpoint_dir):
    #    self.model = keras.models.load_model(checkpoint_dir)

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

    def set_data(self, x_train, y_train, x_val, y_val, x_train_weights, x_val_weights):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_train_weights = x_train_weights
        self.x_val_weights = x_val_weights

    #def run_trial(self, trial):
    #    model = Sequential()
    #    model.add(layers.BatchNormalization())

    #    for j, item in enumerate(self.config["arch"]):
    #        model.add(layers.Dense(item))
    #        if self.config["batchnorm"] is True:
    #            batchnorm_str = "_bn_"
    #            model.add(layers.BatchNormalization())
    #        model.add(layers.Activation(config["activation"]))
    #        model.add(layers.Dropout(self.config["drop_vals"][j]))
    #    model.add(layers.Dense(1))

    #    model.compile(
    #        optimizer=keras.optimizers.Adam(
    #            learning_rate=self.config["learning_rate"]
    #        ),
    #        loss=self.config["lossf"],
    #    )

    #    history = model.fit(
    #        self.x_train,
    #        self.y_train,
    #        validation_data=(self.x_val, self.y_val, self.x_val_weights),
    #        sample_weight=self.x_train_weights,
    #        epochs=10,
    #        callbacks=[WandbCallback()],
    #        verbose=0,
    #    )

    #    # Report result to Ray Tune
    #    tune.report(val_loss=history.history["val_loss"][-1])

    #    # Log results to WandB
    #    wandb.init(project=self.project_name)
    #    wandb.config.update(self.config)
    #    wandb.log({"val_loss": history.history["val_loss"][-1]})
    #    wandb.finish()

def run_hyperparameter_search(x_train, y_train, x_val, y_val, x_train_weights, x_val_weights, project_name):
    #tuner = MyTuner(x_train, y_train, x_val, y_val, x_train_weights, x_val_weights, project_name)
    config = {
    "n_layers": tune.grid_search([4,8,12,16,20,24]),
    "n_nodes": tune.grid_search([16,32,64,128]),
    "batchnorm": tune.choice([True, False]),
    "drop": tune.choice([True, False]),
    "lossf": tune.choice(["mean_squared_error", "log_cosh"]),
    "activation": tune.choice(["PReLU", "ReLU"]),
    "drop_vals": tune.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5]),
    "learning_rate": tune.choice([1e-3, 1e-4, 1e-5, 1e-6]),
}
    trainable = MyTrainable()
    trainable.set_data(x_train, y_train, x_val, y_val, x_train_weights, x_val_weights)

    # Configure the ConcurrencyLimiter to limit the number of concurrent
    # trainings and use the ASHA scheduler to stop unpromising trials early
    # and allocate more resources to more promising trials.
    scheduler = ASHAScheduler(
        max_t=10, grace_period=1, reduction_factor=2
    )
    # Set up Optuna as the search algorithm
    search_alg = optuna.create_study(direction="minimize")

    tuner = tune.Tuner(
        #tuner,
        trainable,
        tune_config=tune.TuneConfig(
            metric="mse",
            mode="min",
            #resources_per_trial={"cpu=10",       },
            num_samples=10,
            scheduler=scheduler,
            search_alg=search_alg,
        ),
        param_space = config,
        #trainable_with_cpu = tune.with_resources(trainable, {"cpu": 10}),
        #progress_reporter=tune.CLIReporter(),
        run_config=RunConfig(name=project_name,
            #local_dir="../log_directory/",
            #verbose=2,
            #checkpoint_config=air.CheckpointConfig(checkpoint_frequency=2),
            #checkpoint_dir="../checkpoint_directory/"
        )
    )
    tuner.fit()
    

    # Get best config and print it
    best_trial = tuner.get_best_trial("val_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["val_loss"]))


    # Log best config to WandB
    wandb.init(project=project_name)
    wandb.config.update(best_trial.config)
    wandb.log({"best_val_loss": best_trial.best_result["val_loss"]})
    for config in tuner.trial_dataframes["config"]:
        wandb.log(config)

    # Print all configs
    print("All configs:")
    for config in tuner.trial_dataframes["config"]:
        print(config)

    wandb.finish()
    return tuner