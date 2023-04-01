import wandb
from wandb.keras import WandbCallback
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter, grid_search
import optuna

class MyTuner(tune.Tuner):
    def __init__(self, x_train, y_train, x_val, y_val, x_train_weights, x_val_weights, project_name):
        #super().__init__(metric="val_loss", mode="min")
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_train_weights = x_train_weights
        self.x_val_weights = x_val_weights
        self.project_name = project_name

    def setup(self, config):
        self.config = config

    def run_trial(self, trial):
        model = Sequential()
        model.add(layers.BatchNormalization())

        for j, item in enumerate(self.config["arch"]):
            model.add(layers.Dense(item))
            if self.config["batchnorm"] is True:
                batchnorm_str = "_bn_"
                model.add(layers.BatchNormalization())
            model.add(layers.Activation(config["activation"]))
            model.add(layers.Dropout(self.config["drop_vals"][j]))
        model.add(layers.Dense(1))

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config["learning_rate"]
            ),
            loss=self.config["lossf"],
        )

        history = model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_val, self.y_val, self.x_val_weights),
            sample_weight=self.x_train_weights,
            epochs=10,
            callbacks=[WandbCallback()],
            verbose=0,
        )

        # Report result to Ray Tune
        tune.report(val_loss=history.history["val_loss"][-1])

        # Log results to WandB
        wandb.init(project=self.project_name)
        wandb.config.update(self.config)
        wandb.log({"val_loss": history.history["val_loss"][-1]})
        wandb.finish()

def run_hyperparameter_search(x_train, y_train, x_val, y_val, x_train_weights, x_val_weights, project_name):
    tuner = MyTuner(x_train, y_train, x_val, y_val, x_train_weights, x_val_weights, project_name)
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

    # Configure the ConcurrencyLimiter to limit the number of concurrent
    # trainings and use the ASHA scheduler to stop unpromising trials early
    # and allocate more resources to more promising trials.
    scheduler = ASHAScheduler(
        max_t=10, grace_period=1, reduction_factor=2
    )
    # Set up Optuna as the search algorithm
    search_alg = optuna.create_study(direction="minimize")

    tune.run(
        tuner,
        resources_per_trial={"cpu=10",
        },
        config=config,
        num_samples=10,
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=tune.CLIReporter(),
        name=project_name,
    )
    

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