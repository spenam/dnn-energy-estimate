from tensorflow import keras
from tensorflow.keras import layers, Sequential
import optuna



class TrainNN():
    def __init__(self, config):
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
        model = Sequential()
        model.add(layers.BatchNormalization())

        for j in range(self.config["n_layers"]):
            model.add(layers.Dense(self.config["n_nodes"]))
            if self.config["batchnorm"] is True:
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
        #return {"val_loss": val_loss}
        return val_loss
    

def run_train(trial, x_train, y_train, x_val, y_val, x_train_weights, x_val_weights):
    config = {
        "n_layers": trial.suggest_int("n_layers",4,32,4),#[4,8,12,16,20,24]),
        "n_nodes": trial.suggest_int("n_nodes",16,128,16),#[16,32,64,128]),
        "batchnorm": trial.suggest_int("batchnorm",0,1),#[True, False]),
        #"drop": trial.suggest_int("drop",[True, False]),
        "lossf": trial.suggest_categorical("lossf",["mean_squared_error", "log_cosh"]),
        "activation": trial.suggest_categorical("activation",["PReLU", "ReLU"]),
        "drop_vals": trial.suggest_float("drop_vals",0,0.5),#[0., 0.1, 0.2, 0.3, 0.4, 0.5]),
        "learning_rate": trial.suggest_float("learning_rate",1e-6,1e-3,log=True),#[1e-3, 1e-4, 1e-5, 1e-6]),
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_train_weights": x_train_weights,
        "x_val_weights": x_val_weights,
    }
    train = TrainNN(config)
    #study.optimize(train.step(), n_trials=10)
    #print("Best config: ", analysis.get_best_config(metric="mse", mode="min"))
    return train.step()
def HPO(x_train, y_train, x_val, y_val, x_train_weights, x_val_weights):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: run_train(trial, x_train, y_train, x_val, y_val, x_train_weights, x_val_weights), n_trials=10)
    print("Best config: ", study.best_params)
    return study