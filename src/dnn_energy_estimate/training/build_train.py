from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, Sequential

class TrainNN():
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.x_train = self.data["x_train"]
        self.y_train = self.data["y_train"]
        self.x_val = self.data["x_val"]
        self.y_val = self.data["y_val"]
        self.x_train_weights = self.data["x_train_weights"]
        self.x_val_weights = self.data["x_val_weights"]
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
            weighted_metrics=[],
        )
        return model

    def _train(self):
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train, self.x_train_weights))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.config["batch_size"])
        history = self.model.fit(
            #self.x_train,
            #self.y_train,
            train_dataset,
            validation_data=(self.x_val, self.y_val, self.x_val_weights),
            #sample_weight=self.x_train_weights,
            epochs=10,
        #    callbacks=[WandbCallback()],
#            callbacks = [TuneReportCallback({"mean_accuracy": "accuracy"})],
            verbose=0,
        )
        val_loss = history.history["val_loss"][-1]
        #    self._save({"val_loss": val_loss})
        #return {"val_loss": val_loss}
        return val_loss