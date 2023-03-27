import numpy as np
from tensorflow import keras


import km3pipe as kp

class apply_model(kp.Module):

    def configure(self):
        # Option for having global verbosity
        self.log.setLevel(self.get("verbosity", default="WARNING"))
        self.input_dir = self.get("input_dir", default="model")
        self.pwd = self.get("pwd", default=".")
        self.output_var = self.get("output_key", default="metric")

        # Register the service to append branches to the pump
        self.require_service("branches")


    def prepare(self):
        self.branches = self.services["branches"]
        # Requiered branches can be now added :
        to_add = []
        self.input_variables = []
        for line in open(self.pwd+"/variables.txt"): # With likelihood
            self.input_variables.append(line[:-1])
            self.branches.append(line[:-1])
        # Load model
        self.model = keras.models.load_model(self.input_dir)

        # Load the variables list from file
        #for b in to_add : self.branches.append(b)

    def process(self, blob):
        # Called for each iteration of the pipeline


        # Format the data in the proper order
        input_data = []
        for var in self.input_variables:
            input_data.append(blob['tree'][var])

        # Set the shape into (n_events, n_vars)
        input_data = np.stack(input_data)
        input_data = np.swapaxes(input_data, 0,1)
        input_data[:,1] = np.log10(input_data[:,1])
        input_data[:,2] = np.log10(input_data[:,2])
        input_data[:,8] = np.log10(input_data[:,8])
        input_data[np.isnan(input_data)] = 0
        input_data[np.isinf(input_data)] = 0



        # Compute the estimate
        blob['tree'][self.output_var] = 10**self.model.predict(input_data).flatten()

        return blob

    def finish(self):
        # Called once when the pipeline is closing
        pass

