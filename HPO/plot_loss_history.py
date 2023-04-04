import matplotlib.pyplot as plt
import json
import glob

for p in (glob.glob("fast_test/*.json")):
    path_to_history =p
    with open(path_to_history, 'r') as f:
      data = json.load(f)
    plt.plot(data["loss"])
    plt.plot(data["val_loss"])
    plt.show()