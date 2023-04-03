import make_pred

data_path = "../../../../for_pred.h5"
model_path = "../../../HPO/search_20230402/n_layers_16-n_nodes_48-batch_size_64-batchnorm_0-lossf_log_cosh-activation_PReLU-drop_vals_0.1699915220737158-learning_rate_2.0513417874545736e-06.h5"
model_name = model_path.split("/")[-1].split(".")[0]
data_name = data_path.split("/")[-1].split(".")[0]
pred_name = data_name+"_" +model_name
#load model
make_pred.make_pred(model_path, data_path, pred_path = "./", pred_name = pred_name)
