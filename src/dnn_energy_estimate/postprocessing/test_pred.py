import postprocessing.make_pred
import os

data_path = "../../../../for_pred.h5"
#model_path = "../../../HPO/search_20230402/n_layers_16-n_nodes_48-batch_size_64-batchnorm_0-lossf_log_cosh-activation_PReLU-drop_vals_0.1699915220737158-learning_rate_2.0513417874545736e-06.h5"
model_path = "../../../HPO/fast_test/n_layers_12-n_nodes_32-lossf_log_cosh-activation_PReLU-drop_vals_0.1-learning_rate_1e-05-l2_0.0015473380105786125-features_fts_1_val-loss_0.0418990.h5"
model_name = model_path.split("/")[-1].split(".h5")[0]
data_name = data_path.split("/")[-1].split(".")[0]
pred_name = data_name+"_" +model_name
fts ={
    "fts_1" : ['JSHOWERFIT_ENERGY', 'JENERGY_ENERGY', 'lik_first_JENERGY', 'lik_first_JSHOWERFIT', 'trig_hits', 'trig_doms', 'trig_lines', 'JSTART_LENGTH_METRES', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf','dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit'],
    "fts_2" : ['JSTART_LENGTH_METRES', 'JSHOWERFIT_ENERGY', 'JENERGY_ENERGY', 'trig_hits', 'trig_doms', 'trig_lines', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf', 'dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit', 'lik_first_JENERGY', 'lik_first_JSHOWERFIT',  'trackscore', 'muonscore'],
    "fts_3" : ['JSTART_LENGTH_METRES', 'JSHOWERFIT_ENERGY', 'JENERGY_ENERGY', 'trig_hits', 'trig_doms', 'trig_lines', 'dir_x_gandalf', 'dir_y_gandalf', 'dir_z_gandalf', 'dir_x_showerfit', 'dir_y_showerfit', 'dir_z_showerfit', 'lik_first_JENERGY', 'lik_first_JSHOWERFIT', 'pos_x_gandalf', 'pos_y_gandalf', 'pos_z_gandalf', 'pos_x_showerfit', 'pos_y_showerfit', 'pos_z_showerfit', 'trackscore', 'muonscore'],
    }
#load model
#make_pred.make_pred(model_path, data_path, fts["fts_1"], pred_path = "./", pred_name = pred_name)

def make_test(model_path, features, study_path):
    data_path = "../../for_pred.h5"
    model_name = model_path.split("/")[-1].split(".h5")[0]
    data_name = data_path.split("/")[-1].split(".")[0]
    pred_name = data_name+"_" +model_name
    pred_path = study_path
    postprocessing.make_pred.make_pred(model_path, data_path, fts[features], pred_path = pred_path, pred_name = pred_name)
    return os.path.abspath(pred_path + "/pred_y/" + pred_name+".h5")

