seldnet_params = {
    "dropout_rate": 0.05,
    "nb_cnn2d_filt": 64,
    "f_pool_size": [4, 4, 2],
    "nb_rnn_layers": 2,
    "nb_fnn_layers": 1,
    "rnn_size": 128,
    "fnn_size": 128,
    "t_pool_size": [5, 6, 10],
    "in_feat_shape": (1, 21, 300, 64),
    "out_shape": (1, 1, 3)
}


feature_params = {
    "is_doping": True,
    "doping_pct": .75,
    "no_dopped_channel": 3
}

result_npy_name = f'infer_all_results_baseline.npy'
recnet_result_npy_name = f'infer_recnet_all_results_{int(feature_params["doping_pct"]*100)}.npy'

if feature_params["is_doping"] == False:
    feat_path =  "normalized_features_full_rank"
else:
    feat_path = f'normalized_features_{int(feature_params["doping_pct"]*100)}'

training_params = {
    "data_loc": "./data",
    "loss_info": "loss_info.json",
    "feat_path": feat_path,
    "result_npy_name": result_npy_name,
    "recnet_result_npy_name": recnet_result_npy_name,
    "norm_feat_path": f'normalized_features_{int(feature_params["doping_pct"]*100)}',
    "type_": "d",  # n: normal, d: doped
    "batch_size": 128,
    "learning_rate": 1e-3,
    "epochs": 10
}
