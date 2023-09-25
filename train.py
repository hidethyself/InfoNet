import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
from parameters.parameters import training_params, seldnet_params, feature_params
from data_loader import load_data
from models.seld_net import CRNN
from utils import cartesian_to_polar_batch, angular_distance, create_folder


def train_epoch(model, data_generator, optimizer, criterion, device):
    model.train()
    train_loss = 0
    nb_train_batches = 0
    for feat, label in tqdm(data_generator, desc="Training: "):
        feat, label = torch.tensor(feat).to(device).float(), torch.tensor(label).to(device).float()
        optimizer.zero_grad()
        output = model(feat)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        nb_train_batches += 1
    train_loss /= nb_train_batches
    return train_loss


def test_epoch(model, data_generator, criterion, device, desc="Validating", save=False, loc=None):
    model.eval()
    test_loss = 0
    nb_test_batches = 0
    average = 0
    all_results = None
    for feat, label in tqdm(data_generator, desc=desc):
        feat, label = torch.tensor(feat).to(device).float(), torch.tensor(label).to(device).float()
        output = model(feat)
        loss = criterion(output, label)
        test_loss += loss.item()
        nb_test_batches += 1
        label = label.detach().cpu().numpy()
        label = np.squeeze(label, axis=1)
        polar_label = cartesian_to_polar_batch(label)
        output = output.detach().cpu().numpy()
        output = np.squeeze(output, axis=1)
        polar_output = cartesian_to_polar_batch(output)
        distance = angular_distance(polar_label, polar_output)
        if save:
            result = np.concatenate((polar_label, polar_output, distance.reshape(-1, 1)), axis=1)
            if all_results is None:
                all_results = result
            else:
                all_results = np.concatenate((all_results, result), axis=0)
        average += distance.mean()

    test_loss /= nb_test_batches
    average /= nb_test_batches
    if save:
        np.save(os.path.join(loc, training_params["result_npy_name"]), all_results)
    return test_loss, average


def train(loc, retrain=True):
    save_path = os.path.join(loc, 'results_baseline_full_rank')
    create_folder(save_path)
    json_name = f'loss_info_baseline_full_rank.json'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CRNN(
        in_feat_shape=seldnet_params["in_feat_shape"],
        out_shape=seldnet_params["out_shape"],
        params=seldnet_params
    ).to(device=device)
    summary(model, seldnet_params["in_feat_shape"][1:])
    model_name = 'seld_net_baseline_full_rank.h5'

    train_loader = load_data(
         data_loc=training_params["data_loc"],
         params=training_params,
         batch_size=training_params["batch_size"]
    )
    val_loader = load_data(
        data_loc=training_params["data_loc"],
        params=training_params,
        batch_size=training_params["batch_size"],
        type_="val"
    )
    test_loader = load_data(
        data_loc=training_params["data_loc"],
        params=training_params,
        batch_size=training_params["batch_size"],
        type_="test"
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_params["learning_rate"])

    best_val_epoch, best_val_loss = 0, 1e10
    train_losses, val_losses = [], []
    flag = 0
    if retrain:
        for epoch in range(training_params["epochs"]):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, avg_ang_dist = test_epoch(model, val_loader, criterion, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f'Epoch[{epoch + 1}]: t_loss: {train_loss} | v_loss: {val_loss} | val_ad: {avg_ang_dist}\n')
            if val_loss <= best_val_loss:
                torch.save(model.state_dict(), os.path.join(save_path, model_name))
                best_val_loss = val_loss
                best_val_epoch = epoch + 1
                flag = 0
            else:
                flag += 1
            if flag > 15:
                break
        print(f'\n\n Best model saved at: {best_val_epoch}')
        loss_dict = {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "best_val_epoch": best_val_epoch
        }
        with open(os.path.join(save_path, json_name), "w") as outfile:
            json.dump(loss_dict, outfile)
    print('Load best model weights')
    model.load_state_dict(torch.load(os.path.join(save_path, model_name), map_location='cpu'))
    print('Loading unseen test dataset:')
    _, avg_ang_dist = test_epoch(model, test_loader, criterion, device, "Testing", save=True, loc=save_path)
    print(f'Test avg angular distance: {avg_ang_dist} (degree)')


if __name__ == '__main__':
    train(loc="./data", retrain=True)