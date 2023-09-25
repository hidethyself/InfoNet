import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
from parameters.parameters import training_params, seldnet_params, feature_params
from data_loader_infonet import load_data
from models.seld_net import CRNN
from early_attention_model import EarlyAttention
from denoiser.encoder_decoder import BaseEncoderDecoder
from utils import cartesian_to_polar_batch, angular_distance, create_folder


def train_epoch(model, denoiser, data_generator, optimizer, optimizer_denoiser, criterion, criterion_denoiser, device):
    model.train()
    denoiser.train()
    train_loss = 0
    denoiser_loss_ = 0
    nb_train_batches = 0
    for feat, act_feat, label, _ in tqdm(data_generator, desc="Training: "):
        feat = torch.tensor(feat).to(device).float()
        act_feat = torch.tensor(feat).to(device).float()
        label = torch.tensor(label).to(device).float()
        optimizer_denoiser.zero_grad()
        denoised_data = denoiser(feat)
        out, attention, channel_map, spatial_map = model.early_attention(feat)
        output = model.downstream_task(out + (1 - attention) * denoised_data)
        loss = criterion(output, label)
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()
        denoiser_loss = criterion_denoiser(denoised_data, act_feat)
        denoiser_loss.backward(retain_graph=True)
        denoiser_loss_ += denoiser_loss.item()
        optimizer.zero_grad()
        optimizer_denoiser.step()
        nb_train_batches += 1
    train_loss /= nb_train_batches
    denoiser_loss_ /= nb_train_batches
    return train_loss, denoiser_loss_


def test_epoch(model, denoiser, data_generator, optimizer, optimizer_denoiser, criterion, criterion_denoiser, device,
               desc="Validating", save=False, loc=None):
    model.eval()
    denoiser.eval()
    test_loss = 0
    denoiser_loss_ = 0
    nb_test_batches = 0
    average = 0
    all_results = None
    with torch.no_grad():
        for feat, act_feat, label, filenames in tqdm(data_generator, desc=desc):
            feat = torch.tensor(feat).to(device).float()
            act_feat = torch.tensor(feat).to(device).float()
            label = torch.tensor(label).to(device).float()
            denoised_data = denoiser(feat)
            denoiser_loss = criterion_denoiser(denoised_data, act_feat)
            denoiser_loss_ += denoiser_loss.item()
            out, attention, channel_map, spatial_map = model.early_attention(feat)
            mapped_data = out + torch.mul(1 - attention, denoised_data)
            output = model.downstream_task(mapped_data)
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
            filenames = np.array(filenames)
            filenames = filenames.reshape(-1, 1)
            if save:
                result = np.concatenate((polar_label, polar_output, distance.reshape(-1, 1), filenames), axis=1)
                if all_results is None:
                    all_results = result
                else:
                    all_results = np.concatenate((all_results, result), axis=0)
            average += distance.mean()

    test_loss /= nb_test_batches
    denoiser_loss_ /= nb_test_batches
    average /= nb_test_batches
    if save:
        np.save(os.path.join(loc, training_params["recnet_result_npy_name"]), all_results)
    return test_loss, denoiser_loss_, average


def train(loc, retrain=True):
    save_path = os.path.join(loc, f'results_{int(feature_params["doping_pct"]*100)}')
    create_folder(save_path)
    json_name = f'recnet_loss_info_{int(feature_params["doping_pct"]*100)}.json'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EarlyAttention(
        data_in=seldnet_params["in_feat_shape"],
        data_out=seldnet_params["out_shape"],
        params=seldnet_params
    ).to(device=device)
    denoiser = BaseEncoderDecoder().to(device=device)
    print("\nEarly attention model:")
    summary(model, seldnet_params["in_feat_shape"][1:])
    print("\nDenoiser model:")
    summary(denoiser, seldnet_params["in_feat_shape"][1:])


    model_name = f'infonet_seld_net_{int(feature_params["doping_pct"] * 100)}.h5'
    denoiser_name = f'infonet_denoiser_{int(feature_params["doping_pct"] * 100)}.h5'

    train_loader = load_data(
        data_loc=training_params["data_loc"],
        feat_path=training_params["feat_path"],
        batch_size=training_params["batch_size"]
    )
    val_loader = load_data(
        data_loc=training_params["data_loc"],
        feat_path=training_params["feat_path"],
        batch_size=training_params["batch_size"],
        type_="val"
    )
    test_loader = load_data(
        data_loc=training_params["data_loc"],
        feat_path=training_params["feat_path"],
        batch_size=training_params["batch_size"],
        type_="test"
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_params["learning_rate"])
    criterion_denoiser = nn.MSELoss()
    optimizer_denoiser = optim.Adam(denoiser.parameters(), lr=1e-4)

    best_val_epoch, best_val_loss = 0, 1e10
    train_losses, denoiser_losses, val_losses, val_denoiser_losses = [], [], [], []
    if retrain:
        flag = 0
        for epoch in range(training_params["epochs"]):
            train_loss, train_denoiser_loss = train_epoch(
                model, denoiser, train_loader, optimizer, optimizer_denoiser,
                criterion, criterion_denoiser, device
            )
            val_loss, val_denoiser_loss, avg_ang_dist = test_epoch(
                model, denoiser, val_loader, optimizer, optimizer_denoiser,
                criterion, criterion_denoiser, device
            )
            train_losses.append(train_loss)
            denoiser_losses.append(train_denoiser_loss)
            val_losses.append(val_loss)
            val_denoiser_losses.append(val_denoiser_loss)
            print(f'Epoch[{epoch + 1}]: t_loss: {train_loss} | d_t_loss: {train_denoiser_loss}| v_loss: {val_loss} | '
                  f'v_d_loss: {val_denoiser_loss} | val_ad: {avg_ang_dist}\n')
            if val_loss <= best_val_loss:
                torch.save(model.state_dict(), os.path.join(save_path, model_name))
                torch.save(denoiser.state_dict(), os.path.join(save_path, denoiser_name))
                best_val_loss = val_loss
                best_val_epoch = epoch + 1
                flag = 0
            else:
                flag += 1
            if flag > 10:
                break
        print(f'\n\n Best model saved at: {best_val_epoch}')
        loss_dict = {
            "train_loss": train_losses,
            "train_denoiser_loss": denoiser_losses,
            "val_loss": val_losses,
            "val_denoiser_loss": val_denoiser_losses,
            "best_val_epoch": best_val_epoch
        }
        with open(os.path.join(save_path, json_name), "w") as outfile:
            json.dump(loss_dict, outfile)
    print('Load best model weights')
    model.load_state_dict(torch.load(os.path.join(save_path, model_name), map_location='cpu'))
    denoiser.load_state_dict(torch.load(os.path.join(save_path, denoiser_name), map_location='cpu'))
    print('Loading unseen test dataset:')
    _, _, avg_ang_dist = test_epoch(
        model, denoiser, test_loader, optimizer, optimizer_denoiser,
        criterion, criterion_denoiser, device,
        "Testing", save=True, loc=save_path
    )
    print(f'Test avg angular distance: {avg_ang_dist} (degree)')


if __name__ == '__main__':
    train(loc="./data", retrain=True)
