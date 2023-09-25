import os
import argparse
import itertools
import json
import random
import shutil
import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib
import librosa
from tqdm import tqdm
from utils import next_greater_power_of_2, nCr, create_folder, trim_or_pad_audio, polar_to_cartesian
from parameters.parameters import feature_params


class ExtractFeature:
    def __init__(
            self,
            feature_params,
            full_rank = False,
            nb_channels=6,
            fs=44100,
            hop_len_s=.02,
            nb_mel_bins=64,
            data_loc="./data"
    ):
        self.eps = 1e-8
        self.feature_params = feature_params
        self.doped_pct = int(self.feature_params["doping_pct"]*100)
        self.no_dopped_channel = self.feature_params["no_dopped_channel"]
        self.nb_channels = nb_channels
        self.fs = fs
        self.hop_len_s = hop_len_s
        self.hop_len = int(self.fs * self.hop_len_s)
        self.win_len = 2 * self.hop_len
        self.nfft = next_greater_power_of_2(self.win_len)
        self.nb_mel_bins = nb_mel_bins
        self.mel_wts = librosa.filters.mel(sr=self.fs, n_fft=self.nfft, n_mels=self.nb_mel_bins).T
        self.data_loc = data_loc
        self.audio_loc = os.path.join(self.data_loc, "audios")
        self.meta_loc = os.path.join(self.data_loc, "meta_data")
        if full_rank:
            self.feat_dir = os.path.join(self.data_loc, 'features_full_rank')
            self.norm_feat_dir = os.path.join(self.data_loc, 'normalized_features_full_rank')
        else:
            self.feat_dir = os.path.join(self.data_loc, f'features_{self.doped_pct}')
            self.norm_feat_dir = os.path.join(self.data_loc, f'normalized_features_{self.doped_pct}')
        create_folder(self.feat_dir)
        create_folder(self.norm_feat_dir)
        self.label_dir = os.path.join(self.data_loc, 'label')
        create_folder(self.label_dir)
        self.audio_names = os.listdir(self.audio_loc)
        self.corelated_channel = {0: 4, 1: 5, 2: 4, 3: 5, 4: 0, 5: 1}

    def dopping_audio_pct(self, audio, pct, no_dopped_channel=1):
        total_dopping_length = int(pct * audio.shape[0])
        x = total_dopping_length
        least = int(.1 * total_dopping_length)
        maximum = int(.25 * total_dopping_length)
        loop_list = []
        while total_dopping_length > least:
            z = random.randint(least, maximum)
            if sum(loop_list) + z <= x:
                loop_list.append(z)
            total_dopping_length -= z
        if sum(loop_list) < x:
            loop_list.append(x - sum(loop_list))
        start_point = 0
        for i in range(len(loop_list)):
            channels_to_dop = random.sample(list(range(audio.shape[1])), no_dopped_channel)
            for channel_to_dop in channels_to_dop:
                end_point = start_point + loop_list[i]
                noise = np.random.normal(0, 1, loop_list[i])
                x = x + noise.shape[0]
                audio[:, channel_to_dop][start_point: end_point] = noise
            start_point = loop_list[i] + start_point
        return audio

    def get_gcc(self, linear_spectra):
        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self.nb_mel_bins, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m + 1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j * np.angle(R)))
                cc = np.concatenate((cc[:, -self.nb_mel_bins // 2:], cc[:, :self.nb_mel_bins // 2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

    def get_mel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self.nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt]) ** 2
            mel_spectra = np.dot(mag_spectra, self.mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat

    def spectrogram(self, audio_input, nb_frames):
        nb_ch = audio_input.shape[1]
        nb_bins = self.nfft // 2
        spectra = []
        for ch_cnt in range(nb_ch):
            stft_ch = librosa.core.stft(
                np.asfortranarray(audio_input[:, ch_cnt]),
                n_fft=self.nfft,
                hop_length=self.hop_len,
                win_length=self.win_len,
                window='hann'
            )
            spectra.append(stft_ch[:, :nb_frames])
        return np.array(spectra).T

    def load_audio(self, audio_path):
        audio, fs = librosa.load(audio_path, mono=False, sr=None, dtype=np.int16)
        audio = audio.T
        audio = trim_or_pad_audio(audio)
        audio = audio[:, :self.nb_channels] / 32768.0 + self.eps
        nb_feat_frames = int(len(audio) / float(self.hop_len))
        return audio, nb_feat_frames

    def extract_all_feature(self):
        for audio_name in tqdm(self.audio_names, desc="Extracting features"):
            audio, nb_feat_frames = self.load_audio(os.path.join(self.audio_loc, audio_name))
            if self.feature_params["is_doping"]:
                audio = self.dopping_audio_pct(
                    audio=audio,
                    pct=self.feature_params["doping_pct"],
                    no_dopped_channel=self.feature_params["no_dopped_channel"]
                )
            spect = self.spectrogram(audio, nb_feat_frames)
            mel_spect = self.get_mel_spectrogram(spect)
            gcc = self.get_gcc(spect)
            feat = np.concatenate((mel_spect, gcc), axis=-1)
            feat_path = os.path.join(self.feat_dir, '{}.npy'.format(audio_name.split('.')[0]))
            np.save(feat_path, feat)

    def preprocess_features(self):
        normalized_features_wts_file = os.path.join(self.data_loc, "spec_scaler")
        df = pd.read_csv(os.path.join(self.data_loc, "train.csv"))
        file_names = df.values.tolist()
        del df
        file_names = list(itertools.chain(*file_names))
        spec_scaler = preprocessing.StandardScaler()
        for file_name in tqdm(file_names, desc="Creating scaler"):
            feat_file_name = os.path.join(self.feat_dir, '{}.npy'.format(file_name.split(".")[0]))
            feat_file = np.load(feat_file_name)
            spec_scaler.partial_fit(feat_file)
            del feat_file
        joblib.dump(
            spec_scaler,
            normalized_features_wts_file
        )
        for file_name in tqdm(os.listdir(self.feat_dir), desc="Normalizing features"):
            file_name = '{}.npy'.format(file_name.split(".")[0])
            feat_file = np.load(os.path.join(self.feat_dir, file_name))
            feat_file = spec_scaler.transform(feat_file)
            np.save(
                os.path.join(self.norm_feat_dir, file_name),
                feat_file
            )
            del feat_file
        shutil.rmtree(self.feat_dir)

    def extract_all_label(self):
        for audio_name in tqdm(self.audio_names, desc="Extracting labels"):
            csv_name = '{}.csv'.format(audio_name.split(".")[0])
            fid = open(os.path.join(self.data_loc, 'meta_data', csv_name), 'r')
            lines = fid.readlines()
            fid.close()
            tmp_val = lines[1].strip().split(',')[1:]
            label_mat = polar_to_cartesian(polar=tmp_val)
            label_mat_name = os.path.join(self.label_dir, '{}.npy'.format(audio_name.split(".")[0]))
            np.save(label_mat_name, label_mat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_rank", default=False, action="store_true")
    args = parser.parse_args()
    extract_feature = ExtractFeature(
        feature_params=feature_params,
        full_rank=args.full_rank,
        data_loc="./data"
    )
    extract_feature.extract_all_feature()
    extract_feature.preprocess_features()
    if args.full_rank:
        extract_feature.extract_all_label()

if __name__ == '__main__':
    main()