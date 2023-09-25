import os
import math
import numpy as np


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print("Folders created")
    else:
        print("Folder already exists!")


def next_greater_power_of_2(x):
    return 2 ** (x - 1).bit_length()


def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n - r)


def trim_or_pad_audio(audio, t=6, fs=44100):
    max_len = t*fs
    shape = audio.shape
    if shape[0] >= max_len:
        audio = audio[:max_len, :]
    else:
        n_pad = max_len - shape[0]
        zero_shape = (n_pad,) + (shape[1], )
        audio = np.concatenate((audio, np.zeros(zero_shape)), axis=0)
    return audio


def polar_to_cartesian(polar):
    ele_rad = int(polar[1]) * np.pi / 180
    azi_rad = int(polar[0]) * np.pi / 180
    tmp_label = np.cos(ele_rad)
    x = np.cos(azi_rad) * tmp_label
    y = np.sin(azi_rad) * tmp_label
    z = np.sin(ele_rad)
    label_mat = np.array([[x, y, z]])
    return label_mat


def cartesian_to_polar(cartesian):
    x, y, z = cartesian[0], cartesian[1], cartesian[2]
    azimuth = np.arctan2(y, x) * 180 / np.pi
    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) * 180 / np.pi
    return [azimuth, elevation]


def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def cartesian_to_polar_batch(cartesian_mat):
    x = cartesian_mat[:, 0]
    y = cartesian_mat[:, 1]
    z = cartesian_mat[:, 2]
    azimuth = np.arctan2(y, x)*(180/np.pi)
    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) * 180 / np.pi
    azimuth = np.expand_dims(azimuth, axis=1)
    elevation = np.expand_dims(elevation, axis=1)
    polar_mat = np.concatenate((azimuth, elevation), axis=1)
    return polar_mat


def angular_distance(actual, predicted):
    ele1, ele2 = actual[:, 1]*(np.pi/180), predicted[:, 1]*(np.pi/180)
    az1, az2 = actual[:, 0]*(np.pi/180), predicted[:, 0]*(np.pi/180)
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def test():
    # polar = [-105, -17]
    # cartesian = polar_to_cartesian(polar)
    # print(cartesian)
    # polar_ = cartesian_to_polar(list(cartesian[0]))
    # print(polar_)
    cartesian_mat = np.random.normal(0, 1, (10, 1, 3))
    cartesian_mat = np.squeeze(cartesian_mat, axis=1)
    actual = cartesian_to_polar_batch(cartesian_mat)
    cartesian_mat = np.random.normal(0, 1, (10, 1, 3))
    cartesian_mat = np.squeeze(cartesian_mat, axis=1)
    predicted = cartesian_to_polar_batch(cartesian_mat)
    dist = angular_distance(actual, predicted)
    print(dist, dist.mean())


if __name__ == '__main__':
    test()