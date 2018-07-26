import os
import librosa # pip install librosa
from tqdm import tqdm # pip install tqdm 显示进度条
import tensorflow as tf
import numpy as np


# 获得训练用的wav文件路径列表
def get_wav_files(parent_dir, sub_dirs):
    wav_files = []
    for l, sub_dir in enumerate(sub_dirs):
        wav_path = os.path.join(parent_dir, sub_dir)
        for (dirpath, dirnames, filenames) in os.walk(wav_path):
            for filename in filenames:
                if filename.endswith('.wav') or filename.endswith('.WAV'):
                    filename_path = os.sep.join([dirpath, filename])
                    wav_files.append(filename_path)
    return wav_files


# 获取文件mfcc特征和对应标签
def extract_features(wav_files):
    inputs = []
    labels = []

    for wav_file in tqdm(wav_files):
        # 读入音频文件
        audio, fs = librosa.load(wav_file)

        # 获取音频mfcc特征
        # [n_steps, n_inputs]
        mfccs = np.transpose(librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=FLAGS.n_inputs), [1, 0])
        inputs.append(mfccs.tolist())
        # 获取label
    for wav_file in wav_files:
        label = wav_file.split('/')[-1].split('-')[1]
        labels.append(label)
    return inputs, np.array(labels, dtype=np.int)

if __name__ == '__main__':
    # 获得训练用的wav文件路径列表
    wav_files = get_wav_files("audio", "1,2,3,4,5")
    # 获取文件mfcc特征和对应标签
    tr_features,tr_labels = extract_features(wav_files)

    np.save('tr_features.npy',tr_features)
    np.save('tr_labels.npy',tr_labels)