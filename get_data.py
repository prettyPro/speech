import os
import librosa # pip install librosa
from tqdm import tqdm # pip install tqdm 显示进度条
import tensorflow as tf
import numpy as np

# FLAGS.n_inputs  40
# 获取文件路径
def get_wav_files(parent_dir,  sub_dirs):
    wav_files = []
    for l,  sub_dir in enumerate(sub_dirs):
        wav_path = os.sep.join([parent_dir,  sub_dir])
        for (dirpath,  dirnames,  filenames) in os.walk(wav_path):
            for filename in filenames:
                filename_path = os.sep.join([dirpath,  filename])
                wav_files.append(filename_path)#filename_path:audio\5\3xing\xing020.wav
    return wav_files

# 获取文件mfcc特征和对应label
def extract_features(wav_files):
    inputs = []
    labels_set = []
    labels_repeat = []
    labels_unique = []
    # for wav_file in tqdm(wav_files):
    #     # 读入音频文件
    #     audio, fs = librosa.load(wav_file)
    #     # 获取音频mfcc特征[n_steps,  n_inputs](分帧的数量，特征)
    #     mfccs = np.transpose(librosa.feature.mfcc(y=audio,  sr=fs,  n_mfcc=40),  [1, 0])
    #     inputs.append(mfccs.tolist())

    #获取对应label
    for wav_file in wav_files:
        label_temp = wav_file.split('\\')
        label = label_temp[1] + label_temp[2]
        labels_repeat.append(label)
    for x in labels_repeat:
        if x not in labels_unique:
            labels_unique.append(x)
    for y in labels_repeat:
        y_index = labels_unique.index(y)
        labels_set.append(y_index)
    return inputs,  np.array(labels_set,  dtype=np.int)

if __name__ == '__main__':
    wav_files = get_wav_files("audio", "1,2,3,4,5")
    train_features, train_labels = extract_features(wav_files)

    np.save('train_features.npy', train_features)
    np.save('train_labels.npy', train_labels)
