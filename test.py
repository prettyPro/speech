import os


def get_wav_files(parent_dir,  sub_dirs):
    wav_files = []
    file = open("labels.txt", "w")
    for l,  sub_dir in enumerate(sub_dirs):
        wav_path = os.sep.join([parent_dir,  sub_dir])
        for (dirpath,  dirnames,  filenames) in os.walk(wav_path):
            # print("filenames:%s"%(filenames))
            for filename in filenames:
                # print("filename:%s"%(filename))
                # if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath,  filename])
                # print("filename_path:%s" % (filename_path))
                wav_files.append(filename_path)#filename_path:audio\5\3xing\xing020.wav
                label_temp = filename_path.split('\\')
                label = label_temp[1] + label_temp[2][0]
                label_pair = filename_path + " " + label
                file.write(label_pair + "\n")
    file.close()
    print("finished!")

def create():
    file = open("labels.txt", "r")
    label_pair = open("label_pair.txt", "w")
    label_set = []
    labels = file.readlines()
    for x in labels:
        pairs = x.split(" ")
        a = pairs[1].rstrip("\n")
        if a not in label_set:
            label_set.append(a)
    for y in labels:
        pairs = y.split(" ")
        a1 = pairs[0]
        a2 = pairs[1].rstrip("\n")
        a2_index = label_set.index(a2)
        s = a1 + " " + str(a2_index) + "\n"
        label_pair.write(s)
    file.close()
    label_pair.close()



if __name__ == '__main__':
    # get_wav_files("audio","1,2,3,4,5")
    create()