import os

import librosa
import soundfile


# 生成数据列表
def get_data_list(audio_path, list_path, min_duration=0.5):
    sound_sum = 0
    audios = os.listdir(audio_path)

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w', encoding='utf-8')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w', encoding='utf-8')
    f_label = open(os.path.join(list_path, 'label_list.txt'), 'w', encoding='utf-8')

    for i in range(len(audios)):
        f_label.write(f'{audios[i]}\n')
        sounds = os.listdir(os.path.join(audio_path, audios[i]))
        for sound in sounds:
            if '.wav' not in sound and '.mp3' not in sound:continue
            sound_path = os.path.join(audio_path, audios[i], sound)
            # 过滤小于1s的音频，数据太短不利于训练，如果觉得必须，可以去掉过滤
            t = librosa.get_duration(filename=sound_path)
            if t <= min_duration:continue
            if sound_sum % 100 == 0:
                f_test.write('%s\t%d\n' % (sound_path, i))
            else:
                f_train.write('%s\t%d\n' % (sound_path, i))
            sound_sum += 1
        print("Audio：%d/%d" % (i + 1, len(audios)))
    f_label.close()
    f_test.close()
    f_train.close()


def create_UrbanSound8K_list(audio_path, metadata_path, list_path, min_duration=0.5):
    sound_sum = 0

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w', encoding='utf-8')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w', encoding='utf-8')
    f_label = open(os.path.join(list_path, 'label_list.txt'), 'w', encoding='utf-8')

    with open(metadata_path) as f:
        lines = f.readlines()

    labels = {}
    for i, line in enumerate(lines):
        if i == 0:continue
        data = line.replace('\n', '').split(',')
        class_id = int(data[6])
        if class_id not in labels.keys():
            labels[class_id] = data[-1]
        sound_path = os.path.join(audio_path, f'fold{data[5]}', data[0])
        t = librosa.get_duration(filename=sound_path)
        if t <= min_duration: continue
        if sound_sum % 100 == 0:
            f_test.write(f'{sound_path}\t{data[6]}\n')
        else:
            f_train.write(f'{sound_path}\t{data[6]}\n')
        sound_sum += 1
    for i in range(len(labels)):
        f_label.write(f'{labels[i]}\n')
    f_label.close()
    f_test.close()
    f_train.close()


if __name__ == '__main__':
    # get_data_list('dataset/audio', 'dataset')
    create_UrbanSound8K_list('dataset/UrbanSound8K/audio', 'dataset/UrbanSound8K/metadata/UrbanSound8K.csv', 'dataset')