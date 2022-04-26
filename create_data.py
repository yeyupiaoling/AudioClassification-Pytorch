import os

import librosa


# 生成数据列表
def get_data_list(audio_path, list_path):
    sound_sum = 0
    audios = os.listdir(audio_path)

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w')
    f_label = open(os.path.join(list_path, 'label_list.txt'), 'w')

    for i in range(len(audios)):
        f_label.write(f'{audios[i]}\n')
        sounds = os.listdir(os.path.join(audio_path, audios[i]))
        for sound in sounds:
            if '.wav' not in sound:continue
            sound_path = os.path.join(audio_path, audios[i], sound)
            t = librosa.get_duration(filename=sound_path)
            if t < 3:continue
            if sound_sum % 100 == 0:
                f_test.write('%s\t%d\n' % (sound_path, i))
            else:
                f_train.write('%s\t%d\n' % (sound_path, i))
            sound_sum += 1
        print("Audio：%d/%d" % (i + 1, len(audios)))
    f_label.close()
    f_test.close()
    f_train.close()


if __name__ == '__main__':
    get_data_list('dataset/UrbanSound8K/audio', 'dataset')