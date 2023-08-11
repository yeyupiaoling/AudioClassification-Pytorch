import os


# 生成数据列表
def get_data_list(audio_path, list_path):
    sound_sum = 0
    audios = os.listdir(audio_path)

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w', encoding='utf-8')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w', encoding='utf-8')
    f_label = open(os.path.join(list_path, 'label_list.txt'), 'w', encoding='utf-8')

    for i in range(len(audios)):
        f_label.write(f'{audios[i]}\n')
        sounds = os.listdir(os.path.join(audio_path, audios[i]))
        for sound in sounds:
            sound_path = os.path.join(audio_path, audios[i], sound).replace('\\', '/')
            if sound_sum % 10 == 0:
                f_test.write('%s\t%d\n' % (sound_path, i))
            else:
                f_train.write('%s\t%d\n' % (sound_path, i))
            sound_sum += 1
        print("Audio：%d/%d" % (i + 1, len(audios)))
    f_label.close()
    f_test.close()
    f_train.close()


# 下载数据方式，执行：./tools/download_3dspeaker_data.sh
# 生成生成方言数据列表
def get_language_identification_data_list(audio_path, list_path):
    labels_dict = {0: 'Standard Mandarin', 3: 'Southwestern Mandarin', 6: 'Central Plains Mandarin',
                   4: 'JiangHuai Mandarin', 2: 'Wu dialect', 8: 'Gan dialect', 9: 'Jin dialect',
                   11: 'LiaoJiao Mandarin', 12: 'JiLu Mandarin', 10: 'Min dialect', 7: 'Yue dialect',
                   5: 'Hakka dialect', 1: 'Xiang dialect', 13: 'Northern Mandarin'}

    with open(os.path.join(list_path, 'train_list.txt'), 'w', encoding='utf-8') as f:
        train_dir = os.path.join(audio_path, 'train')
        for root,  dirs, files in os.walk(train_dir):
            for file in files:
                if not file.endswith('.wav'): continue
                label = int(file.split('_')[-1].replace('.wav', '')[-2:])
                file = os.path.join(root, file)
                f.write(f'{file}\t{label}\n')

    with open(os.path.join(list_path, 'test_list.txt'), 'w', encoding='utf-8') as f:
        test_dir = os.path.join(audio_path, 'test')
        for root,  dirs, files in os.walk(test_dir):
            for file in files:
                if not file.endswith('.wav'): continue
                label = int(file.split('_')[-1].replace('.wav', '')[-2:])
                file = os.path.join(root, file)
                f.write(f'{file}\t{label}\n')

    with open(os.path.join(list_path, 'label_list.txt'), 'w', encoding='utf-8') as f:
        for i in range(len(labels_dict)):
            f.write(f'{labels_dict[i]}\n')


# 创建UrbanSound8K数据列表
def create_UrbanSound8K_list(audio_path, metadata_path, list_path):
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
        sound_path = os.path.join(audio_path, f'fold{data[5]}', data[0]).replace('\\', '/')
        if sound_sum % 10 == 0:
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
    # 生成生成方言数据列表
    # get_language_identification_data_list(audio_path='dataset/language',
    #                                       list_path='dataset/')
    # 创建UrbanSound8K数据列表
    create_UrbanSound8K_list(audio_path='dataset/UrbanSound8K/audio',
                             metadata_path='dataset/UrbanSound8K/metadata/UrbanSound8K.csv',
                             list_path='dataset')
