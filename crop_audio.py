import os
import uuid
import wave
from pydub import AudioSegment


# 按秒截取音频
def get_part_wav(sound, start_time, end_time, part_wav_path):
    save_path = os.path.dirname(part_wav_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    start_time = int(start_time) * 1000
    end_time = int(end_time) * 1000
    word = sound[start_time:end_time]
    word.export(part_wav_path, format="wav")


def crop_wav(path, crop_len):
    for src_wav_path in os.listdir(path):
        wave_path = os.path.join(path, src_wav_path)
        print(wave_path[-4:])
        if wave_path[-4:] != '.wav':
            continue
        file = wave.open(wave_path)
        # 帧总数
        a = file.getparams().nframes
        # 采样频率
        f = file.getparams().framerate
        # 获取音频时间长度
        t = int(a / f)
        print('总时长为 %d s' % t)
        # 读取语音
        sound = AudioSegment.from_wav(wave_path)
        for start_time in range(0, t, crop_len):
            save_path = os.path.join(path, os.path.basename(wave_path)[:-4], str(uuid.uuid1()) + '.wav')
            get_part_wav(sound, start_time, start_time + crop_len, save_path)


if __name__ == '__main__':
    crop_len = 3
    crop_wav('save_audio', crop_len)
