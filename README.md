# 前言

本章我们来介绍如何使用Pytorch训练一个区分不同音频的分类模型，例如你有这样一个需求，需要根据不同的鸟叫声识别是什么种类的鸟，这时你就可以使用这个方法来实现你的需求了。

# 环境准备

主要介绍libsora，PyAudio，pydub的安装，其他的依赖包根据需要自行安装。

- Python 3.7
- Pytorch 1.8.1

## 安装libsora

最简单的方式就是使用pip命令安装，如下：

```shell
pip install pytest-runner
pip install librosa
```

如果pip命令安装不成功，那就使用源码安装，下载源码：[https://github.com/librosa/librosa/releases/](https://github.com/librosa/librosa/releases/)， windows的可以下载zip压缩包，方便解压。

```shell
pip install pytest-runner
tar xzf librosa-<版本号>.tar.gz 或者 unzip librosa-<版本号>.tar.gz
cd librosa-<版本号>/
python setup.py install
```

如果出现 `libsndfile64bit.dll': error 0x7e`错误，请指定安装版本0.6.3，如 `pip install librosa==0.6.3`

安装ffmpeg， 下载地址：[http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/](http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/)，笔者下载的是64位，static版。
然后到C盘，笔者解压，修改文件名为 `ffmpeg`，存放在 `C:\Program Files\`目录下，并添加环境变量 `C:\Program Files\ffmpeg\bin`

最后修改源码，路径为 `C:\Python3.7\Lib\site-packages\audioread\ffdec.py`，修改32行代码，如下：

```python
COMMANDS = ('C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe', 'avconv')
```

## 安装PyAudio

使用pip安装命令，如下：

```shell
pip install pyaudio
```

在安装的时候需要使用到C++库进行编译，如果读者的系统是windows，Python是3.7，可以在这里下载whl安装包，下载地址：[https://github.com/intxcc/pyaudio_portaudio/releases](https://github.com/intxcc/pyaudio_portaudio/releases)

## 安装pydub

使用pip命令安装，如下：

```shell
pip install pydub
```

# 训练分类模型

把音频转换成训练数据最重要的是使用了librosa，使用librosa可以很方便得到音频的梅尔频谱（Mel Spectrogram），使用的API为 `librosa.feature.melspectrogram()`，输出的是numpy值，可以直接用tensorflow训练和预测。关于梅尔频谱具体信息读者可以自行了解，跟梅尔频谱同样很重要的梅尔倒谱（MFCCs）更多用于语音识别中，对应的API为 `librosa.feature.mfcc()`。同样以下的代码，就可以获取到音频的梅尔频谱。

```python
wav, sr = librosa.load(data_path, sr=16000)
spec_mag = librosa.feature.melspectrogram(y=wav, sr=sr, hop_length=256)
```

## 生成数据列表

生成数据列表，用于下一步的读取需要，`audio_path`为音频文件路径，用户需要提前把音频数据集存放在`dataset/audio`目录下，每个文件夹存放一个类别的音频数据，每条音频数据长度在3秒以上，如 `dataset/audio/鸟叫声/······`。`audio`是数据列表存放的位置，生成的数据类别的格式为 `音频路径\t音频对应的类别标签`，音频路径和标签用制表符 `\t`分开。读者也可以根据自己存放数据的方式修改以下函数。

Urbansound8K 是目前应用较为广泛的用于自动城市环境声分类研究的公共数据集，包含10个分类：空调声、汽车鸣笛声、儿童玩耍声、狗叫声、钻孔声、引擎空转声、枪声、手提钻、警笛声和街道音乐声。数据集下载地址：[https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz](https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz)。以下是针对Urbansound8K生成数据列表的函数。如果读者想使用该数据集，请下载并解压到 `dataset`目录下，把生成数据列表代码改为以下代码。

```python
# 生成数据列表
def get_data_list(audio_path, list_path):
    sound_sum = 0
    audios = os.listdir(audio_path)

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w')

    for i in range(len(audios)):
        sounds = os.listdir(os.path.join(audio_path, audios[i]))
        for sound in sounds:
            if '.wav' not in sound:continue
            sound_path = os.path.join(audio_path, audios[i], sound)
            t = librosa.get_duration(filename=sound_path)
            # 过滤小于2.1秒的音频
            if t >= 2.1:
                if sound_sum % 100 == 0:
                    f_test.write('%s\t%d\n' % (sound_path, i))
                else:
                    f_train.write('%s\t%d\n' % (sound_path, i))
                sound_sum += 1
        print("Audio：%d/%d" % (i + 1, len(audios)))

    f_test.close()
    f_train.close()


if __name__ == '__main__':
    get_data_list('dataset/UrbanSound8K/audio', 'dataset')
```


创建 `reader.py`用于在训练时读取数据。编写一个 `CustomDataset`类，用读取上一步生成的数据列表。

```python
class CustomDataset(Dataset):
    def __init__(self, data_list_path, model='train', spec_len=128):
        super(CustomDataset, self).__init__()
        with open(data_list_path, 'r') as f:
            self.lines = f.readlines()
        self.model = model
        self.spec_len = spec_len

    def __getitem__(self, idx):
        audio_path, label = self.lines[idx].replace('\n', '').split('\t')
        spec_mag = load_audio(audio_path, mode=self.model, spec_len=self.spec_len)
        return spec_mag, np.array(int(label), dtype=np.int64)

    def __len__(self):
        return len(self.lines)
```

下面是在训练时或者测试时读取音频数据，训练时对转换的梅尔频谱数据随机裁剪，如果是测试，就取前面的，最好要执行归一化。

```python
def load_audio(audio_path, mode='train', spec_len=128):
    # 读取音频数据
    wav, sr = librosa.load(audio_path, sr=16000)
    spec_mag = librosa.feature.melspectrogram(y=wav, sr=sr, hop_length=256)
    if mode == 'train':
        crop_start = random.randint(0, spec_mag.shape[1] - spec_len)
        spec_mag = spec_mag[:, crop_start:crop_start + spec_len]
    else:
        spec_mag = spec_mag[:, :spec_len]
    mean = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    spec_mag = (spec_mag - mean) / (std + 1e-5)
    spec_mag = spec_mag[np.newaxis, :]
    return spec_mag
```

## 训练

接着就可以开始训练模型了，创建 `train.py`。我们搭建简单的卷积神经网络，如果音频种类非常多，可以适当使用更大的卷积神经网络模型。通过把音频数据转换成梅尔频谱，数据的shape也相当于灰度图，所以为 `(1, 128, 128)`。然后定义优化方法和获取训练和测试数据。要注意 `CLASS_DIM`参数的值，这个是类别的数量，要根据你数据集中的分类数量来修改。

```python
def train(args):
    # 数据输入的形状
    input_shape = eval(args.input_shape)
    # 获取数据
    train_dataset = CustomDataset(args.train_list_path, model='train', spec_len=input_shape[3])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_dataset = CustomDataset(args.test_list_path, model='test', spec_len=input_shape[3])
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # 获取模型
    device = torch.device("cuda")
    model = resnet34(num_classes=args.num_classes)
    model.to(device)

    # 获取优化方法
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=5e-4)
    # 获取学习率衰减函数
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.8, verbose=True)

    # 获取损失函数
    loss = torch.nn.CrossEntropyLoss()
```

最后执行训练，每100个batch打印一次训练日志，训练一轮之后执行测试和保存模型，在测试时，把每个batch的输出都统计，最后求平均值。保存的模型为预测模型，方便之后的预测使用。

```python
    for epoch in range(args.num_epoch):
        loss_sum = []
        accuracies = []
        for batch_id, (spec_mag, label) in enumerate(train_loader):
            spec_mag = spec_mag.to(device)
            label = label.to(device).long()
            output = model(spec_mag)
            # 计算损失值
            los = loss(output, label)
            optimizer.zero_grad()
            los.backward()
            optimizer.step()

            # 计算准确率
            output = torch.nn.functional.softmax(output)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            accuracies.append(acc)
            loss_sum.append(los)
            if batch_id % 100 == 0:
                print('[%s] Train epoch %d, batch: %d/%d, loss: %f, accuracy: %f' % (
                    datetime.now(), epoch, batch_id, len(train_loader), sum(loss_sum) / len(loss_sum), sum(accuracies) / len(accuracies)))
        scheduler.step()
        # 评估模型
        acc = test(model, test_loader, device)
        print('='*70)
        print('[%s] Test %d, accuracy: %f' % (datetime.now(), epoch, acc))
        print('='*70)
        model_path = os.path.join(args.save_model, 'resnet34.pth')
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.jit.save(torch.jit.script(model), model_path)
```

# 预测

在训练结束之后，我们得到了一个预测模型，有了预测模型，执行预测非常方便。我们使用这个模型预测音频，在执行预测之前，需要把音频转换为梅尔频谱数据，并把数据shape转换为(1, 1, 128, 128)，第一个为输入数据的batch大小，如果想多个音频一起数据，可以把他们存放在list中一起预测。最后输出的结果即为预测概率最大的标签。

```python
model_path = 'models/resnet34.pth'
device = torch.device("cuda")
model = torch.jit.load(model_path)
model.to(device)
model.eval()


# 读取音频数据
def load_data(data_path, spec_len=128):
    # 读取音频
    wav, sr = librosa.load(data_path, sr=16000)
    spec_mag = librosa.feature.melspectrogram(y=wav, sr=sr, hop_length=256).astype(np.float32)
    mean = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    spec_mag = (spec_mag - mean) / (std + 1e-5)
    spec_mag = spec_mag[np.newaxis, np.newaxis, :]
    return spec_mag


def infer(audio_path):
    data = load_data(audio_path)
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    output = model(data)
    result = torch.nn.functional.softmax(output)
    result = result.data.cpu().numpy()
    print(result)
    # 显示图片并输出结果最大的label
    lab = np.argsort(result)[0][-1]
    return lab


if __name__ == '__main__':
    # 要预测的音频文件
    path = 'dataset/UrbanSound8K/audio/fold5/156634-5-2-5.wav'
    label = infer(path)
    print('音频：%s 的预测结果标签为：%d' % (path, label))
```

# 其他

为了方便读取录制数据和制作数据集，这里提供了两个程序，首先是 `record_audio.py`，这个用于录制音频，录制的音频帧率为44100，通道为1，16bit。

```python
import pyaudio
import wave
import uuid
from tqdm import tqdm
import os

s = input('请输入你计划录音多少秒：')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = int(s)
WAVE_OUTPUT_FILENAME = "save_audio/%s.wav" % str(uuid.uuid1()).replace('-', '')

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("开始录音, 请说话......")

frames = []

for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS))):
    data = stream.read(CHUNK)
    frames.append(data)

print("录音已结束!")

stream.stop_stream()
stream.close()
p.terminate()

if not os.path.exists('save_audio'):
    os.makedirs('save_audio')

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print('文件保存在：%s' % WAVE_OUTPUT_FILENAME)
os.system('pause')
```

创建 `crop_audio.py`，在训练是只是裁剪前面的3秒的音频，所以我们要把录制的硬盘安装每3秒裁剪一段，把裁剪后音频存放在音频名称命名的文件夹中。最后把这些文件按照训练数据的要求创建数据列表和训练数据。

```python
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
```

创建 `infer_record.py`，这个程序是用来不断进行录音识别，录音时间之所以设置为6秒，就是要保证裁剪后的音频长度大于等于2.97秒。因为识别的时间比较短，所以我们可以大致理解为这个程序在实时录音识别。通过这个应该我们可以做一些比较有趣的事情，比如把麦克风放在小鸟经常来的地方，通过实时录音识别，一旦识别到有鸟叫的声音，如果你的数据集足够强大，有每种鸟叫的声音数据集，这样你还能准确识别是那种鸟叫。如果识别到目标鸟类，就启动程序，例如拍照等等。

```python
# 加载模型
model_path = 'models/resnet34.pth'
device = torch.device("cuda")
model = torch.jit.load(model_path)
model.to(device)
model.eval()

# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 6
WAVE_OUTPUT_FILENAME = "infer_audio.wav"

# 打开录音
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


# 读取音频数据
def load_data(data_path, spec_len=128):
    # 读取音频
    wav, sr = librosa.load(data_path, sr=16000)
    spec_mag = librosa.feature.melspectrogram(y=wav, sr=sr, hop_length=256).astype(np.float32)
    mean = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    spec_mag = (spec_mag - mean) / (std + 1e-5)
    spec_mag = spec_mag[np.newaxis, np.newaxis, :]
    return spec_mag


# 获取录音数据
def record_audio():
    print("开始录音......")

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("录音已结束!")

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return WAVE_OUTPUT_FILENAME


# 预测
def infer(audio_path):
    data = load_data(audio_path)
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    output = model(data)
    result = torch.nn.functional.softmax(output)
    result = result.data.cpu().numpy()
    print(result)
    # 显示图片并输出结果最大的label
    lab = np.argsort(result)[0][-1]
    return lab


if __name__ == '__main__':
    try:
        while True:
            try:
                # 加载数据
                audio_path = record_audio()
                # 获取预测结果
                label = infer(audio_path)
                print('预测的标签为：%d' % label)
            except:
                pass
    except Exception as e:
        print(e)
        stream.stop_stream()
        stream.close()
        p.terminate()
```
