import librosa
import numpy as np


def vad(wav, top_db=20, overlap=200):
    # Split an audio signal into non-silent intervals
    intervals = librosa.effects.split(wav, top_db=top_db)
    if len(intervals) == 0:
        return wav
    wav_output = [np.array([])]
    for sliced in intervals:
        seg = wav[sliced[0]:sliced[1]]
        if len(seg) < 2 * overlap:
            wav_output[-1] = np.concatenate((wav_output[-1], seg))
        else:
            wav_output.append(seg)
    wav_output = [x for x in wav_output if len(x) > 0]

    if len(wav_output) == 1:
        wav_output = wav_output[0]
    else:
        wav_output = concatenate(wav_output)
    return wav_output


def concatenate(wave, overlap=200):
    total_len = sum([len(x) for x in wave])
    unfolded = np.zeros(total_len)

    # Equal power crossfade
    window = np.hanning(2 * overlap)
    fade_in = window[:overlap]
    fade_out = window[-overlap:]

    end = total_len
    for i in range(1, len(wave)):
        prev = wave[i - 1]
        curr = wave[i]

        if i == 1:
            end = len(prev)
            unfolded[:end] += prev

        max_idx = 0
        max_corr = 0
        pattern = prev[-overlap:]
        # slide the curr batch to match with the pattern of previous one
        for j in range(overlap):
            match = curr[j:j + overlap]
            corr = np.sum(pattern * match) / [(np.sqrt(np.sum(pattern ** 2)) * np.sqrt(np.sum(match ** 2))) + 1e-8]
            if corr > max_corr:
                max_idx = j
                max_corr = corr

        # Apply the gain to the overlap samples
        start = end - overlap
        unfolded[start:end] *= fade_out
        end = start + (len(curr) - max_idx)
        curr[max_idx:max_idx + overlap] *= fade_in
        unfolded[start:end] += curr[max_idx:]
    return unfolded[:end]


def cmvn_floating_kaldi(x, LC, RC, norm_vars=True):
    """Mean and variance normalization over a floating window.
    x is the feature matrix (nframes x dim)
    LC, RC are the number of frames to the left and right defining the floating
    window around the current frame. This function uses Kaldi-like treatment of
    the initial and final frames: Floating windows stay of the same size and
    for the initial and final frames are not centered around the current frame
    but shifted to fit in at the beginning or the end of the feature segment.
    Global normalization is used if nframes is less than LC+RC+1.
    """
    N, dim = x.shape
    win_len = min(len(x), LC + RC + 1)
    win_start = np.maximum(np.minimum(np.arange(-LC, N - LC), N - win_len), 0)
    f = np.r_[np.zeros((1, dim)), np.cumsum(x, 0)]
    x = x - (f[win_start + win_len] - f[win_start]) / win_len
    if norm_vars:
        f = np.r_[np.zeros((1, dim)), np.cumsum(x ** 2, 0)]
        x /= np.sqrt((f[win_start + win_len] - f[win_start]) / win_len)
    return x


# 将音频流转换为numpy
def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)
