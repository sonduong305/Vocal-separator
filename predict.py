import torch
import numpy as np
import librosa
from pydub import AudioSegment
import time
from youtube_download import get_audio
n_fft = 1024
def convert_stft(x):
    # Convert x data to stft 
    stft_label = librosa.stft(x, hop_length = 768, n_fft=n_fft )
    stft_label_mag = np.log1p(np.abs(stft_label))
    return torch.from_numpy(stft_label_mag)

def get_phase(x):
    stft_label = librosa.stft(x, hop_length = 768, n_fft=n_fft )
    return np.angle(stft_label)

kernel = np.zeros((4,128))
kernel = np.concatenate((kernel, np.ones((509, 128))), axis = 0)


def predict_part(x_part, model):
    # start = time.time()
    feature = convert_stft(x_part)
    phase = get_phase(x_part)
    # print(time.time() - start)
    # start = time.time()
    out_voice = model(feature[None,:,:].cuda())
    out_voice = out_voice.cpu().detach().numpy()[0]
    # print(time.time() - start)
    # start = time.time()
    # out_voice = cv2.dilate(out_voice,  closing)
    # out_voice = signal.fftconvolve(out_voice, closing, mode='same')
    feature = feature.numpy()
    out_voice *= 1.15
    mask = np.where(feature > 0 , out_voice / feature, 0)
    out_voice = feature * mask
    out_voice = np.where(out_voice > feature, feature, out_voice)
    
    # out_voice = out_voice * kernel
    out_voice  = np.exp(out_voice) - 1
    out_voice = out_voice * np.exp(1j*phase)
    y_voice = librosa.istft(out_voice , hop_length = 768)
    # print(time.time() - start)
    return y_voice

def predict_song(path, model):
    start = time.time()
    model.eval()
    
    win_len = 768*127
    # print(path.shape)
    # x, sr = librosa.load(path, sr = 44100 // 2, mono=True)
    x = path #librosa.resample(path, 44100, 22050)
    y_out = np.zeros((win_len,))
    y_out_2 = np.zeros((win_len // 4,))
    # Padding for x
    x_pad = np.pad(x, (0, win_len), mode = "constant")
    
    l = len(x)
    
    for i in range(win_len, l, win_len):
        x_part = x_pad[i:i + win_len]
        y_part = predict_part(x_part, model)

        y_out = np.concatenate((y_out, y_part), axis=0)

    # for i in range(win_len // 4, l - (win_len // 4) , win_len):
    #     x_part = x_pad[i:i + win_len]
    #     y_part = predict_part(x_part, model)

    #     y_out_2 = np.concatenate((y_out_2, y_part), axis=0)
    # y_out_2 = np.pad(y_out_2, (0, (win_len * 2) ), mode = "constant")[:len(y_out)]
    # y_out = np.where(y_out > y_out_2, y_out, y_out_2)

    y_out_2 = np.zeros((win_len // 2,))
    for i in range(win_len // 2, l - win_len // 2 , win_len):
        x_part = x_pad[i:i + win_len]
        y_part = predict_part(x_part, model)

        y_out_2 = np.concatenate((y_out_2, y_part), axis=0)
    y_out_2 = np.pad(y_out_2, (0, (win_len * 2)), mode = "constant")[:len(y_out)]

    # y_out = np.where(y_out > y_out_2, y_out, y_out_2)

    # y_out_2 = np.zeros(((win_len * 3) // 4,))
    # for i in range((win_len * 3) // 4, l - ((win_len * 3) // 4)  , win_len):
    #     x_part = x_pad[i:i + win_len]
    #     y_part = predict_part(x_part, model)

    #     y_out_2 = np.concatenate((y_out_2, y_part), axis=0)

    # y_out_2 = np.pad(y_out_2, (0, win_len * 2  ), mode = "constant")[:len(y_out)]
    # y_out = np.where(y_out < y_out_2, y_out, y_out_2)

    y_out = (y_out + y_out_2 ) / 2

    # print(y_out.shape)
    print("overall: " ,time.time() - start)
    return np.array(y_out)[:len(x)]

def to_int_wav(filename, arr, sr):
    song = np.array(arr * 2147483647 , dtype = np.int32)
    audio_segment = AudioSegment(song.tobytes(), frame_rate=sr,sample_width=4, channels=1)
    audio_segment.export(filename, format="mp3", bitrate='128k')

def predict_from_youtube(link, model):
    name = get_audio(link)
    song , sr  = librosa.load('temp.wav',sr = 44100)
    y = predict_song(song, model)
    beat = song - y
    return y, beat, name