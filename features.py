import librosa, warnings, os
import numpy as np

warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import librosa.display

path = your audio path
for item in os.listdir(path):
    audio_file = path+item
    y, sr = librosa.load(audio_file, sr=None)

    os.system('mkdir -p zxx-image-features/'+item)

    envelope = librosa.feature.rms(y=y, hop_length=512)

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    stft = librosa.stft(y)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    plt.figure()
    plt.plot(envelope[0])
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('zxx-image-features/'+item+'/'+'envelope.png', bbox_inches='tight', pad_inches=0,transparent=True)

    plt.figure()
    librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('zxx-image-features/'+item+'/'+'mel_spectrogram.png', bbox_inches='tight', pad_inches=0,transparent=True)

    plt.figure()
    librosa.display.specshow(spectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='linear')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('zxx-image-features/'+item+'/'+'spectrogram.png', bbox_inches='tight', pad_inches=0,transparent=True)

    plt.close('all')
