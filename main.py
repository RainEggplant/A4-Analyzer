# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import librosa
import librosa.display


def main():
    # %% Configuration
    # audio_path = './audio/1_C3-3_E3.wav'
    audio_path = './audio/test.wav'
    sr = 22050
    offset = 0.2
    duration = 4
    n_fft = 8192
    win_len = 4096
    hop_len = 256
    min_freq = 128
    max_freq = 1024
    min_pitch_frame = 3
    pitch_threshold = 0.2

    # %% Detect onset frames
    y, t_sr = librosa.load(
        audio_path, sr=None, mono=True, offset=offset, duration=duration)
    onset_frames = librosa.onset.onset_detect(y=y, sr=t_sr, hop_length=hop_len)
    onset_time = librosa.frames_to_time(
        onset_frames, sr=t_sr, hop_length=hop_len)

    # %% STFT
    y, sr = librosa.load(
        audio_path, sr=sr, mono=True, offset=offset, duration=duration)
    Y = librosa.stft(y, n_fft=n_fft, win_length=win_len, hop_length=hop_len)
    Ydb = librosa.amplitude_to_db(abs(Y), ref=np.max)

    # %% Set parameters for spectrogram
    bin_range = sr / n_fft
    min_bin = int(min_freq / bin_range + 0.5)
    max_bin = int(max_freq / bin_range + 0.5)
    bin_ypos = np.arange((min_bin - 0.5) * bin_range,
                         (max_bin + 0.5) * bin_range, bin_range)
    bin_xpos = librosa.frames_to_time(
        range(0, Y.shape[1] + 1), sr=sr, hop_length=hop_len)

    # %% Create a spectrogram
    plt.figure(figsize=(14, 10))

    # %% Configure frequency-time figure
    ax_freq_t = plt.gca()
    ax_freq_t.set_xlabel('Time')
    ax_freq_t.set_ylabel('Hz')
    ax_freq_t.set_yscale('log', basey=2)
    ax_freq_t.set_ylim([min_freq, max_freq])
    ax_freq_t.yaxis.set_major_formatter(librosa.display.LogHzFormatter())

    # %% Plot frequency-time figure
    cmap = plt.cm.get_cmap('inferno')
    plt.pcolormesh(
        bin_xpos, bin_ypos, Ydb[min_bin: max_bin, :],  cmap=cmap)
    plt.colorbar(format='%+2.0f dB', pad=0.1)

    # %% Plot onset time
    for i in onset_time:
        plt.vlines(i, 0, sr / 2, colors='w', linestyles='solid')

    # %% Detect pitches
    pitches, magnitudes = librosa.piptrack(
        S=Y, sr=sr, hop_length=hop_len, threshold=pitch_threshold)
    pitches = pitches[min_bin: max_bin, :]

    # %% Pick out and plot pitches
    tunes = []
    for row in range(0, pitches.shape[0]):
        line_frames = []
        line_freq = []
        for col in range(0, pitches.shape[1]):
            if(pitches[row, col] != 0):
                line_frames.append(col)
                line_freq.append(pitches[row, col])
            else:
                if (len(line_frames) > 0):
                    if (len(line_frames) >= min_pitch_frame):
                        line_time = librosa.frames_to_time(
                            line_frames, sr=sr, hop_length=hop_len)
                        plt.plot(line_time, line_freq, color='g',
                                 linestyle='--', linewidth=1)
                        if (line_freq[0] < 512):
                            tunes.append(
                                (line_time.tolist(), line_freq.copy()))
                    line_frames.clear()
                    line_freq.clear()

    # %% Configure note-time axis
    ax_note_t = ax_freq_t.twinx()
    ax_note_t.set_ylabel('Note (relative to A4=440)')
    ax_note_t.set_yscale('log', basey=2)
    ax_note_t.set_ylim([min_freq, max_freq])
    ax_note_t.yaxis.set_major_locator(
        matplotlib.ticker.LogLocator(base=2, subs=[2 ** i for i in np.arange(0, 1, 1/12)]))
    ax_note_t.yaxis.set_major_formatter(librosa.display.NoteFormatter())
    ax_note_t.format_coord = make_format(ax_note_t, ax_freq_t)

    # %% Show the spectrogram
    plt.show()


def make_format(current, other):
    # current and other are axes
    def format_coord(x, y):
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x, y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        ax_coord = inv.transform(display_coord)
        coords = [ax_coord, (x, y)]
        return ('Time: {:.3f}, Frequency: {:.2f}, Note: {}.'.format(ax_coord[0], ax_coord[1], librosa.hz_to_note(ax_coord[1], cents=True)))
        # return ('Left: {:<40}    Right: {:<}'
        #        .format(*['({:.3f}, {:.3f})'.format(x, y) for x, y in coords]))
    return format_coord


if __name__ == "__main__":
    main()
