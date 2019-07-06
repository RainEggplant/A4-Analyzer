# %%
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import librosa
import librosa.display
import colorama
from colorama import Fore, Back, Style


def main():
    colorama.init()
    parser = argparse.ArgumentParser(
        description='''This tool can analyze audio files and estimate the frequecy of A4.''')
    parser.add_argument("filename")
    parser.add_argument('-s', '--silent', action="store_true", dest='silent',
                        help='process the given audio file silently')
    parser.add_argument('-o', '--offset', dest='offset', type=float, default=0,
                        help='the offset of the audio to process')
    parser.add_argument('-d', '--duration', dest='duration', type=float,
                        help='the duration of the audio to process')

    args = parser.parse_args()
    if (args.silent):
        auto_process(args.filename, args.offset, args.duration)
    else:
        print(Fore.YELLOW + Back.RED + Style.BRIGHT +
              'Welcome to RainEggplant\'s concert pitch analyzer!\n' + Style.RESET_ALL)
        print(Fore.YELLOW + Style.BRIGHT +
              'Please follow the instructions to get the result:' + Style.RESET_ALL)
        print(Fore.YELLOW + Style.BRIGHT + '[1] ' + Style.RESET_ALL +
              'We are going to generate a spectrogram with pitch lines.\n' +
              'After the window pops up, please come back to watch the instructions.')
        input("Press Enter to continue...")
        print('This may take serveral seconds, please wait.\n')
        show_spectrogram(args.filename, args.offset, args.duration)

        print(Fore.YELLOW + Style.BRIGHT + '[2] ' + Style.RESET_ALL +
              'Now you have seen the spectrogram.\n' +
              '- The green lines are peak frequency of that location.\n' +
              '- The white vertical lines divide the spectrogram into serveral fragments, according to pitch and volume changes.\n' +
              '  They are labeled with index. If they are overlapped, you can zoom in to see clearly.'
              '- You can use the tools in the tool bar to zoom, drag, etc.\n' +
              '- The time, frequency and note(relative to A4=440Hz) which you are pointing at will be shown in the status bar.\n')

        print(Fore.YELLOW + Style.BRIGHT + '[3] ' + Style.RESET_ALL +
              'Now you need to select the fragments used for analyzing.')

        print("This")
        input()


def show_spectrogram(filename, offset, duration):
    # %% Configuration
    sr = 22050
    n_fft = 8192
    win_len = 4096
    hop_len = 256
    min_freq = 128
    max_freq = 1024
    min_pitch_frame = 3
    pitch_threshold = 0.2

    # %% Detect onset frames
    y, t_sr = librosa.load(
        filename, sr=None, mono=True, offset=offset, duration=duration)
    onset_frames = librosa.onset.onset_detect(y=y, sr=t_sr, hop_length=hop_len)
    onset_time = librosa.frames_to_time(
        onset_frames, sr=t_sr, hop_length=hop_len)

    # %% STFT
    y, sr = librosa.load(
        filename, sr=sr, mono=True, offset=offset, duration=duration)
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
    fig = plt.figure(figsize=(14, 10))

    # %% Add axes for onset text
    rect_onset = [0.052, 0.96, 0.88, 0.01]
    rect_freq = [0.05, 0.05, 0.88, 0.9]

    # Remove boarder and ticks of the onset axes
    ax_onset_t = fig.add_axes(rect_onset)
    ax_onset_t.spines['top'].set_visible(False)
    ax_onset_t.spines['right'].set_visible(False)
    ax_onset_t.spines['bottom'].set_visible(False)
    ax_onset_t.spines['left'].set_visible(False)
    ax_onset_t.set_xticks([])
    ax_onset_t.set_yticks([])

    # Configure frequency-time figure
    ax_freq_t = fig.add_axes(rect_freq, sharex=ax_onset_t)
    ax_freq_t.set_xlabel('Time')
    ax_freq_t.set_ylabel('Hz')
    ax_freq_t.set_yscale('log', basey=2)
    ax_freq_t.set_ylim([min_freq, max_freq])
    ax_freq_t.yaxis.set_major_formatter(librosa.display.LogHzFormatter())

    # %% Plot frequency-time figure
    cmap = plt.cm.get_cmap('inferno')
    ax_freq_t.pcolormesh(
        bin_xpos, bin_ypos, Ydb[min_bin: max_bin, :],  cmap=cmap)
    # plt.colorbar(format='%+2.0f dB', pad=0.1)

    # %% Plot onset time
    count = 0
    for i in onset_time:
        ax_freq_t.vlines(i, 0, sr / 2, colors='w', linestyles='solid')
        ax_onset_t.text(i, 0, count)
        count += 1

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
    plt.ion()
    plt.show()
    return onset_frames, line_frames, line_freq


def auto_process(filename, offset, duration):
    # %%
    y, sr = librosa.load(
        filename, mono=True, offset=offset, duration=duration)
    pitches, magnitudes = librosa.piptrack(y, sr)
    # Select out pitches with high energy
    # pitches = pitches[magnitudes > np.median(magnitudes)]
    # %%
    offset_to_a4 = librosa.pitch_tuning(pitches)
    # %%
    a4 = 440 * (2 ** (offset_to_a4 / 12))
    # %%
    print('Estimated frequency of A4 is {:.1f}'.format(a4))


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
