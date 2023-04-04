
import glob
import numpy as np
import mido
from music21 import converter, instrument, note, chord, stream, duration
from mido import MidiFile
from keras.models import Model, load_model
from keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from matplotlib import pyplot as plt
import argparse
import json
import contextlib
from fractions import Fraction
import os
with contextlib.redirect_stdout(None):
    import pygame
            

def get_notes(fname):
    if '.mid' not in fname:
        fname += '.mid'
    midi_file = MidiFile(fname)
    print(fname)
    # Initialize a dictionary to store note states
    note_states = {}
    notes = {'Right Hand':[], 'Left Hand':[]}
    min_duration = 9999
    # Extract notes for all channels
    for i, track in enumerate(midi_file.tracks):
        if track.name == 'Right Hand' or track.name == 'Left Hand':
            curr_track = track.name
        elif len(midi_file.tracks) == 3:
            if i==0:
                continue
            elif i == 1:
                curr_track = 'Right Hand'
            elif i == 2:
                curr_track = 'Left Hand'
        else:
            continue
        
        current_time = 0
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_states[msg.channel, msg.note] = current_time
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                start_time = note_states.get((msg.channel, msg.note))
                if start_time is not None:
                    note_duration = (current_time - start_time)
                    if note_duration < min_duration:
                        min_duration = note_duration
                    note_info = [msg.channel, msg.note, start_time, note_duration]
                    notes[curr_track].append(note_info)
                    del note_states[msg.channel, msg.note]

    # Sort notes by start time
    notes['Right Hand'].sort(key=lambda x: x[2])
    notes['Left Hand'].sort(key=lambda x: x[2])
    min_duration *= 4
    for n in notes['Right Hand']:
        n[3] /= min_duration
    for n in notes['Left Hand']:
        n[3] /= min_duration
    if all(n for n in notes.values()):
        return notes
    return None

def get_all_notes(pattern='Music/bach/aof/*.mid'):
    # notes to nx88 categorical
    all_right_notes = []
    all_right_durs = []
    all_left_notes = []
    all_left_durs = []
    for file in glob.glob(pattern):
        notes = get_notes(file)
        if notes:
            r = 0
            l = 0
            N_r = len(notes['Right Hand'])
            N_l = len(notes['Left Hand'])
            right_notes = []
            right_durs = []
            left_notes = []
            left_durs = []
            while(r < N_r and l < N_l):
                curr_r = notes['Right Hand'][r]
                curr_l = notes['Left Hand'][l]
                # If start times equal
                if curr_l[2] == curr_r[2]:
                    r += 1
                    l += 1
                    right_notes.append(curr_r[1])
                    right_durs.append(curr_r[3])
                    left_notes.append(curr_l[1])
                    left_durs.append(curr_l[3])   
                elif curr_l[2] > curr_r[2]:
                    r += 1
                    right_notes.append(curr_r[1])
                    right_durs.append(curr_r[3])
                    left_notes.append(0)
                    left_durs.append(0)
                else:
                    l += 1
                    right_notes.append(0)
                    right_durs.append(0)
                    left_notes.append(curr_l[1])
                    left_durs.append(curr_l[3])

    
            all_right_notes.append(right_notes)
            all_right_durs.append(right_durs)
            all_left_notes.append(left_notes)
            all_left_durs.append(left_durs)
    return all_right_notes, all_right_durs, all_left_notes, all_left_durs

def preprocess_data(all_right_notes, all_right_durs, all_left_notes, all_left_durs, lookback=100):
    dur_names = sorted(set(item for durations in all_left_durs+all_right_durs for item in durations))
    dur_to_int = {dur:number for number, dur in enumerate(dur_names)}

    in_right_notes = []
    in_right_durs = []
    in_left_notes = []
    in_left_durs = []
    out_right_notes = []
    out_right_durs = []
    out_left_notes = []
    out_left_durs = []
    for r_notes, r_durs, l_notes, l_durs in zip(all_right_notes, all_right_durs, all_left_notes, all_left_durs):
        for i in range(len(r_notes) - lookback):
            r_in_notes = r_notes[i:i + lookback]
            r_in_durs = r_durs[i:i + lookback]
            r_out_note = r_notes[i + lookback]
            r_out_dur = r_durs[i + lookback]
            l_in_notes = l_notes[i:i + lookback]
            l_in_durs = l_durs[i:i + lookback]
            l_out_note = l_notes[i + lookback]
            l_out_dur = l_durs[i + lookback]

            in_right_notes.append(r_in_notes)
            in_right_durs.append([dur_to_int[dur] for dur in r_in_durs])
            out_right_notes.append(r_out_note)
            out_right_durs.append(dur_to_int[r_out_dur])

            in_left_notes.append(l_in_notes)
            in_left_durs.append([dur_to_int[dur] for dur in l_in_durs])
            out_left_notes.append(l_out_note)
            out_left_durs.append(dur_to_int[l_out_dur])

    in_right_notes = np.reshape(in_right_notes, (len(in_right_notes), lookback, 1))
    in_right_durs = np.reshape(in_right_durs, (len(in_right_durs), lookback, 1))
    in_left_notes = np.reshape(in_left_notes, (len(in_left_notes), lookback, 1))
    in_left_durs = np.reshape(in_left_durs, (len(in_left_durs), lookback, 1))

    # 89 notes: 88 + 1 for no note played
    out_right_notes = to_categorical(out_right_notes, num_classes=89)
    out_right_durs = to_categorical(out_right_durs)
    out_left_notes = to_categorical(out_left_notes, num_classes=89)
    out_left_durs = to_categorical(out_left_durs)

    return in_right_notes, in_right_durs, in_left_notes, in_left_durs, out_right_notes, out_right_durs, out_left_notes, out_left_durs, dur_names

def build_model(lookback, output_size_durations):
    in_right_notes = Input(shape=(lookback, 1))  # Adjusted input shape for notes
    in_right_durs = Input(shape=(lookback, 1))  # Adjusted input shape for durations
    in_left_notes = Input(shape=(lookback, 1))  # Adjusted input shape for notes
    in_left_durs = Input(shape=(lookback, 1))  # Adjusted input shape for durations
    
    # Process right notes
    lstm_right_notes_1 = LSTM(256, return_sequences=True)(in_right_notes)
    dropout_right_notes_1 = Dropout(0.3)(lstm_right_notes_1)
    lstm_right_notes_2 = LSTM(256)(dropout_right_notes_1)

    # Process left notes
    lstm_left_notes_1 = LSTM(256, return_sequences=True)(in_left_notes)
    dropout_left_notes_1 = Dropout(0.3)(lstm_left_notes_1)
    lstm_left_notes_2 = LSTM(256)(dropout_left_notes_1)
    
    # Process right durations
    lstm_right_durs_1 = LSTM(256, return_sequences=True)(in_right_durs)
    dropout_right_durs_1 = Dropout(0.3)(lstm_right_durs_1)
    lstm_right_durs_2 = LSTM(256)(dropout_right_durs_1)

    # Process left durations
    lstm_left_durs_1 = LSTM(256, return_sequences=True)(in_left_durs)
    dropout_left_durs_1 = Dropout(0.3)(lstm_left_durs_1)
    lstm_left_durs_2 = LSTM(256)(dropout_left_durs_1)
    
    # Fusion layer
    fusion = Concatenate()([lstm_right_notes_2, lstm_left_notes_2, lstm_right_durs_2, lstm_left_durs_2])
    dense_fusion = Dense(512)(fusion)
    dropout_fusion = Dropout(0.3)(dense_fusion)

    # Output layers for notes and durations
    # 89 notes: 88 + 1 for no note played
    out_right_notes = Dense(89, activation="softmax", name="right_notes")(dropout_fusion)
    out_right_durs = Dense(output_size_durations, activation="softmax", name="right_durs")(dropout_fusion)
    out_left_notes = Dense(89, activation="softmax", name="left_notes")(dropout_fusion)
    out_left_durs = Dense(output_size_durations, activation="softmax", name="left_durs")(dropout_fusion)
    
    model = Model(inputs=[in_right_notes, in_right_durs, in_left_notes, in_left_durs],
                  outputs=[out_right_notes, out_right_durs, out_left_notes, out_left_durs])
    model.compile(loss=["categorical_crossentropy", "categorical_crossentropy",
                        "categorical_crossentropy", "categorical_crossentropy"],
                  optimizer='adam', # 'rmsprop',
                  metrics=["accuracy"])
    print(model.summary())
    return model

def train_model(model, in_right_notes, in_right_durs, in_left_notes, in_left_durs, out_right_notes, out_right_durs, out_left_notes, out_left_durs, epochs, batch_size, fname='model'):
    callbacks = [
        ModelCheckpoint(filepath=f'Models/{fname}.best.h5',
                        save_best_only=True,
                        monitor='loss',
                        save_freq="epoch",
                        mode='min'),
        TensorBoard(log_dir=f'./Logs/{fname}/')
    ]
    history = model.fit([in_right_notes, in_right_durs, in_left_notes, in_left_durs],
                        [out_right_notes, out_right_durs, out_left_notes, out_left_durs],
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
    model.save("Models/" + fname + ".h5")
    json.dump(history.history, open(f"./History/history_{fname}", 'w'))
    return history

def load_music_model(fname='model'):
    try:
        return load_model("Models/" + fname + ".h5")
    except OSError:
        print(f'Error: could not load "Models/{fname}.h5')
        exit()

def load_history(fname):
    if 'history' not in fname:
        fname = 'history_' + fname
    with open(f'History/{fname}', 'r') as f:
        history = json.load(f)

    return history

def plot_history(history, fname):
    fig, ax = plt.subplots(2, 2, figsize=(10,5))
    if isinstance(history, dict):
        ax[0,0].plot(history['right_notes_accuracy'])
        ax[0,0].plot(history['left_notes_accuracy'])
        ax[0,1].plot(history['right_durs_accuracy'])
        ax[0,1].plot(history['left_durs_accuracy'])
        ax[1,0].plot(history['right_notes_loss'])
        ax[1,0].plot(history['left_notes_loss'])
        ax[1,1].plot(history['right_durs_loss'])
        ax[1,1].plot(history['left_durs_loss'])
        ax[0,0].set_title('Notes Accuracy')
        ax[0,0].set_xlabel('epoch')
        ax[0,1].set_title('Duration Accuracy')
        ax[0,1].set_xlabel('epoch')
        ax[1,0].set_title('Notes Loss')
        ax[1,0].set_xlabel('epoch')
        ax[1,1].set_title('Duration Loss')
        ax[1,1].set_xlabel('epoch')
        ax[0,0].legend(['Right Hand', 'Left Hand'], loc='lower right')
        ax[0,1].legend(['Right Hand', 'Left Hand'], loc='lower right')
        ax[1,0].legend(['Right Hand', 'Left Hand'], loc='upper right')
        ax[1,1].legend(['Right Hand', 'Left Hand'], loc='upper right')
    elif isinstance(history.history, dict):
        ax[0,0].plot(history.history['right_notes_accuracy'])
        ax[0,0].plot(history.history['left_notes_accuracy'])
        ax[0,1].plot(history.history['right_durs_accuracy'])
        ax[0,1].plot(history.history['left_durs_accuracy'])
        ax[1,0].plot(history.history['right_notes_loss'])
        ax[1,0].plot(history.history['left_notes_loss'])
        ax[1,1].plot(history.history['right_durs_loss'])
        ax[1,1].plot(history.history['left_durs_loss'])
        ax[0,0].set_title('Notes Accuracy')
        ax[0,0].set_xlabel('epoch')
        ax[0,1].set_title('Duration Accuracy')
        ax[0,1].set_xlabel('epoch')
        ax[0,0].set_title('Notes Loss')
        ax[1,0].set_xlabel('epoch')
        ax[1,1].set_title('Duration Loss')
        ax[1,1].set_xlabel('epoch')
        ax[0,0].legend(['Right Hand', 'Left Hand'], loc='lower right')
        ax[0,1].legend(['Right Hand', 'Left Hand'], loc='lower right')
        ax[1,0].legend(['Right Hand', 'Left Hand'], loc='upper right')
        ax[1,1].legend(['Right Hand', 'Left Hand'], loc='upper right')
    else:
        return
    plt.tight_layout()
    plt.savefig("Figures/twohand_" + fname + ".png")
    plt.show()


def generate_music(model, in_right_notes, in_right_durs, in_left_notes, in_left_durs, dur_names, num_notes):
    start = np.random.randint(0, len(in_right_notes) - 1)
    int_to_duration = {number: dur for number, dur in enumerate(dur_names)}

    pattern_right_notes = in_right_notes[start]
    pattern_right_durations = in_right_durs[start]
    pattern_left_notes = in_left_notes[start]
    pattern_left_durations = in_left_durs[start]
    
    prediction_output = []

    for _ in range(num_notes):
        prediction_input_right_notes = np.reshape(pattern_right_notes, (1, len(pattern_right_notes), 1))
        prediction_input_right_durations = np.reshape(pattern_right_durations, (1, len(pattern_right_durations), 1))
        prediction_input_left_notes = np.reshape(pattern_left_notes, (1, len(pattern_left_notes), 1))
        prediction_input_left_durations = np.reshape(pattern_left_durations, (1, len(pattern_left_durations), 1))

        right_notes_pred, right_durs_pred, left_notes_pred, left_durs_pred = model.predict(
            [prediction_input_right_notes, prediction_input_right_durations, prediction_input_left_notes, prediction_input_left_durations], verbose=0)
        
        right_note_result = np.argmax(right_notes_pred)
        index_right_durations = np.argmax(right_durs_pred)
        left_note_result = np.argmax(left_notes_pred)
        index_left_durations = np.argmax(left_durs_pred)
        
        right_duration_result = int_to_duration[index_right_durations]
        left_duration_result = int_to_duration[index_left_durations]
        
        prediction_output.append(((right_note_result, right_duration_result), (left_note_result, left_duration_result)))

        pattern_right_notes = np.append(pattern_right_notes, right_note_result)
        pattern_right_notes = pattern_right_notes[1:]
        pattern_right_durations = np.append(pattern_right_durations, index_right_durations)
        pattern_right_durations = pattern_right_durations[1:]
        
        pattern_left_notes = np.append(pattern_left_notes, left_note_result)
        pattern_left_notes = pattern_left_notes[1:]
        pattern_left_durations = np.append(pattern_left_durations, index_left_durations)
        pattern_left_durations = pattern_left_durations[1:]

    return prediction_output

def create_midi(music, fname="twohand_music"):
    offset_right = 0
    offset_left = 0
    output_notes_right = []
    output_notes_left = []

    for (right_pattern, right_dur), (left_pattern, left_dur) in music:
        new_note = note.Note(right_pattern)
        new_note.offset = offset_right
        new_note.duration = duration.Duration(right_dur)
        new_note.storedInstrument = instrument.Piano()
        output_notes_right.append(new_note)

        new_note = note.Note(left_pattern)
        new_note.offset = offset_left
        new_note.duration = duration.Duration(left_dur)
        new_note.storedInstrument = instrument.Piano()
        output_notes_left.append(new_note)

        offset_right += right_dur
        offset_left += left_dur

    midi_stream_right = stream.Stream(output_notes_right)
    midi_stream_left = stream.Stream(output_notes_left)
    
    midi_stream = stream.Stream([midi_stream_right, midi_stream_left])
    midi_stream.write("midi", fp="Generated/" + fname + ".mid")


def play_music(fname):
        # mixer config
        freq = 44100  # audio CD quality
        bitsize = -16   # unsigned 16 bit
        channels = 2  # 1 is mono, 2 is stereo
        buffer = 1024   # number of samples
        pygame.mixer.init(freq, bitsize, channels, buffer)
        clock = pygame.time.Clock()
        try:
            pygame.mixer.music.load("Generated/" + fname + '.mid')
        except:
            return
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            clock.tick(30) # check if playback has finished

def save_data(fname, in_notes, in_durations, out_notes, out_durations, pitchnames, duration_names):
    if not os.path.exists(f'Data/{fname}'):
        os.mkdir(f'Data/{fname}')
    np.save(f'Data/{fname}/in_notes.npy', in_notes)
    np.save(f'Data/{fname}/in_durations.npy', in_durations)
    np.save(f'Data/{fname}/out_notes.npy', out_notes)
    np.save(f'Data/{fname}/out_durations.npy', out_durations)

    with open(f'Data/{fname}/pitchnames', 'w') as f:
        s = ""
        for p in pitchnames:
            s += p + " "

        f.write(s[:-1])

    with open(f'Data/{fname}/duration_names', 'w') as f:
        s = ""
        for d in duration_names:
            s += str(d) + " "

        f.write(s[:-1])

def load_data(fname):
    in_notes = np.load(f'Data/{fname}/in_notes.npy')
    in_durations = np.load(f'Data/{fname}/in_durations.npy')
    out_notes = np.load(f'Data/{fname}/out_notes.npy')
    out_durations = np.load(f'Data/{fname}/out_durations.npy')

    with open(f'Data/{fname}/pitchnames', 'r') as f:
        pitchnames = [x for x in f.read().split()]

    with open(f'Data/{fname}/duration_names', 'r') as f:
        duration_names = []
        for x in f.read().split():
            if '.' in x:
                duration_names.append(float(x))
            elif '/' in x:
                duration_names.append(Fraction(x))
            else:
                print("Error: duration_name string is unreadable:", x)
                exit()

    return in_notes, in_durations, out_notes, out_durations, pitchnames, duration_names

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m',  '--Music',    nargs = '*', help='input mid foldername(s)')
    argparser.add_argument('-t',  '--Train',    action='store_true', help='input whether to train model')
    argparser.add_argument('-n',  '--Name',     help='input name for model, figure, generated music')
    argparser.add_argument('-e',  '--Epochs',   help='input name for model, figure, generated music')
    argparser.add_argument('-s',  '--SaveData', action='store_true', help='Save preprocessed data')
    argparser.add_argument('-l',  '--LoadData', action='store_true', help='Load preprocessed data')
    argparser.add_argument('-u',  '--Use',      help='Use a specific model')
    args = argparser.parse_args()
    if not args.Music:
        raise ValueError("Music input music file or files.")
    
    if args.Name:
        fname = args.Name
    elif len(args.Music) == 1:
        fname = args.Music[0].replace('/','_')
    else:
        fname = ".".join(args.Music).replace('/','_')

    print(f'\t--Music    : {", ".join(args.Music)}')
    print(f'\t--Train    : {args.Train}')
    print(f'\t--Name     : {args.Name}')
    print(f'\t--Epochs   : {args.Epochs}')
    print(f'\t--SaveData : {args.SaveData}')
    print(f'\t--LoadData : {args.LoadData}')
    print(f'\t--Use      : {args.Use}')
    print()
    return args, fname

def music_generation_pipeline(lookback=128, epochs=400, batch_size=32, num_notes=100):
    fname = 'twohand_test'
    epochs = 40

    all_right_notes, all_right_durs, all_left_notes, all_left_durs = get_all_notes()
    in_right_notes, in_right_durs, in_left_notes, in_left_durs, out_right_notes, out_right_durs, out_left_notes, out_left_durs, dur_names = preprocess_data(all_right_notes, all_right_durs, all_left_notes, all_left_durs, lookback=lookback)
    model = build_model(lookback=lookback, output_size_durations=len(dur_names))
    history = train_model(model, in_right_notes, in_right_durs, in_left_notes, in_left_durs, out_right_notes, out_right_durs, out_left_notes, out_left_durs, epochs=epochs, batch_size=batch_size, fname=fname)
    model = load_music_model(fname)
    history = load_history(fname)
    plot_history(history, fname)
    music = generate_music(model, in_right_notes, in_right_durs, in_left_notes, in_left_durs, dur_names, num_notes)
    create_midi(music, fname)
    play_music(fname)

    exit()
    print()
    args, fname = parse_args()
    if args.LoadData:
        in_notes, in_durations, out_notes, out_durations, pitchnames, duration_names = load_data(fname)
    else:
        all_notes, all_durations = feature_extraction(args.Music)
        print("\nTransforming data...")
        in_notes, in_durations, out_notes, out_durations, pitchnames, duration_names = preprocess_data(all_notes, all_durations, lookback)
    
    if args.SaveData:
        save_data(fname, in_notes, in_durations, out_notes, out_durations, pitchnames, duration_names)

    print("\nLoading data...\nInput Shapes: ", in_notes.shape, in_durations.shape)
    if args.Train:
        if args.Epochs:
            try:
                epochs = int(args.Epochs)
            except ValueError:
                pass
        print()
        model = build_model3(in_notes.shape, in_durations.shape, out_notes.shape[1], out_durations.shape[1])
        history = train_model(model, in_notes, in_durations, out_notes, out_durations, epochs, batch_size, fname)
        plot_history(history, fname)
        print("\nFinished training.")
    
    if args.Use:
        model = load_music_model(args.Use)
    else:
        model = load_music_model(fname)
    print("\nGenerating music.")
    generated_music = generate_music(model, in_notes, in_durations, pitchnames, duration_names, num_notes)
    print(f"\nSaving midi file at: 'Music/{fname}.mid'")
    create_midi(generated_music, fname)
    print("\nPlaying music.")
    play_music(fname)

if __name__=="__main__":
    music_generation_pipeline()

