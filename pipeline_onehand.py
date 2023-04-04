
import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream, duration
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


def feature_extraction(folders):
    all_notes = []
    all_durations = []
    for folder in folders:
        for file in glob.glob("Music/"+ folder + "/*.mid"):
            notes = []
            durations = []
            print("Extracting:", file)
            midi = converter.parse(file)
            notes_to_parse = instrument.partitionByInstrument(midi).flat

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                    durations.append(element.duration.quarterLength)
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))
                    durations.append(element.duration.quarterLength)
            all_notes.append(notes)
            all_durations.append(durations)
    
    return all_notes, all_durations

def preprocess_data(all_notes, all_durations, lookback=100):
    pitchnames = sorted(set(item for notes in all_notes for item in notes))
    note_to_int = {note:number for number, note in enumerate(pitchnames)}
    duration_names = sorted(set(item for durations in all_durations for item in durations))
    duration_to_int = {dur:number for number, dur in enumerate(duration_names)}

    in_notes = []
    in_durations = []
    out_notes = []
    out_durations = []
    for notes, durations in zip(all_notes, all_durations):
        for i in range(len(notes) - lookback):
            sequence_in_notes = notes[i:i + lookback]
            sequence_in_durations = durations[i:i + lookback]
            sequence_out_notes = notes[i + lookback]
            sequence_out_durations = durations[i + lookback]

            in_notes.append([note_to_int[char] for char in sequence_in_notes])
            in_durations.append([duration_to_int[char] for char in sequence_in_durations])
            out_notes.append(note_to_int[sequence_out_notes])
            out_durations.append(duration_to_int[sequence_out_durations])

    in_notes = np.reshape(in_notes, (len(in_notes), lookback, 1))
    in_durations = np.reshape(in_durations, (len(in_durations), lookback, 1))

    out_notes = to_categorical(out_notes)
    out_durations = to_categorical(out_durations)

    return in_notes, in_durations, out_notes, out_durations, pitchnames, duration_names

def build_model(lookback, output_size_notes, output_size_durations):

    in_notes = Input(shape=(lookback, 1))  # Adjusted input shape for notes
    in_durations = Input(shape=(lookback, 1))  # Adjusted input shape for durations
    
    # Process notes
    lstm_notes_1 = LSTM(256, return_sequences=True)(in_notes)
    dropout_notes_1 = Dropout(0.3)(lstm_notes_1)
    lstm_notes_2 = LSTM(256)(dropout_notes_1)
    
    # Process durations
    lstm_durations_1 = LSTM(256, return_sequences=True)(in_durations)
    dropout_durations_1 = Dropout(0.3)(lstm_durations_1)
    lstm_durations_2 = LSTM(256)(dropout_durations_1)
    
    # Fusion layer
    fusion = Concatenate()([lstm_notes_2, lstm_durations_2])
    dense_fusion = Dense(256)(fusion)
    dropout_fusion = Dropout(0.3)(dense_fusion)

    # Output layers for notes and durations
    out_notes = Dense(output_size_notes, activation="softmax", name="notes")(dropout_fusion)
    out_durations = Dense(output_size_durations, activation="softmax", name="durations")(dropout_fusion)
    
    model = Model(inputs=[in_notes, in_durations], outputs=[out_notes, out_durations])
    model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"],
                  optimizer='adam', # 'rmsprop',
                  metrics=["accuracy"])
    print(model.summary())
    return model

def train_model(model, in_notes, in_durations, out_notes, out_durations, epochs, batch_size, fname='model'):
    my_callbacks = [
        ModelCheckpoint(filepath=f'Models/{fname}.best.h5',
                        save_best_only=True,
                        monitor='notes_accuracy',
                        mode='max'),
        TensorBoard(log_dir=f'./Logs/{fname}/')
    ]
    history = model.fit([in_notes, in_durations], [out_notes, out_durations],
                        epochs=epochs, batch_size=batch_size, callbacks=my_callbacks,
                        verbose=2)
    model.save("Models/" + fname + ".h5")
    json.dump(history.history, open(f"./History/history_{fname}", 'w'))
    return history

def plot_history(history, plot_fname='training_curves'):
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].plot(history.history['notes_accuracy'])
    ax[0].plot(history.history['durations_accuracy'])
    ax[0].set_title('Accuracy Curves')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['notes_accuracy', 'durations_accuracy'], loc='lower right')
    ax[1].plot(history.history['notes_loss'])
    ax[1].plot(history.history['durations_loss'])
    ax[1].set_title('Loss Curves')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['notes_loss', 'durations_loss'], loc='upper right')
    plt.savefig("Figures/" + plot_fname + ".png")
    plt.show()

def load_music_model(model_fname='model'):
    try:
        return load_model("Models/" + model_fname + ".h5")
    except OSError:
        print(f'Error: could not load "Models/{model_fname}.h5')
        exit()
    
def generate_music(model, network_input_notes, network_input_durations, pitchnames, duration_names, num_notes):
    start = np.random.randint(0, len(network_input_notes) - 1)
    int_to_note = {number:note for number, note in enumerate(pitchnames)}
    int_to_duration = {number:dur for number, dur in enumerate(duration_names)}

    pattern_notes = network_input_notes[start]
    pattern_durations = network_input_durations[start]
    prediction_output = []

    for _ in range(num_notes):
        prediction_input_notes = np.reshape(pattern_notes, (1, len(pattern_notes), 1))
        prediction_input_durations = np.reshape(pattern_durations, (1, len(pattern_durations), 1))

        note_prediction, duration_prediction = model.predict([prediction_input_notes, prediction_input_durations], verbose=0)
        index_notes = np.argmax(note_prediction)
        index_durations = np.argmax(duration_prediction)
        note_result = int_to_note[index_notes]
        duration_result = int_to_duration[index_durations]
        prediction_output.append((note_result, duration_result))

        pattern_notes = np.append(pattern_notes, index_notes)
        pattern_notes = pattern_notes[1:]
        pattern_durations = np.append(pattern_durations, index_durations)
        pattern_durations = pattern_durations[1:]

    return prediction_output

def create_midi(prediction_output, out_fname="music"):
    offset = 0
    output_notes = []

    for pattern, dur in prediction_output:
        if ("." in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split(".")
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.duration = duration.Duration(dur)
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.duration = duration.Duration(dur)
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += dur

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp="Generated/" + out_fname + ".mid")

def play_music(in_fname):
        # mixer config
        freq = 44100  # audio CD quality
        bitsize = -16   # unsigned 16 bit
        channels = 2  # 1 is mono, 2 is stereo
        buffer = 1024   # number of samples
        pygame.mixer.init(freq, bitsize, channels, buffer)
        clock = pygame.time.Clock()
        try:
            pygame.mixer.music.load("Generated/" + in_fname + '.mid')
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

def music_generation_pipeline(lookback=128, epochs=75, batch_size=64, num_notes=250):
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
        model = build_model(lookback, out_notes.shape[1], out_durations.shape[1])
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
