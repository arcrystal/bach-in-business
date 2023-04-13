
import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream, duration, midi
from keras.models import Model, load_model
from keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from matplotlib import pyplot as plt
import argparse
import contextlib
from fractions import Fraction
import os
with contextlib.redirect_stdout(None):
    import pygame

def flatten_midi(file):
    midi = converter.parse(file)
    parts = instrument.partitionByInstrument(midi)
    try:
        for p in parts.recurse():
            if 'Instrument' in p.classes: # or 'Piano'
                p.activeSite.replace(p, instrument.Piano())
    except:
        for p in parts.parts:
            p.insert(0, instrument.Piano())

    return parts.flatten()

def extract(folders, flatten=True):
    all_notes = []
    all_durs = []
    for folder in folders:
        if '.mid' in folder:
            files = ["Music/"+ folder]
        else:
            files = glob.glob("Music/"+ folder + "/*.mid")
        for file in files:
            print("Extracting:", file)
            if flatten:
                notes_to_parse = flatten_midi(file)
            else:
                midi = converter.parse(file)
                parts = instrument.partitionByInstrument(midi)
                max_length = 0
                notes_to_parse = None
                for p in parts.parts:
                    if max_length < len(p):
                        max_length = len(p)
                        notes_to_parse = p.notes.stream()

            for one_key_notes in transpose_to_all_keys(notes_to_parse):
                notes = []
                durs = []
                for element in one_key_notes:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                        durs.append(element.duration.quarterLength)
                    elif isinstance(element, chord.Chord):
                        notes.append(".".join(str(n) for n in element.normalOrder))
                        durs.append(element.duration.quarterLength)

                all_notes.append(notes)
                all_durs.append(durs)

    return all_notes, all_durs

def transpose_to_all_keys(notes_to_parse):
    notes = [notes_to_parse]
    for halfSteps in range(1,12):
        one_key_notes = notes_to_parse.transpose(halfSteps)
        notes.append(one_key_notes)
    
    return notes

def transform(all_notes, all_durs, lookback=100):
    pitchnames = sorted(set(item for notes in all_notes for item in notes))
    note_to_int = {note:number for number, note in enumerate(pitchnames)}
    dur_names = sorted(set(item for durs in all_durs for item in durs))
    dur_to_int = {dur:number for number, dur in enumerate(dur_names)}

    in_notes = []
    in_durs = []
    out_notes = []
    out_durs = []
    for notes, durs in zip(all_notes, all_durs):
        for i in range(len(notes) - lookback):
            sequence_in_notes = notes[i:i + lookback]
            sequence_in_durs = durs[i:i + lookback]
            sequence_out_notes = notes[i + lookback]
            sequence_out_durs = durs[i + lookback]

            in_notes.append([note_to_int[char] for char in sequence_in_notes])
            in_durs.append([dur_to_int[char] for char in sequence_in_durs])
            out_notes.append(note_to_int[sequence_out_notes])
            out_durs.append(dur_to_int[sequence_out_durs])

    in_notes = np.reshape(in_notes, (len(in_notes), lookback, 1))
    in_durs = np.reshape(in_durs, (len(in_durs), lookback, 1))

    out_notes = to_categorical(out_notes)
    out_durs = to_categorical(out_durs)

    return in_notes, in_durs, out_notes, out_durs, pitchnames, dur_names

def build_model(lookback, output_size_notes, output_size_durs, n_units=512):

    in_notes = Input(shape=(lookback, 1))  # Adjusted input shape for notes
    in_durs = Input(shape=(lookback, 1))  # Adjusted input shape for durs
    
    # Process notes
    lstm_notes_1 = LSTM(n_units, return_sequences=True)(in_notes)
    dropout_notes_1 = Dropout(0.3)(lstm_notes_1)
    lstm_notes_2 = LSTM(n_units)(dropout_notes_1)
    
    # Process durs
    lstm_durs_1 = LSTM(n_units, return_sequences=True)(in_durs)
    dropout_durs_1 = Dropout(0.3)(lstm_durs_1)
    lstm_durs_2 = LSTM(n_units)(dropout_durs_1)
    
    # Fusion layer
    fusion = Concatenate()([lstm_notes_2, lstm_durs_2])
    dense_fusion = Dense(n_units)(fusion)
    dropout_fusion = Dropout(0.3)(dense_fusion)

    # Output layers for notes and durs
    out_notes = Dense(output_size_notes, activation="softmax", name="notes")(dropout_fusion)
    out_durs = Dense(output_size_durs, activation="softmax", name="durs")(dropout_fusion)
    
    model = Model(inputs=[in_notes, in_durs], outputs=[out_notes, out_durs])
    model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"],
                  optimizer='adam', # 'rmsprop',
                  metrics=["accuracy"],
                  loss_weights=[1, 0.05]) # 12 transposed keys have same durations (should be >=12x)
    print(model.summary())
    return model

def train_model(model, in_notes, in_durs, out_notes, out_durs,
                epochs, batch_size, fname='model', verbose=2):
    my_callbacks = [
        ModelCheckpoint(filepath=f'Models/{fname}.h5',
                        save_best_only=True,
                        monitor='loss',
                        mode='min'),
        TensorBoard(log_dir=f'./Logs/{fname}/')
    ]
    history = model.fit([in_notes, in_durs], [out_notes, out_durs],
                        epochs=epochs, batch_size=batch_size, callbacks=my_callbacks,
                        verbose=verbose)
    return history

def plot_history(history, plot_fname='training_curves'):
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].plot(history.history['notes_accuracy'])
    ax[0].plot(history.history['durs_accuracy'])
    ax[0].set_title('Accuracy Curves')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['notes_accuracy', 'durs_accuracy'], loc='lower right')
    ax[1].plot(history.history['notes_loss'])
    ax[1].plot(history.history['durs_loss'])
    ax[1].set_title('Loss Curves')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['notes_loss', 'durs_loss'], loc='upper right')
    plt.savefig("Figures/" + plot_fname + ".png")
    plt.show()

def load_music_model(fname='model'):
    try:
        return load_model("Models/" + fname + ".h5")
    except OSError:
        print(f'Error: could not load "Models/{fname}.h5')
        exit()
    
def generate_music(model, in_notes, in_durs, pitchnames, dur_names, num_notes):
    if len(in_notes) < 100:
        start = np.random.randint(0, len(in_notes) - 1)
    else:
        start = np.random.randint(50, len(in_notes) - 50)
    int_to_note = {number:note for number, note in enumerate(pitchnames)}
    int_to_dur = {number:dur for number, dur in enumerate(dur_names)}

    pattern_notes = in_notes[start]
    pattern_durs = in_durs[start]
    prediction_output = []

    for _ in range(num_notes):
        pred_in_notes = np.reshape(pattern_notes, (1, len(pattern_notes), 1))
        pred_in_durs = np.reshape(pattern_durs, (1, len(pattern_durs), 1))

        note_pred, dur_pred = model.predict([pred_in_notes, pred_in_durs], verbose=0)
        index_notes = np.argmax(note_pred)
        index_durs = np.argmax(dur_pred)
        note_result = int_to_note[index_notes]
        dur_result = int_to_dur[index_durs]
        prediction_output.append((note_result, dur_result))

        pattern_notes = np.append(pattern_notes, index_notes)
        pattern_notes = pattern_notes[1:]
        pattern_durs = np.append(pattern_durs, index_durs)
        pattern_durs = pattern_durs[1:]

    return prediction_output

def create_midi(prediction_output, fname="music"):
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
    filename = "Generated/" + fname + ".mid"
    i = 2
    while(os.path.exists(filename)):
        filename = "Generated/" + fname + f"{i}.mid"
        i += 1

    midi_stream.write("mid", fp=filename)
    return filename

def play_music(fname):
        # mixer config
        freq = 44100  # audio CD quality
        bitsize = -16   # unsigned 16 bit
        channels = 2  # 1 is mono, 2 is stereo
        buffer = 1024   # number of samples
        pygame.mixer.init(freq, bitsize, channels, buffer)
        clock = pygame.time.Clock()
        try:
            pygame.mixer.music.load(fname)
        except:
            return
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            clock.tick(30) # check if playback has finished

def save_data(fname, in_notes, in_durs, out_notes, out_durs, pitchnames, dur_names):
    if not os.path.exists(f'Data/{fname}'):
        os.mkdir(f'Data/{fname}')
    np.save(f'Data/{fname}/in_notes.npy', in_notes)
    np.save(f'Data/{fname}/in_durs.npy', in_durs)
    np.save(f'Data/{fname}/out_notes.npy', out_notes)
    np.save(f'Data/{fname}/out_durs.npy', out_durs)

    with open(f'Data/{fname}/pitchnames', 'w') as f:
        s = ""
        for p in pitchnames:
            s += p + " "

        f.write(s[:-1])

    with open(f'Data/{fname}/dur_names', 'w') as f:
        s = ""
        for d in dur_names:
            s += str(d) + " "

        f.write(s[:-1])

def load_data(fname):
    in_notes = np.load(f'Data/{fname}/in_notes.npy')
    in_durs = np.load(f'Data/{fname}/in_durs.npy')
    out_notes = np.load(f'Data/{fname}/out_notes.npy')
    out_durs = np.load(f'Data/{fname}/out_durs.npy')

    with open(f'Data/{fname}/pitchnames', 'r') as f:
        pitchnames = [x for x in f.read().split()]

    with open(f'Data/{fname}/dur_names', 'r') as f:
        dur_names = []
        for x in f.read().split():
            if '.' in x:
                dur_names.append(float(x))
            elif '/' in x:
                dur_names.append(Fraction(x))
            else:
                print("Error: dur_name string is unreadable:", x)
                exit()

    return in_notes, in_durs, out_notes, out_durs, pitchnames, dur_names

def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m',  '--Music',     nargs='*', help='input .mid foldername or pattern')
    argparser.add_argument('-t',  '--Train',     action='store_true', help='Whether to train a model')
    argparser.add_argument('-n',  '--Name',      help='Name for model, figure, and generated music')
    argparser.add_argument('-e',  '--Epochs',    help='Number of training epochs')
    argparser.add_argument('-s',  '--SaveData',  action='store_true', help='Save preprocessed data')
    argparser.add_argument('-l',  '--LoadData',  action='store_true', help='Load preprocessed data')
    argparser.add_argument('-u',  '--Use',       action='store_true', help='Use a specific model')
    argparser.add_argument('-b',  '--BatchSize', help='Size of training batches')
    argparser.add_argument('-v',  '--Verbose',   help='Tensorflow model fitting parameter')
    argparser.add_argument('-nn', '--NumNotes',  help='Number of notes generated')
    argparser.add_argument('-lb', '--Lookback',  help='Lookback of model architecture')
    
    args = argparser.parse_args()
    if not args.Music:
        if not (args.LoadData and args.Name):
            raise ValueError("Music input music file or files.")
    
    if args.Name:
        fname = args.Name
    else:
        if len(args.Music) == 1:
            fname = args.Music[0].replace('/','_')
        else:
            fname = ".".join(args.Music).replace('/','_')

    if args.Music:
        print(f'\t--Music      : {", ".join(args.Music)}')
    else:
        print(f'\t--Music      : {fname}')

    print(f'\t--Train      : {args.Train}')
    print(f'\t--Name       : {args.Name}')
    print(f'\t--Epochs     : {args.Epochs}')
    print(f'\t--SaveData   : {args.SaveData}')
    print(f'\t--LoadData   : {args.LoadData}')
    print(f'\t--Use        : {args.Use}')
    print(f'\t--BatchSize  : {args.BatchSize}')
    print(f'\t--NumNotes   : {args.NumNotes}')
    print(f'\t--Verbose    : {args.Verbose}')
    print(f'\t--Lookback   : {args.Lookback}')
    print()
    return args, fname

def music_generation_pipeline(lookback=512, epochs=75, batch_size=64, num_notes=200, verbose=1):
    print()
    args, fname = parse_args()
    if args.Lookback:
        try:
            lookback = int(args.Lookback)
        except ValueError:
            pass
    if args.LoadData:
        in_notes, in_durs, out_notes, out_durs, pitch, durs = load_data(fname)
    else:
        all_notes, all_durs = extract(args.Music)
        print("\nTransforming data...")
        in_notes, in_durs, out_notes, out_durs, pitch, durs = transform(all_notes, all_durs, lookback)
    
    if args.SaveData:
        save_data(fname, in_notes, in_durs, out_notes, out_durs, pitch, durs)

    print("\nLoading data. Shape:", in_notes.shape)
    if args.NumNotes:
        try:
            num_notes = int(args.NumNotes)
        except ValueError:
            pass

    if args.Train:
        if args.Epochs:
            try:
                epochs = int(args.Epochs)
            except ValueError:
                pass
            
        if args.BatchSize:
            try:
                batch_size = int(args.BatchSize)
            except ValueError:
                pass
            
        if args.Verbose:
            try:
                verbose = int(args.Verbose)
            except ValueError:
                pass

        print()
        if args.Use and args.Name:
            model = load_music_model(args.Name)
        else:
            model = build_model(lookback, out_notes.shape[1], out_durs.shape[1])

        history = train_model(model, in_notes, in_durs, out_notes, out_durs,
                              epochs, batch_size, fname, verbose)
        plot_history(history, fname)
        print("\nFinished training.")
    if args.Use and args.Name:
        model = load_music_model(args.Name)
    else:
        model = load_music_model(fname)
    print("\nGenerating music.")
    generated_music = generate_music(model, in_notes, in_durs, pitch, durs, num_notes)
    f = create_midi(generated_music, fname)
    print(f"Saved midi file at: '{f}'")
    print("\nPlaying music.")
    play_music(f)

def get_music_filename(fname, num_notes):
    in_notes, in_durs, _, _, pitch, durs = load_data(fname)
    model = load_music_model(fname)
    generated_music = generate_music(model, in_notes, in_durs, pitch, durs, num_notes)
    midi = create_midi(generated_music, fname)
    print(midi)
    play_music(midi)
    return midi

if __name__=="__main__":
    music_generation_pipeline()
    #get_music_filename('fugues', 100)

