
from flask import Flask, render_template, request, redirect, url_for
import pipeline
import os, glob, shutil

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    files = []
    for f in glob.glob("static/Generated/keep/*.mid"):
        fname = os.path.join('Generated', 'keep', os.path.basename(f))
        files.append(url_for('static', filename=fname))
    return render_template('home.html', files=files)

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    music_input = request.form['music_input']
    names = request.form['names']
    save = 'save_data' in request.form
    load = 'load_data' in request.form
    epochs = int(request.form['epochs'])
    lookback = int(request.form['lookback'])
    use_cpu = 'use_cpu' in request.form
    use_model = 'load_model' in request.form
    batch_size = int(request.form['batchsize'])
    verbose = int(request.form['verbose'])
    pipeline.train(music_input, names, save, load, use_model, epochs, lookback, batch_size, verbose, use_cpu)
    return redirect(url_for('train'))

@app.route('/generate')
def generate():
    model_folder = os.path.join('static', 'Models')
    models = [f.split('.')[0] for f in os.listdir(model_folder) if f.endswith('.h5')]
    files = []
    for f in glob.glob("static/Generated/*.mid"):
        fname = os.path.join('Generated', os.path.basename(f))
        files.append(url_for('static', filename=fname))
    return render_template('generate.html', models=models, generated=files)

@app.route('/generate_music', methods=['POST'])
def generate_music():
    num_notes = int(request.form['num_notes'])
    use_cpu = 'use_cpu' in request.form
    name = request.form['action']
    pipeline.generate(name, num_notes, use_cpu)
    return redirect(url_for('generate'))

@app.route('/save_midi')
def save_midi():
    name = request.args.get('name', default='example', type=str)
    src = os.path.join('static', 'Generated', name)
    dest = os.path.join('static', 'Generated', 'keep', name)

    if os.path.exists(src):
        shutil.move(src, dest)
        return "MIDI file saved successfully", 200
    else:
        return "MIDI file not found", 404

# Update the delete_midi route
@app.route('/delete_midi')
def delete_midi():
    name = request.args.get('name', default='example', type=str)
    file_path = os.path.join('static', 'Generated', name)

    if os.path.exists(file_path):
        os.remove(file_path)
        return "MIDI file deleted successfully", 200
    else:
        return "MIDI file not found", 404


if __name__ == '__main__':
    app.run(debug=True)
