from flask import Flask, render_template, request, redirect, url_for
import pipeline
import os, glob

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
    return render_template('generate.html', models=models)

@app.route('/generate_music', methods=['POST'])
def generate_music():
    num_notes = int(request.form['num_notes'])
    use_cpu = 'use_cpu' in request.form
    name = request.form['action']
    pipeline.generate(name, num_notes, use_cpu)
    return redirect(url_for('generate'))


if __name__ == '__main__':
    app.run(debug=True)
