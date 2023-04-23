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

@app.route('/pipeline')
def pipeline_page():
    return render_template('pipeline.html')

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    action = request.form['action']
    music_input = request.form['music_input']
    names = request.form['names']
    save = 'save_data' in request.form
    load = 'load_data' in request.form
    epochs = int(request.form['epochs'])
    num_notes = int(request.form['num_notes'])
    lookback = int(request.form['lookback'])
    use_cpu = 'use_cpu' in request.form
    use_model = 'load_model' in request.form
    batch_size = int(request.form['batchsize'])
    verbose = int(request.form['verbose'])
    if action == 'train':
        pipeline.train(music_input, names, save, load, use_model, epochs, lookback, batch_size, verbose, use_cpu)
    elif action == 'generate':
        pipeline.generate(names, num_notes, use_cpu)

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
