function validateSaveLoad() {
    const saveData = document.getElementById("save_data");
    const loadData = document.getElementById("load_data");

    if (saveData.checked && loadData.checked) {
        alert("Save Data and Load Data cannot both be true.");
        return false;
    }
    return true;
}

function validateForm() {
    return validateSaveLoad();
}

function updateSliderValue(sliderId, valueId) {
    const slider = document.getElementById(sliderId);
    const valueDisplay = document.getElementById(valueId);

    slider.addEventListener('input', function() {
        valueDisplay.textContent = this.value;
    });
}



document.addEventListener('DOMContentLoaded', function() {
    updateSliderValue('epochs', 'epochs-value');
    updateSliderValue('num_notes', 'num_notes-value');
    updateSliderValue('lookback', 'lookback-value');
    updateSliderValue('batchsize', 'batchsize-value');
    updateSliderValue('verbose', 'verbose-value');
});

const form = document.getElementById('pipeline-form');
const output = document.getElementById('output-content');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    const response = await fetch("{{ url_for('train') }}", {
        method: 'POST',
        body: formData
    });

    if (response.ok) {
        console.log("Training instance submitted");
    } else {
        console.error("Error submitting training instance");
    }
});

const socket = io.connect('{{ request.url_root }}');

socket.on('console_output', (data) => {
    const message = document.createElement('p');
    message.textContent = data.message;
    output.appendChild(message);
});