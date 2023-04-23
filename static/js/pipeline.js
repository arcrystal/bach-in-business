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
