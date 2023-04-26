document.addEventListener('DOMContentLoaded', () => {
    const saveButtons = document.querySelectorAll('.save-button');
    const deleteButtons = document.querySelectorAll('.delete-button');
    saveButtons.forEach(button => {
        button.addEventListener('click', async () => {
            const fileName = button.getAttribute('data-filename');
            const saveUrl = button.getAttribute('data-url');
            const response = await fetch(`${saveUrl}?name=${encodeURIComponent(fileName.split('/').pop())}`);
            if (response.ok) {
                alert("MIDI file saved successfully\nSaved files are located in the 'home' tab.");
                location.reload();
            } else {
                alert("Error moving MIDI file");
            }
        });
    });
    deleteButtons.forEach(button => {
        button.addEventListener('click', async () => {
            const fileName = button.getAttribute('data-filename');
            const deleteUrl = button.getAttribute('data-url');
            const response = await fetch(`${deleteUrl}?name=${encodeURIComponent(fileName.split('/').pop())}`);
            if (response.ok) {
                alert("MIDI file deleted successfully");
                location.reload();
            } else {
                alert("Error deleting MIDI file");
            }
        });
    });
});