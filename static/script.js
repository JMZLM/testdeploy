// Control playback (pause, next, previous)
function controlPlayer(action) {
    fetch(`/control/${action}`)
        .then(response => response.json())
        .then(data => {
            if (data.status === action || data.status === 'paused') {
                window.location.reload(); // Reload to update current song
            } else {
                alert(`Error: Unable to ${action}.`);
            }
        });
}
