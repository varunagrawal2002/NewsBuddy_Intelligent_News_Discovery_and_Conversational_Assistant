function submitChat(index) {
    var userMessage = document.getElementById('userMessage' + index).value;
    var chatOutput = document.getElementById('chatOutput' + index);

    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'user_message=' + encodeURIComponent(userMessage),
    })
    .then(response => response.json())
    .then(data => {
        // Display the chatbot's response in the chatOutput div
        var chatMessage = document.createElement('p');
        chatMessage.textContent = 'Bot: ' + data.response;
        chatOutput.appendChild(chatMessage);
    });

    return false; // Prevent form submission
}
