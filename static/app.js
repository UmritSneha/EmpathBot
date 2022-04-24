class Chatbox {
    constructor() {
        this.args = {
            chatBox: document.querySelector('.chat-screen'),
            chatForm: document.querySelector('.chat-form'),
            chatBody: document.querySelector('.chat-body'),
            chatInput: document.querySelector('.chat-input'),
            sendButton: document.getElementById('send-button'),
            startButton: document.getElementById('start-button'),
            optionHeader: document.querySelector('.chat-header-option'),
            endButton: document.querySelector('.end-chat'),
            feedbackForm: document.querySelector('.chat-session-end'),
            greatButton: document.getElementById('great-button'),
            badButton: document.getElementById('bad-button')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const {startButton, chatForm, chatBody, chatInput, chatBox, sendButton, optionHeader, endButton, feedbackForm} = this.args;
        

        startButton.addEventListener('click', () => this.onStartButton(chatForm, chatBody, chatInput, optionHeader));

        sendButton.addEventListener('click', () => this.onSendButton(chatInput));

        const node = chatInput.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(chatInput)
            }
        });

        endButton.addEventListener('click', () => this.onEndButton(chatBody, chatInput, feedbackForm, optionHeader));

    }


    getUserDetails(chatForm) {
        var username = chatForm.getElementById("userName").value;
        var useremail = chatForm.getElementById('userEmail').value;

        return username, useremail
    }

    onStartButton(chatForm, chatBody, chatInput, optionHeader) {
        this.state = true;
        chatForm.classList.add('hide');
        chatBody.classList.remove('hide');
        chatInput.classList.remove('hide');
        optionHeader.classList.remove('hide');
        
    }

    onSendButton(chatInput) {
        var textField = chatInput.querySelector('input');
        let text1 = textField.value

        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
          })
          .then(r => r.json())
          .then(r => {
            let msg2 = { name: "Zoey", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText()
            textField.value = ''

        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText()
            textField.value = ''
          });
    }

    updateChatText() {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "Zoey")
            {
              const re = /[.!?]/;
              const sentences = (item.message).split(re);
              const sent_count = sentences.length - 1
              if (sent_count>3){
                for (let i=sent_count-1; i>=0; i--){
                        item.message = sentences[i]
                        if (item.message != "") {
                            html += '<div class="chat-bubble you">' + item.message + '</div>'
                        }
                    }

              } else {
                html += '<div class="chat-bubble you">' + item.message + '</div>'
              }

            }
            else
            {
                html += '<div class="chat-bubble me">' + item.message + '</div>'

            }
          });
        const chatmessage = document.querySelector('.chat-body');
        chatmessage.innerHTML = html;

    }




    onEndButton(chatBody, chatInput, feedbackForm, optionHeader){
        chatBody.classList.add('hide');
        optionHeader.classList.add('hide');
        chatInput.classList.add('hide');
        feedbackForm.classList.remove('hide');
    }

}

const chatbox = new Chatbox();
chatbox.display();