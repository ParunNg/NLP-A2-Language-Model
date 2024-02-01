from flask import Flask, render_template, request
from lstm import LSTMLanguageModel, generate
import torch
import torchtext

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = torch.load('./model/vocab')
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

params, state = torch.load('./model/best-val-lstm_lm.pt')
model = LSTMLanguageModel(**params).to(device)
model.load_state_dict(state)
model.eval()

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        # Hyperparameters for model inference
        prompt = request.form['prompt']
        max_seq_len = int(request.form['max_seq_len'])
        temperature = float(request.form['temperature'])
        seed = int(request.form['seed'])

        output = generate(prompt, max_seq_len, temperature, model, tokenizer,
                            vocab, device, seed)
        return render_template('home.html', output=' '.join(output), show_text="block")

    else:
        return render_template('home.html', show_text="none")

if __name__ == '__main__':
    app.run(debug=True)