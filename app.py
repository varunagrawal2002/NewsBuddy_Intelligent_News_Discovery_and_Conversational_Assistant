from flask import Flask, render_template, request, jsonify
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import requests
import torch
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import spacy

app = Flask(__name__)

# Setting up the BERT Model with 12 layers and trained on lowercase letters
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Set up the News API and collect the articles from the api_endpoint
api_key = '6efab5ad26f54f2da8d4ee61071a5d82'
api_endpoint = 'https://newsapi.org/v2/top-headlines'
source = 'google-news-in'

# Create a chatbot instance
chatbot = ChatBot(
    'MyBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand.',
            'maximum_similarity_threshold': 0.50,
        }
    ],
    preprocessors=[
        'chatterbot.preprocessors.clean_whitespace'
    ],
    language='english',  # Specify the language here
)

# Create a new trainer for the chatbot
trainer = ListTrainer(chatbot)

# Train the chatbot on English language data
trainer.train([
    'Hi',
    'Hello',
    'I need your assistance regarding my order',
    'Please, Provide me with your order id',
    'I have a complaint.',
    'Please elaborate, your concern',
    'How long it will take to receive an order ?',
    'An order takes 3-5 Business days to get delivered.',
    'Okay Thanks',
    'No Problem! Have a Good Day!'
])

# Function to fetch news articles
def news_articles(user_search):
    params = {
        'sources': source,
        'apiKey': api_key
    }
    response = requests.get(api_endpoint, params=params)
    articles = response.json()['articles']
    return articles

# Function to clean and preprocess text data
def clean_data(text):
    text = re.sub(r'[:,;.''""\|/&^—-]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text).strip()
    cleaned_text = text.lower()
    return cleaned_text

# Function to encode user_search and articles_title
def encode_data(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()

# Function for lexical text similarity using cosine similarity
def similarity_scores(user_embedding, article_embedding):
    similarity = cosine_similarity(user_embedding.reshape(1, -1), article_embedding.reshape(1, -1))[0][0]
    return similarity

# Route to render the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the search functionality
@app.route('/search', methods=['POST'])
def search():
    user_search = request.form['search_query']
    articles = news_articles(user_search)

    similar_articles = []
    user_text = clean_data(user_search)
    user_embedding = encode_data(user_text)

    for a in articles:
        article_text = clean_data(a['title'])
        article_embedding = encode_data(article_text)
        simil = similarity_scores(user_embedding, article_embedding)
        similar_articles.append((a, simil))

    similar_articles = sorted(similar_articles, key=lambda x: x[1], reverse=True)
    recommended_articles = [a[0] for a in similar_articles][:2]

    return render_template('result.html', articles=recommended_articles)

# Route to handle chat functionality
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']

    # Get a response from the chatbot
    response = chatbot.get_response(user_message)

    return jsonify({'response': str(response)})

if __name__ == '__main__':
    app.run(debug=True)
