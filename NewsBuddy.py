#Import all the packages 
import openai
from transformers import AutoTokenizer, AutoModel
import requests
import torch
import logging
import re
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR) 

#Setting up the BERT Model with 12 layers and trained on lowercase letters
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

#Set up the News API and collect the articles from the api_endpoint
api_key = '6efab5ad26f54f2da8d4ee61071a5d82'
api_endpoint = 'https://newsapi.org/v2/top-headlines'
source = 'google-news-in'

#Taking the user's input
user_search = input("Mention the News you are looking for: ")

#Define a function to fetch all the news articles
def news_articles(user_search):
    params = {
        'sources': source,
        'apiKey': api_key
    }
    response = requests.get(api_endpoint, params=params)
    articles = response.json()['articles']
    return articles
articles = news_articles(user_search)

#Function to clean and preprocess the text data
def clean_data(text):
    text = re.sub(r'[:,;.''""\|/&^—-]', ' ', text)
    text = re.sub(r'[^\w\s]', '', text).strip()
    cleaned_text = text.lower() 
    return cleaned_text

#Function to encode the users_search and articles_title
def encode_data(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()

#Function for lexical text similarity using cosine similarity
def similarity_scores(user_embedding, article_embedding):
    similarity = cosine_similarity(user_embedding.reshape(1, -1), article_embedding.reshape(1, -1))[0][0]
    return similarity

#Using all the functions in a loop,creating the similarity scores
similar_articles = []
user_text = clean_data(user_search)
user_embedding = encode_data(user_text)
for a in articles:
    article_text = clean_data(a['title'])
    article_embedding = encode_data(article_text)
    simil = similarity_scores(user_embedding, article_embedding)
    similar_articles.append((a, simil))

#Sorting the top 5 articles as per the similarity scores
similar_articles = sorted(similar_articles, key=lambda x: x[1], reverse=True)
recommended_articles = [a[0] for a in similar_articles][:5]

#A ChatBOT that uses the OpenAI API for generating appropriate responses to user's messages
class Chat:
    def __init__(self):
        openai.api_key = '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'
        self.messages = []
    def chat(self, message):
        self.messages.append({"role": "system", "content": "You are a coding tutor bot to help users find news articles and engage in discussions."})
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            max_tokens=50
        )
        self.messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
        return response["choices"][0]["message"]["content"]
ChatBOT = Chat()   

#Using ChatBOT to allow the user's to ask their queries for each article
for i, a in enumerate(recommended_articles, start=1):
    print(f"Article {i}:")
    print(f"Title: {a['title']}")
    print(f"URL: {a['url']}")
    print("------")
    user_doubt = input("Enter your queries or type 'next' to move to the next article or type 'exit' to exit the program: ")
    if user_doubt.lower() == "exit":
        break
    exit_flag = False
    while user_doubt.lower() != "next":
        if user_doubt.lower() == "exit":
            exit_flag = True
            break
        response = ChatBOT.chat(user_doubt)
        print("AI:", response)
        user_doubt = input("Enter your queries or type 'next' to move to the next article or type 'exit' to exit the program: ")
    print("------")
    if exit_flag:
        break