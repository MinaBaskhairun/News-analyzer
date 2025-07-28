#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import spacy
from collections import Counter
from wordcloud import WordCloud


# In[2]:


API_KEY = "4ce91b386876467e8cd91ea7c2bc2b12"

def fetch_news(api_key, query="quantum", language="en", page_size=10):
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&pageSize={page_size}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    with open("articles.json", "w") as f:
        json.dump(data, f, indent=2)
    return data["articles"]

articles = fetch_news(API_KEY)


# In[3]:


def extract_contents(articles):
    return [a["content"] or "" for a in articles if a["content"]] # Get non-empty article content

contents = extract_contents(articles)
df = pd.DataFrame(articles)[["title", "source", "url"]]
df


# In[4]:


nlp = spacy.load("en_core_web_sm")

def extract_entities(contents):
    entity_counter = Counter()
    for text in contents:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                entity_counter[ent.text.strip()] += 1
    return entity_counter.most_common(20)

top_entities = extract_entities(contents)
top_entities


# In[5]:


import matplotlib.pyplot as plt

def plot_entities(entities):
    labels, values = zip(*entities)
    plt.figure(1,figsize=(10, 6))
    plt.barh(labels, values, color="skyblue")
    plt.xlabel("Frequency")
    plt.title("Top Named Entities in News Articles")
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

plot_entities(top_entities)


# In[6]:


def generate_wordcloud(entities):
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(dict(entities))
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()

generate_wordcloud(top_entities)

