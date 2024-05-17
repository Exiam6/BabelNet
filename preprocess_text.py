import pandas as pd
import numpy as np
import imdb
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import torch
import transformers
import pandas as pd
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import CLIPProcessor, CLIPModel


i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('/scratch/zz4330/ml-100k/Text/u.item', sep='|', names=i_cols,
encoding='latin-1')

ia = imdb.IMDb()
def get_summary(movie_name):
    mv = ia.search_movie(movie_name)[0]
    url = ia.get_imdbURL(mv)
    movie = ia.get_movie(mv.movieID) 
    
    r = requests.get(url=url)
    soup = BeautifulSoup(r.text, 'html.parser')
    
    cast = ("|").join([x['name'] for x in movie['cast'][:5]])
    director = movie['director'][0]['name']
    runtime = movie['runtimes']
    summ = soup.find("div", attrs = {'data-testid': 'storyline-plot-summary'}).text
    rating = soup.find("span", attrs = {'class': 'AggregateRatingButton__RatingScore-sc-1ll29m0-1 iTLWoV'}).text
    users = soup.find("div", attrs = {'class': 'AggregateRatingButton__TotalRatingAmount-sc-1ll29m0-3 jkCVKJ'}).text
    return(url, summ, cast, director, runtime, rating, users)

def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9]+", ' ', str(text))
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ',  text)
    text = re.sub(r" +", ' ', text)
    return text
    
for index, row in tqdm(items.iterrows(), total=items.shape[0]):
    try:
        movie_name = row['movie title']
        url, summ, cast, director, runtime, rating, users = get_summary(movie_name)
        items.loc[index, "IMDb URL"] = url
        items.loc[index, "Summary"] = summ
        items.loc[index, "Cast"] = cast
        items.loc[index, "Director"] = director
        items.loc[index, "Rating"] = rating
        items.loc[index, "Runtime"] = runtime
        items.loc[index, "No. of ratings"] = users
    except:
        print(movie_name)
        continue
def get_embeddings(input_txt):
    encodings = tokenizer.encode_plus(input_txt, add_special_tokens=True, max_length=16, return_tensors='pt', return_token_type_ids=False, return_attention_mask=True, padding="longest")
    attention_mask = pad_sequences(encodings['attention_mask'], maxlen=20, dtype=torch.Tensor ,truncating="post",padding="post")
    attention_mask = attention_mask.astype(dtype = 'int64')
    attention_mask = torch.tensor(attention_mask).to(device)

    input_ids = pad_sequences(encodings['input_ids'], maxlen=20, dtype=torch.Tensor ,truncating="post",padding="post")
    input_ids = input_ids.astype(dtype = 'int64')
    input_ids = torch.tensor(input_ids).to(device)
    
    with torch.no_grad():
        outputs = model.forward(input_ids, attention_mask)
        last_hidden_states = outputs.last_hidden_state.cpu().detach().numpy()
    
    torch.cuda.empty_cache()
    return(last_hidden_states)

items.to_csv('/scratch/zz4330/ml-100k/items.csv')
items['Summary'] = items['Summary'].apply(clean_text)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = model.to(device)

embeddings = []
for input_txt in tqdm(items['Summary'], total = items['Summary'].shape[0]):
    embedding = get_embeddings(input_txt)[0, -1, :]
    embeddings.append(embedding)

import csv

with open("/scratch/zz4330/ml-100k/Text/embeddings.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(embeddings)
