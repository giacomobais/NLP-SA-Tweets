import spacy
from data_loader import load_data
import pandas as pd
from tqdm import tqdm
import re
import json
import yaml

def extract_text(df, n_samples = None):
    # extract the text column from the dataframe for the first n_samples, ignore nan but make sure there are still n_samples
    if n_samples is None or n_samples >= len(df):
        n_samples = len(df)
    text = df['text'][:n_samples].dropna()
    if len(text) < n_samples:
        # if there are less than n_samples, we need to extract more
        # starting from the n_samples+1 until we have n_samples that are not nan
        for i in range(n_samples+1, len(df)):
            if not pd.isna(df['text'][i]):
                text = pd.concat([text, df['text'][i]])
                if len(text) >= n_samples:
                    break
    return text


def clean_text(text):
    print('Removing special characters...')
    # add a tqdm progress bar
    for i in tqdm(range(len(text))):
        # remove special \n and \t
        text[i] = text[i].replace('\n', ' ').replace('\t', ' ')
        
        # remove http links
        text[i] = re.sub(r'http\S+', '', text[i])
        # remove everything after @handles
        text[i] = re.sub(r'@\S+', '', text[i])
        # remove multiple spaces
        text[i] = ' '.join(text[i].split())
    return text


def extract_sentiment(df, n_samples = None):
    mapping = {'Positive': 0, 'Negative': 1}
    return mapping, df['sentiment'][:n_samples]

def shuffle_data(df, seed = 42):
    return df.sample(frac=1, random_state = seed).reset_index(drop=True)

def drop_blanks(text, sentiment):
    for i, t in enumerate(text):
        if t == '':
            # if the text is empty, we remove it
            text = text.drop(i)
            sentiment = sentiment.drop(i)
    return text, sentiment

def save_dataset(text, sentiment, mapping):
    clean_df = pd.DataFrame({'text': text, 'sentiment': sentiment})
    clean_df.to_csv('data/processed/cleaned_tweets.csv', index=False)

    # save the mapping to a json
    with open('data/processed/category_mapping.json', 'w') as f:
        json.dump(mapping, f)

def main():
    config = yaml.safe_load(open('config/config.yaml'))
    print('Loading data...')
    df = load_data('data/raw/sentiment_tweets.csv')
    df = shuffle_data(df)
    print('Extracting text...')
    text = extract_text(df, config['n_samples'])
    print('Cleaning text...')
    text = clean_text(text)
    print('Extracting ticket category...')
    mapping, sentiment = extract_sentiment(df, config['n_samples'])
    # invert the mapping
    inversed_mapping = {v: k for k, v in mapping.items()}
    # convert sentiment to 0 and 1 for convenience
    for i in range(len(sentiment)):
        if sentiment[i] == 4:
            sentiment[i] = 1
    # drop blanks
    text, sentiment = drop_blanks(text, sentiment)

    print('Saving dataset...')
    save_dataset(text, sentiment, inversed_mapping)

if __name__ == '__main__':
    main()
