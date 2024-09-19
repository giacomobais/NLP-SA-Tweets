import spacy
from data_loader import load_data, save_data
import pandas as pd
from tqdm import tqdm
import re
import json

def extract_text(df, n_samples = None):
    # extract the text column from the dataframe for the first n_samples, ignore nan but make sure there are still n_samples
    if n_samples is None or n_samples >= len(df):
        n_samples = len(df)
    text = df['Ticket Description'][:n_samples].dropna()
    if len(text) < n_samples:
        # if there are less than n_samples, we need to extract more
        # starting from the n_samples+1 until we have n_samples that are not nan
        for i in range(n_samples+1, len(df)):
            if not pd.isna(df['Ticket Description'][i]):
                text = pd.concat([text, df['Ticket Description'][i]])
                if len(text) >= n_samples:
                    break
    return text


def clean_text(text):
    nlp = spacy.load('en_core_web_sm')
    def remove_special_characters(text):
        print('Removing special characters...')
        # add a tqdm progress bar
        for i in tqdm(range(len(text))):
            doc = nlp(text[i])
            # remove special \n and \t
            text[i] = text[i].replace('\n', ' ').replace('\t', ' ')
            # make sure all words are separated by a single space
            # remove multiple spaces
            text[i] = ' '.join(text[i].split())
            # remove html tags
            text[i] = re.sub(r'<.*?>', '', text[i])
            # remove everything between brackets
            text[i] = re.sub(r'\[.*?\]', '', text[i])
        return text
    def lemmatize(text):
        print('Lemmatizing...')
        # add a tqdm progress bar
        for i in tqdm(range(len(text))):
            doc = nlp(text[i])
            text[i] = ' '.join([token.lemma_ for token in doc])
            text[i] = re.sub(r'\{\s+', '{', text[i])
            text[i] = re.sub(r'\s+\}', '}', text[i])
        return text
    text = remove_special_characters(text)
    text = lemmatize(text)
    return text

def extract_ticket_category(df):
    # Extract the Ticket Type column and convert it to categorical integers
    print('Extracting ticket category...')
    categories = df['Ticket Type'].unique()
    category_to_int = {cat: i for i, cat in enumerate(categories)}
    
    return df['Ticket Type'].map(category_to_int), category_to_int

def main():
    print('Loading data...')
    df = load_data('data/raw/customer_support_tickets.csv')
    print('Extracting text...')
    text = extract_text(df)
    print('Cleaning text...')
    text = clean_text(text)
    print('Extracting ticket category...')
    category, mapping = extract_ticket_category(df)
    clean_data = pd.DataFrame({'text': text, 'category': category})
    save_data(clean_data, 'data/processed/cleaned_tickets.csv')
    # save the mapping to a json
    with open('data/processed/category_mapping.json', 'w') as f:
        json.dump(mapping, f)

if __name__ == '__main__':
    main()
