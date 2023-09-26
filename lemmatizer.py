import emoji
import nltk
import re
import string
import unicodedata

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U0001FB00-\U0001FBFF"  # Symbols for Legacy Computing
                               u"\U0001FC00-\U0001FCFF"  # Symbols for Legacy Computing
                               u"\U0001F004-\U0001F0CF"  # Miscellaneous Symbols and Pictographs
                               u"\U0001F0D0-\U0001F0FF"  # Playing Cards
                               u"\U0001F10D-\U0001F10F"  # Dingbats
                               u"\U0001F170-\U0001F19A"  # Enclosed Alphanumeric Supplement
                               u"\U0001F1E6-\U0001F1FF"  # Regional Indicator Symbols
                               u"\U0001F201-\U0001F251"  # Squared Latin Abbreviations
                               u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols
                               u"\U0001F600-\U0001F64F"  # Emoticons
                               u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                               u"\U0001F700-\U0001F77F"  # Alchemical Symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U0001FB00-\U0001FBFF"  # Symbols for Legacy Computing
                               u"\U0001FC00-\U0001FCFF"  # Symbols for Legacy Computing
                               u"\U0001F004-\U0001F0CF"  # Miscellaneous Symbols and Pictographs
                               u"\U0001F0D0-\U0001F0FF"  # Playing Cards
                               u"\U0001F10D-\U0001F10F"  # Dingbats
                               u"\U0001F170-\U0001F19A"  # Enclosed Alphanumeric Supplement
                               u"\U0001F1E6-\U0001F1FF"  # Regional Indicator Symbols
                               u"\U0001F201-\U0001F251"  # Squared Latin Abbreviations
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_text(dataset):
    df = dataset.copy(deep=True)
    df['full_text'] = df['full_text'].str.lower()
    stop_words = stopwords.words(['danish', 'english', 'dutch', 'norwegian', 'swedish', 'french', 'german'])
    df['text_without_sw'] = df['full_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    df['text_unicode'] = df['text_without_sw'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
    df['text_greater_than_three_char'] = df['text_unicode'].str.replace(r'\b\w{1,2}\b', '')
    df['text_without_link'] = df['text_greater_than_three_char'].str.replace(r'\bhttp.*\b', '')
    df['text_without_emojis'] = df['text_without_link'].apply(remove_emojis)
    df['text_without_punctuations'] = df['text_without_emojis'].str.translate(str.maketrans('', '', string.punctuation)).str.strip()
    df['text_replacing_with_NA'] = df[df['text_without_punctuations'].astype(bool)]['text_without_punctuations']
    df['text_replacing_with_NA'] = df['text_replacing_with_NA'].str.replace('\d+', '')
    df['text_replacing_with_NA'] = df['text_replacing_with_NA'].str.strip()
    df['text_without_extra_spaces'] = df['text_replacing_with_NA'].str.replace(r'^\s*|\s\s*', ' ')
    df.dropna(subset=['text_without_extra_spaces'], inplace=True)
    return df['text_without_extra_spaces']