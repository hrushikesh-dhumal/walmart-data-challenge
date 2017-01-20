

# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# http://scikit-learn.org/stable/modules/multiclass.html


# Usage: python predict.py data/train.tsv data/test.tsv

import sys
import utils as ut

from bs4 import BeautifulSoup

import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


# Global Variables
bs_parser = "lxml"

TRANSLATOR_REMOVE_PUNCTUATION = str.maketrans({key: None for key in string.punctuation})
NLTK_STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


def remove_html(s):
    soup = BeautifulSoup(s, bs_parser)
    return soup.get_text().encode('ascii', 'ignore').decode("utf-8") 

def remove_punctuation(string):
    '''
    Source: http://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate
    '''
    # pass the translator to the string's translate method.
    return string.translate(TRANSLATOR_REMOVE_PUNCTUATION)

def lowercase_check_for_stopword_stem(token):
    
    token = token.lower()
    if token not in NLTK_STOPWORDS:
        return True, STEMMER.stem(token)
    else:
        return False, None

def remove_punctuation_lowercase_check_for_stopword_stem(string):
    
    string = remove_punctuation(string)
    
    # Not using nltk.word_tokenize(s) cause it takes long time to process.
    # Can be considered in case of complex text    
    token_list = string.split()

    op = []

    for token in token_list:
        a,b = lowercase_check_for_stopword_stem(token)
        if a:
            op.append(b)
  
    return ' '.join(op)


def generate_features_from_data_frame(df_cleaned, npl_features=1000):
    
    vectorizer = CountVectorizer(analyzer = "word", \
                                 tokenizer = None, \
                                 preprocessor = None, \
                                 ngram_range = (1, 1), \
                                 strip_accents = 'unicode', \
                                 max_features = npl_features)

    feature_matrix = vectorizer.fit_transform( \
                            df_cleaned['Product Long Description'])
    
    return feature_matrix



numargv = len(sys.argv)
if (numargv == 1) | (numargv == 2):
    sys.exit("usage: predict.py train.tsv test.tsv")
elif numargv == 3:
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
else:
    sys.exit("Invalid number of parameters!")


# Read the input file
df_train = ut.pd.read_table(train_data_path, sep='\t', \
                            encoding='utf-8', \
                            header=0, na_values=" NaN")
print("\nRead data")

df_train['Product Long Description'].loc[ \
            df_train[ 'Product Long Description'].isnull()] = "No data "

df_train['Product Name'].loc[df_train['Product Name'].isnull()] \
                    = "No data"
       
df_train['Product Long Description'] = \
            df_train['Product Long Description'] + " " + \
            df_train['Product Name']

df = df_train.loc[:, ['item_id', 'Item Class ID', \
                          'Product Long Description', 'Product Name', \
                          'tag']]

df['tag'] = df['tag'].apply(ut.change_string_to_int_list)

df_train_without_na = df.dropna(axis=0, how='any')
df_train_without_na = df_train_without_na.set_index('item_id')
df_train_without_na.index.name = None



df_train_without_na_after_bs = df_train_without_na.copy()

df_train_without_na_after_bs['Product Long Description'] = \
    df_train_without_na_after_bs['Product Long Description'].apply(remove_html)

df_train_without_na_after_bs['Product Long Description'].loc[ \
    (df_train_without_na_after_bs['Product Long Description'] == "")] = \
    "No data " + \
    df_train_without_na_after_bs['Product Name'].loc[ \
        (df_train_without_na_after_bs['Product Long Description'] == "")]


    
df_train_without_na_after_bs_nlp = df_train_without_na_after_bs.copy()

df_train_without_na_after_bs_nlp['Product Long Description'] = \
    df_train_without_na_after_bs_nlp['Product Long Description'].apply( \
                    remove_punctuation_lowercase_check_for_stopword_stem)

 
file_to_save = ut.os.path.join(ut.DATA_DIR, 'cleaned_train2.tsv')
df_train_without_na_after_bs_nlp.to_csv(file_to_save, encoding='utf-8', \
                           sep="\t",index=False)
print("\nProcessed train data and wrote to disk at " + file_to_save)


##########TEST READ
df_test = ut.pd.read_table(test_data_path, sep='\t', encoding='utf-8', \
                            header=0, na_values=" NaN")
print("\nRead data")

df_test = df_test.loc[:, ['item_id', 'Item Class ID','Product Long Description', 'Product Name']]


df_test['Product Long Description'].loc[ \
            df_test[ 'Product Long Description'].isnull()] = "No data"

df_test['Product Name'].loc[df_test['Product Name'].isnull()] \
                    = "No data"

df_test['Product Long Description'] = \
            df_test['Product Long Description'] + " "+ \
            df_test['Product Name']
        
        

df_test_after_bs = df_test.copy()

df_test_after_bs['Product Long Description'] = \
    df_test_after_bs['Product Long Description'].apply(remove_html)
    
    
df_test_after_bs['Product Long Description'].loc[ \
    (df_test_after_bs['Product Long Description'] == "")] = \
    "No data " + \
    df_test_after_bs['Product Name'].loc[ \
        (df_test_after_bs['Product Long Description'] == "")]
    
    
    
    
df_test_after_bs_nlp = df_test.copy()

df_test_after_bs_nlp['Product Long Description'] = \
    df_test_after_bs_nlp['Product Long Description'].apply( \
            remove_punctuation_lowercase_check_for_stopword_stem)
    
    
file_to_save = ut.os.path.join(ut.DATA_DIR, 'cleaned_test2.tsv')
df_test_after_bs_nlp.to_csv(file_to_save, encoding='utf-8', \
                           sep="\t",index=False)
print("\nProcessed test data and wrote to disk at " + file_to_save)



##########MODEL TRAINING & PREDICITON
X_train = generate_features_from_data_frame(df_train_without_na_after_bs_nlp)

X_test = generate_features_from_data_frame(df_test_after_bs_nlp)

mlb = MultiLabelBinarizer()
dummy_tags = mlb.fit_transform(df_train_without_na_after_bs_nlp['tag'])

classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(X_train, dummy_tags)
print("\n Training model")

yt = classif.predict(X_test)

t = mlb.inverse_transform(yt)

prediction = []

for e in t:
    prediction.append(list(e))


df_res = ut.pd.Series(prediction, name='tag', index=df_test.item_id)

df_res.to_csv('tags.tsv', sep="\t", encoding='utf-8', header=True)
print("\n Prediction complete")