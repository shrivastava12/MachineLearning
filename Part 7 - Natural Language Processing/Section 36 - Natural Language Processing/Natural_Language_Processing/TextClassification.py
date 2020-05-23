# Text classification model 

# Importing library
import nltk
import sklearn
import pandas as pd
import numpy as np


# Importing dataset of sms messages
df = pd.read_table('SMSSpamCollection',header = None,encoding = 'utf-8')

# Printing useful information about the datasets 
df.info()
df.head()

# Check class distribution
classes =  df[0]
classes.value_counts()

# Preprocessing the dataset

# Convert class labels to binary values 0 = ham, 1 =  spam
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(classes)

# Store the SMS message data
text_messages = df[1]

# use regular expressions to replace email addresses,urls,phone number,other numbers
# Processing email address with the 'email'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')

# Replace URLS with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')

# Replace money symbols with 'moneysymb'
processed = processed.str.replace(r'£|\$', 'moneysymb')

# Replace 10 digit phone numbers (formate include paranthesis, space,nospace,dashes) with phonenumber
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')

# Replace numbers with numbr
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

# Remove punctuation 
processed = processed.str.replace(r'^\s+|\s+?$', ' ')

# Removing withspace between term with a single space
processed = processed.str.replace(r'\s+', ' ')

# Removing Leading and trailing whitespace
processed  = processed.str.replace(r'^\s+|\s+?$', '')

# change words to lower case  -Hellom,HELLO,hello are all the same
processed = processed.str.lower()


from nltk.corpus import stopwords
# remove stop words from text messages
stop_words = set(stopwords.words('english'))
processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

# Remove word stems using a porter stemmer
ps = nltk.PorterStemmer()
processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

#  Generating Features
from nltk.tokenize import word_tokenize

# Create bag_of _words
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)


# Use the 1500 most common words as feature
word_features = list(all_words.most_common())[:1500]


# The find_features function will determine which of the 1500 word feature are the contained in the list
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)
        
    return features

# let see an example
features = find_features(processed[0])
for key,value in features.items():
    if value == True:
        print(key)
        
# Now lets do it for all the messages
messages = zip(processed,y)

# define a seed for reproducibility
seed = 1
np.random.seed = seed
np.random.shuffle(messages)

# call find_features function for each SMS message
featuresets = [(find_features(text), label) for (text, label) in messages]
# we can split the featuresets into training and testing datasets using sklearn
from sklearn import model_selection

# split the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)




# We can use sklearn algorithms in NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC(kernel = 'linear'))

# train the model on the training data
model.train(training)

# and test on the testing dataset!
accuracy = nltk.classify.accuracy(model, testing)*100