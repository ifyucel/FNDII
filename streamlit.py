import tensorflow.python.keras.backend as K
from tensorflow.keras.models import Model, load_model
import streamlit as st
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, STOPWORDS
from nltk import word_tokenize
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Bidirectional
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
nltk.download("stopwords")
from tensorflow.keras.models import load_model

nltk.download('wordnet')



stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

st.title("IZZET FATIH YUCEL 1804010003")

def remove_stopwords(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
            
    return result

user_input = st.text_area("Fake News Detection created by ifyucel (untrustworty)")

user_input = remove_stopwords(user_input)
user_input = " ".join(list(user_input))

list_of_words = []
for i in user_input:
    for j in i:
        list_of_words.append(j)

total_words = len(list(set(list_of_words)))



maxlen = -1
tokens = nltk.word_tokenize(user_input)
if(maxlen<len(tokens)):
    maxlen = len(tokens)

model = tf.keras.models.load_model('models/lstm_model.h5')

tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts([user_input])
test_sequences = tokenizer.texts_to_sequences([user_input])
padded_test = pad_sequences(test_sequences,maxlen = maxlen,padding = 'post')

if user_input:
    pred = model.predict(padded_test)

if user_input:
    if pred[0][0] >= 0.50:
        st.markdown(pred[0][0])
        st.markdown('True news')
    else:
        st.markdown(pred[0][0])
        st.markdown('False news')
		

