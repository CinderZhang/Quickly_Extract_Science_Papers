# %% combine all text files in a folder into one text file
import os
import glob

# specify your path
path = 'D:\\NPort\\Quickly_Extract_Science_Papers\\output'

# use glob to match the pattern '.txt'
files = glob.glob(os.path.join(path, '*.txt'))

# combine all files
combined_text = ''
for file in files:
    with open(file, 'r', encoding='utf8') as f:
        combined_text += f.read() + '\n'  # add a newline character between texts from different files
# %% save the combined text
with open('combined_text.txt', 'w') as f:
    f.write(combined_text)
# %% reorganize the combined text based on topics
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel

nltk.download('punkt')
nltk.download('stopwords')

# Tokenize your text
tokens = word_tokenize(combined_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokens = [i for i in tokens if not i in stop_words]

# Create a dictionary from the data
id2word = corpora.Dictionary([tokens])

# Create corpus
texts = [tokens]

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Apply LDA
lda_model = LdaModel(corpus=corpus,
                     id2word=id2word,
                     num_topics=10,  # You can change this number depending on how many topics you think there might be
                     random_state=100,
                     update_every=1,
                     chunksize=100,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)

# Print the keyword in the topics
print(lda_model.print_topics())
