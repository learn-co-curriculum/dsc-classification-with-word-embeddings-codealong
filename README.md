
# Classification With Word Embeddings - Codealong

## Introduction

In this lesson, you'll use everything you've learned in this section to perform text classification using word embeddings!

## Objectives

You will be able to:

- Effectively incorporate embedding layers into neural networks using Keras
- Import and use pretrained word embeddings from popular pretrained models such as GloVe
- Understand and explain the concept of a mean word embedding, and how this can be used to vectorize text at the sentence, paragraph, or document level


## Getting Started

Load the data, and all the frameworks and libraries. 


```python
import pandas as pd
import numpy as np
np.random.seed(0)
from nltk import word_tokenize
from gensim.models import word2vec
```

    C:\Users\medio\AppData\Local\Continuum\anaconda3\lib\site-packages\gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")


Now, load the dataset. You'll be working with the same dataset you worked with in the previous lab for this section, which you'll find inside `News_Category_Dataset_v2.zip`.  **_Go into the repo and unzip this file before continuing._**

Once you've unzipped this dataset, go ahead and use pandas to read the data stored in `News_Category_Dataset_v2.json` in the cell below. Then, display the head of the DataFrame to ensure everything worked correctly. 

**_NOTE:_** When using the `pd.read_json()` function, be sure to include the `lines=True` parameter, or else it will crash!


```python
df = pd.read_json('News_Category_Dataset_v2.json', lines=True)
df = df.sample(frac=0.2)
print(len(df))
df.head()
```

    40171





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>authors</th>
      <th>category</th>
      <th>date</th>
      <th>headline</th>
      <th>link</th>
      <th>short_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20315</th>
      <td>Janis Ian, ContributorSongwriter, author, perf...</td>
      <td>POLITICS</td>
      <td>2017-07-28</td>
      <td>This Is Not Government. It's Savagery.</td>
      <td>https://www.huffingtonpost.com/entry/this-is-n...</td>
      <td>I want to move to Hawaii solely to be able to ...</td>
    </tr>
    <tr>
      <th>105941</th>
      <td>Amanda Terkel</td>
      <td>POLITICS</td>
      <td>2014-11-23</td>
      <td>GOP Senator Urges Republicans To Move On From ...</td>
      <td>https://www.huffingtonpost.com/entry/jeff-flak...</td>
      <td></td>
    </tr>
    <tr>
      <th>173890</th>
      <td>Marcus Samuelsson, Contributor\nAward-Winning ...</td>
      <td>FOOD &amp; DRINK</td>
      <td>2012-11-14</td>
      <td>10 Recipes, 10 Ways To Deliciously Use Your Th...</td>
      <td>https://www.huffingtonpost.com/entry/10-recipe...</td>
      <td>My 10 favorite ways to make sure you're not wa...</td>
    </tr>
    <tr>
      <th>177657</th>
      <td>Michelle Manetti</td>
      <td>HOME &amp; LIVING</td>
      <td>2012-10-04</td>
      <td>Doorless Refrigerator Wall By Electrolux Desig...</td>
      <td>https://www.huffingtonpost.com/entry/doorless-...</td>
      <td>Hey, here's a way to subtract a step between y...</td>
    </tr>
    <tr>
      <th>97563</th>
      <td>Jon Hartley, ContributorWorld Economic Forum G...</td>
      <td>BUSINESS</td>
      <td>2015-02-27</td>
      <td>The Emerging Markets Housing Bubble</td>
      <td>https://www.huffingtonpost.com/entry/the-emerg...</td>
      <td>While in most advanced economies, housing pric...</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's transform the dataset, as you did in the previous lab. 

In the cell below:

*  Store the column that will be the target, `category`, in the variable `target`.
* Combine the `headline` and `short_description` columns and store the result in a column called `combined_text`. When concatenating these two columns, make sure they are separated by a space character (`' '`)!
* Use the `combined_text` column's map function to use the `word_tokenize` function on every piece of text. 
* Store the `.values` from the newly tokenized `combined_text` column inside the variable data


```python
target = df.category
df['combined_text'] = df.headline + ' ' + df.short_description
data = df['combined_text'].map(word_tokenize).values
```

## Loading A Pretrained GloVe Model

For this lab, you'll be loading the pretrained weights from **_GloVe_** (short for _Global Vectors for Word Representation_) from the [Stanford NLP Group](https://nlp.stanford.edu/projects/glove/).  These are commonly accepted as some of the best pre-trained word vectors available, and they're open source, so you can get them for free! Even the smallest file is still over 800 MB, so you'll you need to download this file manually. 

Note that there are several different sizes of pretrained word vectors available for download from the page linked above&mdash;for the purposes, you'll only need to use the smallest one, which still contains pretrained word vectors for over 6 billion words and phrases! To download this file, follow the link above and select the file called `glove.6b.zip`.  For simplicity's sake, you can also start the download by clicking [this link](http://nlp.stanford.edu/data/glove.6B.zip).  You'll be using the GloVe file containing 100-dimensional word vectors for 6 billion words. Once you've downloaded the file, unzip it, and move the file `glove.6B.50d.txt` into the same directory as this jupyter notebook. 

### Getting the Total Vocabulary

Although the pretrained GloVe data contains vectors for 6 billion words and phrases, you don't need all of them. Instead, you only need the vectors for the words that appear in the dataset. If a word or phrase doesn't appear in the dataset, then there's no reason to waste memory storing the vector for that word or phrase. 

This means that you need to start by computing the total vocabulary of the dataset. You can do this by adding every word in the dataset into a python `set` object. This is easy, since you've already tokenized each comment stored within `data`.

In the cell below, add every token from every comment in data into a set, and store the set in the variable `total_vocabulary`.

**_HINT_**: Even though this takes a loop within a loop, you can still do this with a one-line list comprehension!


```python
total_vocabulary = set(word for headline in data for word in headline)
```


```python
len(total_vocabulary)
print("There are {} unique tokens in the dataset.".format(len(total_vocabulary)))
```

    There are 71277 unique tokens in our dataset.


Now that you have gotten the total vocabulary, you can get the appropriate vectors out of the GloVe file. 

For the sake of expediency, the code to read the appropriate vectors from the file is included below. 


```python
glove = {}
with open('glove.6B.50d.txt', 'rb') as f:
    for line in f:
        parts = line.split()
        word = parts[0].decode('utf-8')
        if word in total_vocabulary:
            vector = np.array(parts[1:], dtype=np.float32)
            glove[word] = vector
```

After running the cell above, you now have all of the words and their corresponding vocabulary stored within the dictionary, `glove`, as key/value pairs. 

Double-check that everything worked by getting the vector for a word from the `glove` dictionary. It's probably safe to assume that the word 'school' will be mentioned in at least one news headline, so let's get the vector for it. 

Get the vector for the word `'school'` from `glove` in the cell below. 


```python
glove['school']
```




    array([-0.90629  ,  1.2485   , -0.79692  , -1.4027   , -0.038458 ,
           -0.25177  , -1.2838   , -0.58413  , -0.11179  , -0.56908  ,
           -0.34842  , -0.39626  , -0.0090178, -1.0691   , -0.35368  ,
           -0.052826 , -0.37056  ,  1.0931   , -0.19205  ,  0.44648  ,
            0.45169  ,  0.72104  , -0.61103  ,  0.6315   , -0.49044  ,
           -1.7517   ,  0.055979 , -0.52281  , -1.0248   , -0.89142  ,
            3.0695   ,  0.14483  , -0.13938  , -1.3907   ,  1.2123   ,
            0.40173  ,  0.4171   ,  0.27364  ,  0.98673  ,  0.027599 ,
           -0.8724   , -0.51648  , -0.30662  ,  0.37784  ,  0.016734 ,
            0.23813  ,  0.49411  , -0.56643  , -0.18744  ,  0.62809  ],
          dtype=float32)



Great&mdash;it worked!  Now that you've gotten the word vectors for every word in the  dataset, the next step is to combine all the vectors for a given headline into a **_Mean Embedding_** by finding the average of all the vectors in that headline. 

## Creating Mean Word Embeddings

For this step, it's worth the extra effort to write your own mean embedding vectorizer class, so that you can make use of pipelines from scikit-learn. Using pipelines will save us time and make the code a bit cleaner. 

The code for a mean embedding vectorizer class is included below, with comments explaining what each step is doing. Take a minute to examine it and try to understand what the code is doing. 


```python
class W2vVectorizer(object):
    
    def __init__(self, w2v):
        # takes in a dictionary of words and vectors as input
        self.w2v = w2v
        if len(w2v) == 0:
            self.dimensions = 0
        else:
            self.dimensions = len(w2v[next(iter(glove))])
    
    # Note from Mike: Even though it doesn't do anything, it's required that this object implement a fit method or else
    # It can't be used in a sklearn Pipeline. 
    def fit(self, X, y):
        return self
            
    def transform(self, X):
        return np.array([
            np.mean([self.w2v[w] for w in words if w in self.w2v]
                   or [np.zeros(self.dimensions)], axis=0) for words in X])
```

## Using Pipelines

Since you've created a mean vectorizer class, you can pass this in as the first step in the pipeline, and then follow it up with the model you'll feed the data into for classification. 

Run the cell below to create pipeline objects that make use of the mean embedding vectorizer that you built above. 


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

rf =  Pipeline([("Word2Vec Vectorizer", W2vVectorizer(glove)),
              ("Random Forest", RandomForestClassifier(n_estimators=100, verbose=True))])
svc = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(glove)),
                ('Support Vector Machine', SVC())])
lr = Pipeline([("Word2Vec Vectorizer", W2vVectorizer(glove)),
              ('Logistic Regression', LogisticRegression())])
```

Now, you'll create a list that contains a tuple for each pipeline, where the first item in the tuple is a name, and the second item in the list is the actual pipeline object. 


```python
models = [('Random Forest', rf),
          ("Support Vector Machine", svc),
          ("Logistic Regression", lr)]
```

You can then use the list you've created above, as well as the `cross_val_score` function from scikit-learn to train all the models, and store their cross validation score in an array. 

**_NOTE:_** Running the cell below may take a few minutes!


```python
scores = [(name, cross_val_score(model, data, target, cv=2).mean()) for name, model, in models]
```

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:   19.8s finished
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:    1.3s finished
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:   21.5s finished
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:    1.4s finished
    C:\Users\medio\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\svm\base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    C:\Users\medio\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\svm\base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    C:\Users\medio\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\medio\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    C:\Users\medio\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\medio\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:460: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)



```python
scores
```




    [('Random Forest', 0.3195585321385761),
     ('Support Vector Machine', 0.3025076172472023),
     ('Logistic Regression', 0.3280980534586512)]



These scores may seem pretty low, but remember that there are 41 possible categories that headlines could be classified into. This means the naive accuracy rate (random guessing) would achieve an accuracy of just over 0.02! Our models have plenty of room for improvement, but they do work!

## Deep Learning With Word Embeddings

To end, you'll see an example of how you can use an **_Embedding Layer_** inside of a Deep Neural Network to compute the own word embedding vectors on the fly, right inside the model! 

Don't worry if you don't understand the code below just yet&mdash;you'll be learning all about **_Sequence Models_** like the one below in the next section!

Run the cells below.

First, you'll import everything you'll need from Keras. 


```python
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, Embedding
from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing import text, sequence
```

Next, you'll convert the labels to a one-hot encoded format.


```python
y = pd.get_dummies(target).values
```

Now, you'll preprocess the text data. To do this, you start from the step where you combined the headlines and short description. You'll then use Keras's preprocessing tools to tokenize each example, convert them to sequences, and then pad the sequences so they're all the same length. 

Note how during the tokenization step, you set a parameter to tell the tokenizer to limit the overall vocabulary size to the `20000` most important words. 


```python
tokenizer = text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(list(df.combined_text))
list_tokenized_headlines = tokenizer.texts_to_sequences(df.combined_text)
X_t = sequence.pad_sequences(list_tokenized_headlines, maxlen=100)
```

Now, construct the neural network. Notice how the **_Embedding Layer_** comes second, after the input layer. In the Embedding Layer, you specify the size you want the word vectors to be, as well as the size of the embedding space itself.  The embedding size you specified is 128, and the size of the embedding space is best as the size of the total vocabulary that we're using. Since you limited the vocab to 20000, that's the size you choose for the embedding layer. 

Once the data has passed through an embedding layer, you feed this data into an LSTM layer, followed by a Dense layer, followed by output layer. You also add some Dropout layers after each of these layers, to help fight overfitting.

Our output layer is a Dense layer with 41 neurons, which corresponds to the 41 possible classes in the labels. You set the activation function for this output layer to `'softmax'`, so that the network will output a vector of predictions, where each element's value corresponds to the percentage chance that the example is the class that corresponds to that element, and where the sum of all elements in the output vector is 1. 


```python
embedding_size = 128
input_ = Input(shape=(100,))
x = Embedding(20000, embedding_size)(input_)
x = LSTM(25, return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.5)(x)
x = Dense(50, activation='relu')(x)
x = Dropout(0.5)(x)
# There are 41 different possible classes, so we use 41 neurons in our output layer
x = Dense(41, activation='softmax')(x)

model = Model(inputs=input_, outputs=x)
```

Once you have designed the model, you still have to compile it, and provide important parameters such as the loss function to use (`'categorical_crossentropy'`, since this is a multiclass classification problem), and the optimizer to use. 


```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

After compiling the model, you quickly check the summary of the model to see what the model looks like, and make sure the output shapes line up with what you expect. 


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_3 (InputLayer)         (None, 100)               0         
    _________________________________________________________________
    embedding_3 (Embedding)      (None, 100, 128)          2560000   
    _________________________________________________________________
    lstm_3 (LSTM)                (None, 100, 25)           15400     
    _________________________________________________________________
    global_max_pooling1d_1 (Glob (None, 25)                0         
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 25)                0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 50)                1300      
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 50)                0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 41)                2091      
    =================================================================
    Total params: 2,578,791
    Trainable params: 2,578,791
    Non-trainable params: 0
    _________________________________________________________________


Finally, you can fit the model by passing in the data, the labels, and setting some other hyperparameters such as the batch size, the number of epochs to train for, and what percentage of the training data to use for validation data. 

If trained for 3 epochs, you'll find the model achieves a validation accuracy of almost 41%. 

Run the cell below for 1 epoch. Note that this is a large network, so the training will take some time!


```python
model.fit(X_t, y, epochs=2, batch_size=32, validation_split=0.1)
```

    Train on 36153 samples, validate on 4018 samples
    Epoch 1/2
    36153/36153 [==============================] - 184s 5ms/step - loss: 2.6305 - acc: 0.3169 - val_loss: 2.4481 - val_acc: 0.3616
    Epoch 2/2
    36153/36153 [==============================] - 184s 5ms/step - loss: 2.3492 - acc: 0.3757 - val_loss: 2.3228 - val_acc: 0.4089





    <keras.callbacks.History at 0x1d757632f98>



After 1 epoch, the model does about as well as the shallow algorithms you tried above. However, the LSTM Network was able to achieve a validation accuracy of over 40% after only 3 epochs of training. It's likely that if you trained for more epochs or added in the rest of the data, the performance would improve even further (but the run time would get much, much longer). 

It's common to add embedding layers in LSTM networks, because both are special tools most commonly used for text data. The embedding layer creates it's own vectors based on the language in the text data it trains on, and then passes that information on to the LSTM network one word at a time. You'll learn more about LSTMs and other kinds of **_Recurrent Neural Networks_** in the next section!

## Summary

In this codealong, you used everything you know about word embeddings to perform text classification, and then you built a Multi-Layer Perceptron model that incorporated a word embedding layer in it's own architecture!
