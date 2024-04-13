import json
import plotly
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals 
import joblib
#from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine
#import dill as pickle
import utils as utils
import autocorrect

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///'+'data/DisasterResponse.db')
df = pd.read_sql_table('message', engine)
df_redefine = df.iloc[ : , -36:]
df_redefine = df_redefine.drop(['related','child_alone'],axis=1)
df_redefine_columnnames = list([x.replace('_', ' ') for x in df_redefine])

words=['flood','floods','aid' ,'request','direct' ,'search' ,'rescue','food' ,'shelter','refugees','buildings','weather related','storm','earthquake','fire' ,'security','medical help','water','clothing','missing','death','electricity','hospital','medical products','military','transport'
'cold']
'''
def tokenize(text):
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower()).strip()

        for w in tokens:  
        #remove stop words
            if w not in stopwords.words("english"):
        #lemmatization
        #reduce words to their root form
                lemmed = WordNetLemmatizer().lemmatize(w)
                clean_tokens.append(lemmed)
        return clean_tokens
'''
 
    

model = joblib.load('models/classifier.pkl')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')


def index():
    
    #create visuals

    #extract data for visuals
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(genre_counts.index)

    category_message = df.iloc[ : , -36:]
    category_message = category_message.drop(['related','child_alone'],axis=1)
    category_counts = category_message.sum(axis=0).sort_values(ascending=False)
    category_names = list([x.replace('_', ' ') for x in category_counts.index])

    #define graphs
    graphs = [
        {
            'data':[
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color='indianred')
                    )       
            ],

            'layout': {
                'title' : 'Distribution of Message Channels',
                'xaxis': {
                    'title': 'Channels'
                    },
                'yaxis': {
                    'title': 'Count'
                }
            }
        },

        {
            'data':[
                Bar(
                    x=category_names,
                    y=category_counts.tolist(),
                    marker=dict(color=category_counts.tolist())
                    )       
            ],

            'layout': {
                'title' : 'Distribution of Disaster Response Categories for Messages',
                'xaxis': {'title': 'Category', 'tickangle': 45},
                'yaxis': {'title': 'Count'}
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    #words_lower = [word.lower() for word in words]
    query = request.args.get('query', '') 
    query_words = query.lower().split()

    if any(word in words for word in query_words):
        query2 = query
    else:
        #query2 = autocorrect.autoCorrect_sentence(query)
        corrected_words = [autocorrect.autoCorrect(word) for word in query_words]
    # Filter out None values from corrected_words
        corrected_words = [word for word in corrected_words if word is not None]
        query2 = ' '.join(corrected_words)

    #query2=autocorrect.autoCorrect(query)
    print(query2)
    # use model to predict classification for query
    classification_labels = model.predict([query2])[0]
    classification_results = dict(zip(df_redefine_columnnames, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

    


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()