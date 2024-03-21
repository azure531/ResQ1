# Disaster Response Project

## Table of Contents

- [Project Overview](#Project-Overview)
- [Project Components](#Project-Components)
- [Local Installation](#Local-Installation)
- [Project Deployment](#Project-Deployment)
- [Live Demo](#Live-Demo)
- [Additional Issues](#Additional-Issues)


![Alt Image text](/media/AppDemo.gif?raw=true)


## Project Overview

In this project, it analyze thousands of real messages that were sent through natural disasters, either through social media or directly to disaster response organizations.

A ETL pipeline is built to precess message and category data from csv files, and load them into a SQLite database.

A machine learning pipeline is created to categorize events that sent classified messages to appropriate disaster relief agencies.

The project also includes the web app  that provides visualizations, and help emergency workers to classify new messages for 36 categories. 

Machine learning are critical to help different organizations to understand which messages are relevant to them, and which messages to prioritize.

## Project Components
There are three components for this project.

1. ETL Pipeline
`process_data.py`, a data cleaning pipeline that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. ML Pipeline
`train_classifier.py`, a machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. Flask Web App
the flask web app:
- Add data visualizations using Plotly in the web app. 

## Local Installation
#### Instructions:
1. Run commands in the project's root directory to set up your database and model.
- To run ETL pipeline that cleans data and stores in database
  `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves
 `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
 `python app/run.py`

## Project Deployment
1. To run the web app, go into the Terminal and type:
`cd project folder path`
`python app/run.py`

2. Next, go to www.heroku.com and create an account.

3. In the terminal, before anything else, update python using the terminal command `conda update python`

4. Create a virtual environment. 
`python3 -m venv DisasterResponsevenv`
Activate the new environment (Mac/Linux)
`source DisasterResponsevenv/bin/activate`

- Note that you can create the virtual environment inside the project folder. But then you would end up uploading that folder to Heroku unecessarily. Consider creating the virtual environment in the workspace folder. 
- Or alternatively, you can create a .gitignore file inside the project folder so that the virtual enviornment folder gets ignored

5. pip install the libraries needed for the web app. In this case those are flask, pandas, plotly, and gunicorn. 
`pip install flask pandas plotly gunicorn`

6. then check the installation:
`heroku --version`

7. next, log into heroku
`heroku login -i`

8. remove `host='0.0.0.0', port=3001` in app.run() from the run.py file. Instead, use `app.run(debug=True)`.

9. go into the deployment folder:
`cd project folder path`

10. create a procfile:
`touch Procfile`
and put the following in the Procfile
`web gunicorn run:app`

- Note: to deploy the ML model we need to create 2 files. The first one is Procfile (no file extension) in this we will write “web: gunicorn run:app”. The web depicts that this is a web app and gunicorn is the server on which our app will run. The follwoing "run" represents the file name from where the Web App should start. The second part "app" represents the name of the app.

11. Then create a requirements file:
`pip freeze > requirements.txt`

12.Use NLTK with Heroku Python. It can't be simplied installed through requirements.txt. Follow the following steps:
    * nltk.txt needs to present at the root folder
    * Add the modules you want to download like punkt, stopwords as separate row items
    * Change the line ending from windows to UNIX.
        - Changing the line ending is a very important step. Can be easily done through Sublime Text or Notepad++. 
        - In Sublime Text, it can done from the View menu, then Line Endings
    * Add NLTK to requirements.txt.

13. Next, initialize a git repository:
`git init`
`git add .`

14. configure the email and user name, you can use these commands:
`git config --global user.email email@example.com`
`git config --global user.name "my name"`

15. make a commit:
`git commit -m "first commit"`

16. create a uniquely named heroku app. 
`heroku create my-app-name`

17. check that heroku added a remote repository:
`git remote -v`

18. before finally push your local git repository to the remote Heroku repository, you will need the following environment variables (kind of secrets) to send along:
-Set any environment variable to pass along with the push
`heroku config:set SLUGIFY_USES_TEXT_UNIDECODE=yes`
`heroku config:set AIRFLOW_GPL_UNIDECODE=yes`
-Verify the variables
`heroku config`

19. push the app to Heroku:
`git push heroku master`
Go to the link for your web app to see if it's working. 

## Live Demo
https://disaster-response-app-minyan.herokuapp.com

## Additional Issues
- #### Issue1: Pickle serialization: module '__main__' has no attribute 'tokenize'

Reference:

https://docs.arthur.ai/user-guide/explainability.html

https://rebeccabilbro.github.io/module-main-has-no-attribute/

https://github.com/iammowgoud/NLP-fake-news-detection/blob/master/predictionModel.py

