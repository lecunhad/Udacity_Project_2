<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- TABELA DE CONTEÚDOS -->
<details open="open">
  <summary>Summary</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the Project</a>
      <ul>
        <li><a href="#built-with">Frameworks</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Pre Requisites</a></li>
      </ul>
    </li>
    <li><a href="#run">Running</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Reference</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About the project

The objective of this project is to create a machine learning pipeline to categorize disaster events, through messages raised from people asking for help. These messages are sorted into categories such as Security, Fire, Military, etc.The project includes a web app where an emergency worker can input a new message and get classification results. This classification in a quickly way will help the emergency workers to address the needed support more efficiently and also display visualizations about the data. 

### Frameworks

To run the notebook, it is necessary to install the following frameworks:

* [Scikit Learn](https://scikit-learn.org/)
* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)
* [Sqlalchemy](https://www.sqlalchemy.org/)
* [NLTK - punkt, stopwords](https://www.nltk.org//)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [Matplotlib](https://matplotlib.org/)


<!-- GETTING STARTED -->
## Getting Started

The notebook was developed in the environment Udacity Workspace IDE

### Prerequisites

The Python version is 3.7+

<!-- RUN -->
## Running

There are three main folders at the workspace:

<ul>
<li>data
  <ul>
    <li>disaster_categories.csv: dataset including all the categories</li>
    <li>disaster_messages.csv: dataset including all the messages</li>
    <li>process_data.py: ETL pipeline to load, clean, and save the merged datasets into a sql database</li>
    <li>DisasterResponse.db: SQLite database for storage the data
models</li>
   </ul>
<li>models
    <ul>
    <li>train_classifier.py: ML pipeline to train and export the model</li>
    <li>classifier.pkl: ML model created</li>
    </li>
    </ul>
<li>app
<ul>
  <li>run.py: Flask file to run the web application</li>
  <li>templates used for the web application</li>
  </li>
</ul>
</ul>


Run the following commands in the project's root directory to set up your database and model:

Run ETL pipeline for data cleansing and stores in sql database: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

Run ML pipeline to create the model: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Go to app folder (cd app) and run the following command : `python run.py`

Open the browser using the url http://0.0.0.0:3001 (if it doesn't work, find the workspace environmental variables with `env | grep WORK`, and you can open a new browser window and go to the address:`http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN` replacing WORKSPACEID and WORKSPACEDOMAIN with your values)

<!-- ROADMAP -->
## Roadmap

Retrain the model using others classifiers to improve the accuracy.

Use [GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html), a cross validation technique for tuning your model.

Balance the data of the database

<!-- CONTACT -->
## Contato

[Gmail](lecunhad@gmail.com)

[![LinkedIn][linkedin-shield]](https://www.linkedin.com/in/leandro-dias-6a446115a/)


<!-- ACKNOWLEDGEMENTS -->
 
 <!--## Referências-->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

