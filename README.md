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
        <li><a href="#installation">Instalation</a></li>
      </ul>
    </li>
    <li><a href="#run">Running</a></li>
    <li><a href="#usecases">Use Cases</a></li>
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

The notebook was developed in the environment [Google Colab](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l01c01_introduction_to_colab_and_python.ipynb)

### Prerequisites

The Python version is 3.7+

### Instalation

!pip install -U scikit-learn

(In case you already have the scikit learn installed, you just need to import the library before running the code)


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
    </ul>
    </li>
<li>app
<ul>
  <li>run.py: Flask file to run the web application</li>
  <li>templates used for the web application</li>
</ul>
</li>


## Run the following commands in the project's root directory to set up your database and model:

Run ETL pipeline for data cleansing and stores in sql database: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

Run ML pipeline to create the model: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Go to app folder (cd app) and run the following command : `python run.py`

Open the browser using the url http://0.0.0.0:3001 (if it doesn't work, find the workspace environmental variables with `env | grep WORK`, and you can open a new browser window and go to the address:`http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN` replacing WORKSPACEID and WORKSPACEDOMAIN with your values)


<!-- USECASES -->
## Use Cases

Motivation: Support students who wants to travel abroad to study English, in order to improve a best choice and improve the planner.
Three business questions:
- Can they spend less money at some specific season or period time of the year?
- What are the accommodation information most correlated to price variation?
- Is possible to predict the price for respective acomodation according it features? 


<!-- ROADMAP -->
## Roadmap

Retrain the models to improve predictions

Collect other relevant pieces of information of accommodations and research about the rules of prices formulation.


<!-- CONTACT -->
## Contato

[Gmail](lecunhad@gmail.com)

[![LinkedIn][linkedin-shield]](https://www.linkedin.com/in/leandro-dias-6a446115a/)


<!-- ACKNOWLEDGEMENTS -->
 
 <!--## Referências-->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

