# Search Engine for Question Hour in Legislative Assembly  

This is a proof of concept search engine application that allows users to ask questions to a ministry in the domain of Legislative Assembly. The application utilizes natural language processing techniques to determine if a similar question has been asked before and provide the corresponding answer.

## Features

- Users can ask questions related to the Legislative Assembly domain.
- The application searches for similar questions in the corpus of previous questions and answers.
- If a similar question is found, the application provides the corresponding answer.
- If the question is not found in the corpus, the application returns "Question not asked before".
- The application takes care of semantic similarity and handles negations.

## Prerequisites

- Python 3.6 or higher
- NLTK library
- Sentence Transformers library
- Flask library

## Installation

1. Clone the repository:

## Steps to be followed:

- Install the required libraries
- Run the application
- Open a web browser and go to `http://localhost:5000` to access the application.
- Enter your question in the search box and click the "Search" button.
- The application will provide the corresponding answer if a similar question has been asked before, or display "Question not asked before" if the question is not found in the corpus.

## Customization

- You can modify the corpus of previous questions and answers by updating the `corpus` variable in the `app.py` file.
- If you want to use a different model for sentence encoding, you can replace the model name in the `model = SentenceTransformer('<model_name>')` line in the `app.py` file.
