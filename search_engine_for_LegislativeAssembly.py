import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet
from flask import Flask, render_template, request

app = Flask(__name__)

# Corpus of previous questions and answers
# Considered 2 ministries- Ministry of finance and Education 
corpus = [
    ("What steps is the government taking to address the issue of tax evasion and increase revenue collection?", "The government is implementing various measures such as stricter auditing processes, digitalization of tax systems, and increased penalties for tax evasion. These efforts have resulted in a significant increase in revenue collection over the past year."),
    ("Could provide an update on the progress of the government's economic reform agenda?", "The government has made substantial progress in its economic reform agenda. We have implemented policies to stimulate investment, streamline regulations, and promote ease of doing business. As a result, we have seen improvements in key economic indicators such as GDP growth, job creation, and foreign direct investment."),
    ("What measures are being taken to improve the quality of education in rural areas?", "The government has launched several initiatives to enhance the quality of education in rural areas. These include setting up additional schools, providing teacher training programs, distributing educational resources, and implementing digital learning platforms. These efforts aim to ensure that students in rural areas have access to quality education on par with their urban counterparts."),
    ("Can the Minister of Education provide an update on the government's efforts to promote skill development programs for students?", "The government has been actively promoting skill development programs to equip students with the necessary skills for employment. We have established vocational training centers, collaborated with industries to offer apprenticeship programs, and introduced skill-oriented courses in schools and colleges. These initiatives are aimed at fostering a skilled workforce and reducing unemployment among the youth.")
]
#1. What measures has the government implemented to tackle the problem of tax evasion and enhance revenue generation?
# How is the government actively combating tax evasion and striving to boost revenue collection?
# What strategies or initiatives has the government adopted to curb tax evasion and augment the amount of revenue collected?
# Can the government provide an update on its efforts to counter tax evasion and achieve higher revenue inflows?

#2. Can the Minister of Finance provide a comprehensive update on the current status and achievements of the government's economic reform agenda?
# What significant milestones and outcomes have been accomplished thus far in the government's economic reform agenda, as reported by the Minister of Finance?
# Could the Minister please share the latest developments and results related to the government's ongoing economic reform agenda?
# Can the Minister provide an overview of the progress made by the government in implementing its economic reform agenda and highlight any notable achievements?

#3. What initiatives or steps has the government implemented to enhance the quality of education in rural areas and bridge the urban-rural education gap?
# Could the Minister provide an overview of the specific measures being taken to uplift the quality of education in rural areas and ensure equitable access to educational resources?
# What strategies or programs are being employed by the government to improve educational standards in rural areas and ensure that students receive a quality education?
# Could you elaborate on the initiatives being pursued to raise the overall quality of education in underserved rural areas?

#4. Could the Minister of Education provide the latest update on the government's initiatives and progress in promoting skill development programs for students?
# What recent steps or actions has the government taken to foster skill development among students, and could the Minister provide an update on their effectiveness?
# Can the Minister provide an overview of the government's ongoing efforts to enhance skill development opportunities for students and highlight any noteworthy achievements or milestones?
# What specific measures, collaborations, or policies have been implemented by the government to integrate skill development programs into the education system, and could the Minister provide an update on their implementation and impact?
# Requesting the Minister of Education to share the most recent information on how the government is actively encouraging skill development programs for students, along with any notable outcomes or success stories.

# Preprocess the corpus
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
preprocessed_corpus = []

for question, _ in corpus:
    # Tokenize and lemmatize the question
    tokens = [lemmatizer.lemmatize(token.lower()) for token in word_tokenize(question) if token.isalnum()]
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stop_words]
    preprocessed_corpus.append(' '.join(filtered_tokens))

# Use Sentence Transformers for encoding and similarity computation
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to preprocess and encode the input text
def encode_text(text):
    return model.encode([text])

# Function to calculate similarity score between two encoded texts
def calculate_similarity(encoded_text1, encoded_text2):
    return util.cos_sim(encoded_text1, encoded_text2)

# Function to check if two words have opposite meanings
def have_opposite_meanings(word1, word2):
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    
    for synset1 in synsets1:
        for synset2 in synsets2:
            if synset1.lemmas() and synset2.lemmas():
                lemma1 = synset1.lemmas()[0]
                lemma2 = synset2.lemmas()[0]
                if lemma1.antonyms() and lemma1.antonyms()[0] == lemma2 or lemma2.antonyms() and lemma2.antonyms()[0] == lemma1:
                    return True
    return False

# Function to search for a similar question and provide the corresponding answer
def search_question(query):
    # Preprocess and encode the query
    query = ' '.join([lemmatizer.lemmatize(token.lower()) for token in word_tokenize(query) if token.isalnum()])
    query_encoded = encode_text(query)

    # Calculate similarity scores between the query and each question in the corpus
    similarity_scores = []
    for preprocessed_question in preprocessed_corpus:
        question_encoded = encode_text(preprocessed_question)
        similarity_scores.append(calculate_similarity(query_encoded, question_encoded))

    # Find the index of the most similar question
    most_similar_index = max(range(len(similarity_scores)), key=similarity_scores.__getitem__)

    # Threshold to determine if the question is similar or not
    similarity_threshold = 0.7

    if similarity_scores[most_similar_index] >= similarity_threshold:
        # Check for opposite meanings
        query_tokens = [lemmatizer.lemmatize(token.lower()) for token in word_tokenize(query) if token.isalnum()]
        question_tokens = [lemmatizer.lemmatize(token.lower()) for token in word_tokenize(corpus[most_similar_index][0]) if token.isalnum()]
        
        # Check for opposite meanings between query and question tokens
        has_opposite_meanings = any(have_opposite_meanings(query_token, question_token) for query_token in query_tokens for question_token in question_tokens)
        
        if not has_opposite_meanings:
            return corpus[most_similar_index][1]  # Return the corresponding answer

    return "Question not asked before."


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    response = search_question(query)
    return render_template('index.html', query=query, response=response)


if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    app.run(debug=True)
