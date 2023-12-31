from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import easyocr
import nltk
from nltk.corpus import stopwords
import os

app = Flask(__name__)

# Set up SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Download NLTK stopwords
nltk.download('stopwords')

# Set the upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def calculate_semantic_similarity(sentence1, sentence2):
    embeddings = model.encode([sentence1, sentence2], convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return cosine_score.item()

def get_text_from_image(image_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    student_text = ' '.join([result[1] for result in results])
    return student_text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

@app.route('/', methods=['GET', 'POST'])
def index():
    similarity = 0.0  # Default similarity value
    image_filename = 'default_image.jpg'  # Default image filename

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part', similarity=similarity, image_filename=image_filename)

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file', similarity=similarity, image_filename=image_filename)

        if file:
            filename = secure_filename(file.filename)
            image_filename = filename

            # Save the file to the upload folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            actual_answer = """Electric vehicles are poised to dominate the automotive sector. With escalating climate concerns, governments will likely impose restrictions on gasoline car production. Increasing fuel prices will challenge the affordability of operating conventional vehicles. Advancements in battery tech will enhance electric cars' range and affordability. Ultimately, electric cars are expected to replace the majority of gasoline-powered vehicles."""

            student_text = get_text_from_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            actual_answer_processed = remove_stopwords(actual_answer)
            student_text_processed = remove_stopwords(student_text)

            similarity = calculate_semantic_similarity(actual_answer_processed, student_text_processed)

    return render_template('index.html', similarity=similarity, image_filename=image_filename)

if __name__ == '__main__':
    app.run(debug=True)
