from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from pypdf import PdfReader
import torch
from torch.nn.functional import cosine_similarity
from modules.feature_extractor import FeatureExtractor
from modules.text_extractor import TextExtractor
import os

app = Flask(__name__)

app.static_folder = 'templates/static'

UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

feature_extractor = FeatureExtractor()
text_extractor = TextExtractor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file):
    text = text_extractor(file)
    return text

def calculate_similarity(job_features, resume):
    resume_features = feature_extractor(resume)
    similarity = cosine_similarity(job_features, resume_features).item()
    return round(similarity, 4)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get job description from the form
        job_description = request.form['job_description']
        job_features = feature_extractor(job_description)
        ranked_resumes = []

        # Iterate through uploaded resume files
        for file in request.files.getlist('resumes'):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Extract text from the resume PDF
                resume_text = extract_text_from_pdf(file_path)

                # Calculate similarity score
                similarity = calculate_similarity(job_features, resume_text)

                # Add resume and similarity score to the ranked list
                ranked_resumes.append((filename, similarity))
                os.system(f'rm {file_path}')

        # Sort resumes based on similarity score
        ranked_resumes.sort(key=lambda x: x[1], reverse=True)

        return render_template('result.html', ranked_resumes=ranked_resumes)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
