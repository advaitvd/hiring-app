from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from modules.feature_extractor import FeatureExtractor
from modules.text_extractor import TextExtractor
from torch.nn.functional import cosine_similarity
import os

UPLOAD_FOLDER = 'data/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

text_extractor = TextExtractor()
feature_extractor = FeatureExtractor()

@app.route('/')
def home():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    text_resume = text_extractor('advait_resume.pdf', file_extension='pdf')
    vector_resume = feature_extractor(text_resume)

    text_requirements = 'Skills:  AWS, Databricks, Leadership, Communication'
    vector_requirements = feature_extractor(text_requirements)
    
    similarity_score = cosine_similarity(vector_resume, vector_requirements).cpu().detach().numpy()[0]

    return render_template('index.html', prediction_text=f'{similarity_score:.4f}')

if __name__=="__main__":
    app.run(debug=True)