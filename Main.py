from flask import Flask, render_template, redirect, request, send_file
from werkzeug.utils import secure_filename
import os
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("summarizer", "summarizer/summarizer.py")
summarizer = importlib.util.module_from_spec(spec)
sys.modules["summarizer"] = summarizer
spec.loader.exec_module(summarizer)

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route("/")
def home():
    return render_template('index.html')
@app.route('/uploads', methods=['POST', 'GET'])
def uploads():
    file = request.files['file1']
    filename = secure_filename(file.filename)

    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        summarizer.run(filename)
        file_name = "summarize.txt"
        return redirect('/download/' + file_name)
    return render_template('index.html')
# def sum1(filename):
#     f = open('static/uploads/' + filename)
#     r = f.read()
#     i = 0
#     while i != 50000000:
#         i += 1
#     txt = "junk food tastes good and looks good but does not fulfil the healthy calorie requirement of the body. kids and children eating junk food are more prone to the type-2 diabetes. junk food is the easiest way to gain unhealthy weight. it is rich in saturated fat, sodium and bad cholesterol. high sodium and bad cholesterol diet increases blood pressure. high blood pressure also causes clogged arteries, heart attack and strokes. junk food is also one of the main reasons for the increase in obesity nowadays. a healthy diet is very important."
#     f = open('static\\uploads\\summarize.txt', 'w')
#     f.write(txt)
@app.route("/download/<file_name>", methods=['GET'])
def download_file(file_name):
    c = print_content()
    print(c)
    return render_template('convert.html', value=file_name, text=c)

def print_content():
    f = open('static/uploads/summarize2.txt', 'r')
    content=[]
    for i in f:
        content.append(i)
    # print(content)
    return content

@app.route('/return-files/<file_name>')
def return_files_tut(file_name):
    file_path = UPLOAD_FOLDER + '/' + file_name
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
