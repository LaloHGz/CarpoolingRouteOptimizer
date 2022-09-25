import os

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
from script import process_csv

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")

def page2():
    return render_template("page2.html")

@app.route('/result',methods=['POST', 'GET'])
def result():
    output = request.form.to_dict()
    name = output["name"]
    name += " /Prueba"
    print(output)    
    
    # return render_template('page2.html')

    return render_template('index.html', name = name)
    
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            new_filename = f'{filename.split(".")[0]}_{str(datetime.now())}.csv'
            save_location = os.path.join('input', new_filename)
            file.save(save_location)

            output_file = process_csv(save_location)
            #return send_from_directory('output', output_file)
            return redirect(url_for('download'))

    return render_template('upload.html')
    
if __name__ == "__main__":
    app.run(debug=True, port=5004)
    
    
    