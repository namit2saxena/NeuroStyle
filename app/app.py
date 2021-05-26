# render_template is to render the HTML page
#request is to handle the requests
from flask import Flask, render_template, request
import os
import cv2, PIL
# style_transfer brings the run_module module
import style_transfer

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FODLER = 'static/processed'
IMG_CNT = 0

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FODLER'] = PROCESSED_FODLER

basedir = os.path.abspath(os.path.dirname(__file__))

@app.route('/', methods=['POST','GET'])
def hello_world():
	request_type_str = request.method
	if request_type_str == 'GET':
		return render_template('index.html', page_name="NeuroStyle", is_home=True)
	else:
		file1 = request.files.get('file1', None)
		file2 = request.files.get('file2', None)

		epochs_cnt = request.form.get('epochs_cnt', type=int)
		if epochs_cnt is None or epochs_cnt>20:
			epochs_cnt = 10

		if file1 and file2:
			global IMG_CNT
			IMG_CNT = IMG_CNT+1

			_, file1_extension = os.path.splitext(file1.filename)
			_, file2_extension = os.path.splitext(file2.filename)

			file1.filename = r"content_{0}.{1}".format(IMG_CNT, file1_extension)
			file2.filename = r"style_{0}.{1}".format(IMG_CNT, file2_extension)

			file1_path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
			file2_path = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)

			
			file1.save(file1_path)
			file2.save(file2_path)

			# reading images to resize them
			img1 = cv2.imread(file1_path)
			img2 = cv2.imread(file2_path)
			# resizing images
			img1 = cv2.resize(img1, (500, 300))
			img2 = cv2.resize(img2, (500, 300))

			img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
			PIL.Image.fromarray(img1).save(file1_path)

			img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
			PIL.Image.fromarray(img2).save(file2_path)

			style_transfer.run_model(epochs_cnt, file1_path, file2_path, app.config['PROCESSED_FODLER'], IMG_CNT)
			return render_template('index.html', page_name = "comparison", is_home=False, 
				f1_name=r"uploads/{0}".format(file1.filename), f2_name=r"uploads/{0}".format(file2.filename),
				f3_name=r"processed/converted_{0}.jpg".format(IMG_CNT))
		else:
			return render_template('index.html', page_name = "NeuroStyle - Compare", is_home=False, not_complete=True)

		