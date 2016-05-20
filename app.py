from flask import Flask, render_template, request, redirect, send_file
from werkzeug import secure_filename
import socket
from myOCR import save_segmented_img
import uuid



my_ip = socket.gethostbyname(socket.gethostname())
app = Flask(__name__)

@app.route('/')
def upload_file_html():
    return render_template('upload.html', ip=my_ip)
	
@app.route('/uploader', methods = ['POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      path = "tempfiles/" + str(uuid.uuid4())
      f.save(path)
      im_path, table = save_segmented_img(path)
      table = (table.to_html(header=False, index=False, classes=['u-full-width'])
                    .replace('border="1"','border="0"'))
      
      return render_template("result.html", path="/" + im_path, dataframe=table)
		
if __name__ == '__main__':
   app.run('0.0.0.0', debug = True, port=5000)