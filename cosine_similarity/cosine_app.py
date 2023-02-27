from flask import Flask,render_template,request
from cosine_nlp_alogrithm import cosine, conv_str_to_vec
app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == 'POST': 
        sentence1_data = request.form["sen1"].lower()
        sentence2_data = request.form["sen2"].lower()

        vect1 = conv_str_to_vec(sentence1_data)
        vect2 = conv_str_to_vec(sentence2_data)
        result = cosine(vect1, vect2)
      
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.debug = True
    app.run()