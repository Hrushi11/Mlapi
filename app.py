import socket
from pred import Mlapi
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

api = Mlapi()

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/api", methods = ['POST', 'GET'])
def api_funct():
    if request.method == 'GET':
        return f"The URL '/api' is accessed directly. Try going to route to submit form"
        # return render_template("error.html")    
    if request.method == 'POST':
        form_data = request.form
        pred = api.url_pred(form_data["url"])
        return render_template("res.html", pred=pred)

@app.route("/<string:null_https_url>")
def route_api(null_https_url):
    # api.img_gen(null_https_url)
    pred = api.make_pred(null_https_url)
    # pred = api.gen_https_url(null_https_url)

    return jsonify(pred)


if __name__ == "__main__":
    socket.getaddrinfo('localhost', 8080)
    app.run(debug=True)