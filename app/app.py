from flask import Flask, render_template, redirect, request

app = Flask(__name__)

@app.route('/')
def main():
	return redirect('/index')

@app.route('/index', methods=['GET'])
def index():
	return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)