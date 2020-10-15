from flask import Flask, render_template, redirect, request
import pandas as pd

app = Flask(__name__)

@app.route('/')
def main():
	return redirect('/index')

@app.route('/index', methods=['GET'])
def index():
	return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def prediction():
	df = process_inputs()

	comment = df.Comment[0]
	return render_template('prediction.html', variable='A BOT')
   

def process_inputs():
	text_keys = ['Comment', 'Subreddit']
	int_keys = ['Score', 'FlairCount']
	inputs = {}

	for key, val in request.form.items():
		if key in int_keys:
			inputs[key] = int(val)
		else:
			inputs[key] = val

	return pd.DataFrame(inputs, index=[0])

if __name__ == '__main__':
    app.run(debug=True)



def process_inputs():
    '''
    Process input data for the model training.
    '''

    int_keys = ('Pclass', 'SibSp', 'Parch')
    inputs = {}