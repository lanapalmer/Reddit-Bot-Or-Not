from flask import Flask, render_template, redirect, request
import pandas as pd
import pickle

# load model
model = pickle.load(open('models/dummyModel.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def main():
	return redirect('/index')

@app.route('/index', methods=['GET'])
def index():
	return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def prediction():
	scaled = process_inputs()

	prediction = model.predict(scaled)

	if prediction == 0:
		result = "Not a Bot"
	else:
		result = "A Bot" 


	return render_template('prediction.html', variable=result)
   

def process_inputs():
	text_keys = ['Comment', 'Subreddit']
	int_keys = ['Score', 'FlairCount']
	inputs = {}

	for key, val in request.form.items():
		if key in int_keys:
			inputs[key] = int(val)
		else:
			inputs[key] = val

	df = pd.DataFrame(inputs, index=[0])

	#Feature Engineering
    #Replace Special Characters
	df['CommentClean'] = df['Comment'].str.replace('[^a-zA-Z]', ' ')
	df['CommentClean'] = df['CommentClean'].str.lower()
	df['CommentLength'] = df['CommentClean'].str.split().str.len()

	#Average Word Length
	df['CommentCharacters'] = df['CommentClean'].str.len()
	df['AvgWordLength'] = df['CommentCharacters'] / df['CommentLength']
	
	#Drop Columns
	df2 = df.drop(['Comment', 'CommentClean', 'Subreddit'], axis=1)

	#Scale
	return scaler.transform(df2)


if __name__ == '__main__':
    app.run(debug=True)
