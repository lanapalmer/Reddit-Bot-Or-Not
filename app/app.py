#Load packages
from flask import Flask, render_template, redirect, request
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# load model and dependencies
cv = pickle.load(open('models/cv.pkl','rb'))
tf = pickle.load(open('models/tf.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))
model = pickle.load(open('models/Logistic_Regression_Model.pkl','rb'))
bsr = pd.read_csv('models/sr.csv')

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
	probability = model.predict_proba(scaled)

	nb_prob = (probability[0][0] * 100).round(2)

	b_prob = (probability[0][1] * 100).round(2)

	if prediction == 1:
		result = "A Bot"
	else:
		result = "Not a Bot"


	return render_template('prediction.html', res_var=result, nb_prob_var= nb_prob, b_prob_var=b_prob)
   

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
	
	#Get dummies for Subreddit
	subreddit = df.iloc[0]['Subreddit']

	if subreddit in bsr.columns.values:
		bsr[subreddit] = 1

	#Join dummies with df
	df2 = pd.concat([df, bsr], axis=1)

	#Count Vectorizer
	count_vector = cv.transform(df2['CommentClean'])

	#TFIDF
	TFIDF = tf.transform(count_vector)

	#Create dataframe of features
	cv_df = pd.DataFrame(TFIDF.toarray(), columns=cv.get_feature_names()).add_prefix('Counts_')

	#Join dataframes
	df2 = df2.join(cv_df)

	#Drop Columns
	df2 = df2.drop(['Comment', 'CommentClean', 'Subreddit'], axis=1)

	#Scale
	return scaler.transform(df2)


if __name__ == '__main__':
    app.run(debug=True)
