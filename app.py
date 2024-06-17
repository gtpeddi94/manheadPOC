import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from joblib import load
import datetime as dt

app = Flask(__name__)

# Load the trained model
model = load('model_per_head_caboost_split.joblib')

input_features = ['artistName', 'Genre', 'Show Day', 'Show Month',
            'Day of Week Num', 'venue name', 'venue city',
            'venue state','log_attendance']

output_features = ['artistName', 'Genre', 'showDate', 'venue name', 'venue city',
            'venue state','attendance', 'predicted_$_per_head']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'csv_file' in request.files:  # Handle CSV upload
            file = request.files['csv_file']
            df = pd.read_csv(file)

            # Convert 'show_date' to datetime and extract features
            df['showDate'] = pd.to_datetime(df['showDate'])
            df['Show Day'] = df['showDate'].dt.day
            df['Show Month'] = df['showDate'].dt.month
            df['Day of Week Num'] = df['showDate'].dt.weekday

            # Apply log transformation to attendance
            df['log_attendance'] = np.log1p(df['attendance'])

            # Make predictions
            predictions = model.predict(df[input_features])

            # Reverse log transformation for revenue per head
            df['predicted_$_per_head'] = np.expm1(predictions)
            df['predicted_$_per_head'] = np.expm1(predictions).round(2)

            output_df = df[output_features]

            return render_template('index.html', table=output_df.to_html())
        else:  # Handle single prediction form
            # Get input values from the form
            show_date_str = request.form['showDate'] 
            show_date = dt.datetime.strptime(show_date_str, "%Y-%m-%d")
            artist_name = request.form['artistName']
            genre = request.form['Genre']
            venue_name = request.form['venue name']
            venue_city = request.form['venue city']
            venue_state = request.form['venue state']
            attendance = int(request.form['attendance'])

            # Create a DataFrame from input values
            data = pd.DataFrame({
                'artistName': [artist_name],
                'Genre': [genre],
                'Show Day': [show_date.day],
                'Show Month': [show_date.month],
                'Day of Week Num': [show_date.weekday()],
                'venue name': [venue_name],
                'venue city': [venue_city],
                'venue state': [venue_state],
                'log_attendance': np.log1p([attendance])
            })

            # Make prediction
            prediction = model.predict(data)[0]

            # Reverse log transformation
            predicted_revenue_per_head = np.expm1(prediction)
            predicted_revenue_per_head = predicted_revenue_per_head.round(2)
            return render_template('index.html', prediction=predicted_revenue_per_head)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)