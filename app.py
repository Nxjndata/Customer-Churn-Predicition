from flask import Flask, render_template, request
import numpy as np
import pickle
import json
import os

app = Flask(__name__, template_folder='.')  # Set template folder to root directory

def churn_prediction(tenure, citytier, warehousetohome, gender, hourspendonapp, 
                    numberofdeviceregistered, satisfactionscore, maritalstatus,
                    numberofaddress, complain, orderamounthikefromlastyear, 
                    couponused, ordercount, daysincelastorder, cashbackamount):
    try:
        with open('end_to_end_deployment/models/churn_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)

        with open("end_to_end_deployment/models/columns.json", "r") as f:
            data_columns = json.load(f)['data_columns']

        input_data = [
            tenure, citytier, warehousetohome, gender,
            hourspendonapp, numberofdeviceregistered, satisfactionscore, maritalstatus,
            numberofaddress, complain, orderamounthikefromlastyear, couponused, 
            ordercount, daysincelastorder, cashbackamount
        ]

        input_dict = {
            "tenure": tenure,
            "citytier": citytier,
            "warehousetohome": warehousetohome,
            "gender": gender,
            "hourspendonapp": hourspendonapp,
            "numberofdeviceregistered": numberofdeviceregistered,
            "satisfactionscore": satisfactionscore,
            "maritalstatus": maritalstatus,
            "numberofaddress": numberofaddress,
            "complain": complain,
            "orderamounthikefromlastyear": orderamounthikefromlastyear,
            "couponused": couponused,
            "ordercount": ordercount,
            "daysincelastorder": daysincelastorder,
            "cashbackamount": cashbackamount
        }

        # One-hot encode categorical variables
        for col in data_columns:
            if col in input_dict and isinstance(input_dict[col], str):
                input_dict[col] = input_dict[col].lower().replace(' ', '_')

        input_array = np.zeros(len(data_columns))

        for i, col in enumerate(data_columns):
            if col in input_dict:
                input_array[i] = input_dict[col]
            elif col in input_dict.keys():
                if f"{col}_{input_dict[col]}" in data_columns:
                    input_array[data_columns.index(f"{col}_{input_dict[col]}")] = 1

        output_probab = model.predict_proba([input_array])[0][1]
        return round(output_probab, 4)
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.0  # Return default value on error

@app.route('/', methods=['GET', 'POST'])
def index_page():
    if request.method == 'POST':
        try:
            form_data = [
                request.form.get('Tenure', 0),
                request.form.get('Citytier', 0),
                request.form.get('Warehousetohome', 0),
                request.form.get('Gender', ''),
                request.form.get('Hourspendonapp', 0),
                request.form.get('Numberofdeviceregistered', 0),
                request.form.get('Satisfactionscore', 0),
                request.form.get('Maritalstatus', ''),
                request.form.get('Numberofaddress', 0),
                request.form.get('Complain', 0),
                request.form.get('Orderamounthikefromlastyear', 0),
                request.form.get('Couponused', 0),
                request.form.get('Ordercount', 0),
                request.form.get('Daysincelastorder', 0),
                request.form.get('Cashbackamount', 0)
            ]

            # Convert numeric values
            form_data = [int(i) if str(i).isdigit() else i for i in form_data]

            output_probab = churn_prediction(*form_data)
            pred = "Churn" if output_probab > 0.4 else "Not Churn"

            data = {
                'prediction': pred,
                'predict_probabality': output_probab
            }

            return render_template('result.html', data=data)
        except Exception as e:
            return render_template('error.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
