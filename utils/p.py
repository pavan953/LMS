import sys
import pandas as pd
import joblib
import json

level_classifier = joblib.load('./trained_model.pkl')
label_encoder = joblib.load('./label_encoder.pkl')

for line in sys.stdin:
    if line.strip() == 'hello':
        print('Received hello message from Node.js')
    else:
        new_data_json = line.strip()
        new_data = json.loads(new_data_json)
        new_data_df = pd.DataFrame(new_data, index=[0])
        for column in new_data_df.columns:
            if new_data_df[column].dtype == 'object':
                new_data_df[column] = label_encoder.transform(new_data_df[column])

        predicted_level = level_classifier.predict(new_data_df)
        print("Predicted Level for new data:", predicted_level[0])

        response = {'predicted_level': predicted_level[0]}
        response_json = json.dumps(response)
        print(response_json)

        sys.stdout.flush()
