import pickle
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

def predict(json_data:dict) -> dict:
    input_data_df = pd.DataFrame([json_data]) 

    with open("src/customer_churn_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("src/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    # pré-processando os dados para adequar ao formato necessário
    for column, encoder in encoders.items():
        input_data_df[column] = encoder.transform(input_data_df[column])

    prediction_probs = model.predict_proba(input_data_df)[0]
    return {"Churn": prediction_probs[1], "No Churn": prediction_probs[0]}

@app.post("/processar")
def processar(data:dict):
    results = predict(data)
    return results
