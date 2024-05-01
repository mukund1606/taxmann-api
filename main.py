from fastapi import FastAPI
from pydantic import BaseModel
import json
from joblib import load

app = FastAPI()
model = load("models/random_forest_model.joblib")
vectorizer = load("models/tfidf_vectorizer.joblib")


def predict(input_text):
    input_vector = vectorizer.transform([input_text])
    predictions = model.predict(input_vector)
    return json.dumps(predictions.tolist())


@app.get("/")
def hello_world():
    return {"message": "Hello World"}


class Data(BaseModel):
    text: str


@app.post("/predict")
def hello_world(data: Data):
    prediction = predict(data.text)
    return {"prediction": prediction}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
