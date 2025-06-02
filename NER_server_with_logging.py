# NER_server_with_logging.py

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines import pipeline

app = FastAPI(title="Label Studio NER Server (Logged)")

# 1. Load ParsBERT‐NER
MODEL_NAME = "HooshvareLab/bert-base-parsbert-ner-uncased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
ner_pipe   = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# 2. Pydantic schemas
class NERRequest(BaseModel):
    data: dict

class EntitySpan(BaseModel):
    start:  int
    end:    int
    text:   str
    labels: list[str]

class ResultItem(BaseModel):
    from_name: str
    to_name:   str
    type:      str
    value:     EntitySpan

class Prediction(BaseModel):
    result: list[ResultItem]

class NERResponse(BaseModel):
    predictions: list[Prediction]

# 3. Health & setup endpoints (unchanged)
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/predict/health")
async def predict_health():
    return {"status": "ok"}

@app.get("/predict/setup")
async def predict_setup_get():
    return {}

@app.post("/predict/setup")
async def predict_setup_post():
    return {}

@app.get("/setup")
async def setup_get():
    return {}

@app.post("/setup")
async def setup_post():
    return {}

# 4. Label mapping (if you still want to remap ParsBERT tags to your own)
def map_label(pars_label: str, span_text: str) -> str:
    # Example: if ParsBERT says "organization", map to "Bank_Name" 
    # (or you could leave it as "organization" if you added that to Label Studio).
    if pars_label == "organization":
        return "organization"
    return pars_label  # leave other labels untouched

# 5. Convert ParsBERT output → Label Studio format
def parsbert_to_ls_entities(text: str) -> list[ResultItem]:
    ls_results = []
    preds = ner_pipe(text)
    if preds is None:
        return ls_results
    for ent in preds:
        # Lower the threshold if needed. At 0.5, "بانک ملت" should pass.
        if not isinstance(ent, dict) or "score" not in ent or ent["score"] is None or ent["score"] < 0.3:
            continue
        start     = ent["start"]
        end       = ent["end"]
        span_text = ent["word"]
        orig_label = ent["entity_group"]
        mapped     = map_label(orig_label, span_text)
        item = ResultItem(
            from_name="label",
            to_name="text",
            type="labels",
            value=EntitySpan(
                start=start,
                end=end,
                text=span_text,
                labels=[mapped]
            )
        )
        ls_results.append(item)
    return ls_results

# 6. The /predict endpoint with logging
@app.post("/predict", response_model=NERResponse)
async def predict(request: Request):
    # 6.1 Log incoming JSON
    body = await request.json()
    print("\n=== /predict RECEIVED ===")
    print(body)

    # 6.2 Extract text and run NER
    text = body.get("data", {}).get("text", "") or ""
    ls_entities = parsbert_to_ls_entities(text)

    # 6.3 Build response object
    response = NERResponse(predictions=[Prediction(result=ls_entities)])
    
    # 6.4 Log outgoing JSON
    resp_json = response.dict()
    print("=== /predict RESPONSE ===")
    print(resp_json)

    return response

if __name__ == "__main__":
    uvicorn.run("NER_server_with_logging:app", host="127.0.0.1", port=5001, reload=True)
