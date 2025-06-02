# NER_server.py

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines import pipeline

app = FastAPI(title="Label Studio NER Server")

# ── 1. Load ParsBERT‐NER (correct, public checkpoint) ──
MODEL_NAME = "HooshvareLab/bert-base-parsbert-ner-uncased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

ner_pipe = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# ── 2. Pydantic schemas ──
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

# ── 3. Health check endpoints ──
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/predict/health")
async def predict_health():
    return {"status": "ok"}

@app.post("/predict/setup")
async def predict_setup_post():
    """
    Label Studio may also do POST /predict/setup. Return same empty JSON.
    """
    return {}

# (Optional) Also catch top-level /setup if LS tries that:
@app.get("/setup")
async def setup_get():
    return {}

@app.post("/setup")
async def setup_post():
    return {}


# ── 5. Convert ParsBERT‐NER output into Label Studio format ──
def parsbert_to_ls_entities(text: str) -> list[ResultItem]:
    ls_results: list[ResultItem] = []
    preds = ner_pipe(text)
    if preds is None:
        return []

    for ent in preds:
        if not isinstance(ent, dict):
            continue
        if not all(k in ent for k in ("start", "end", "word", "entity_group")):
            continue

        start     = ent["start"]
        end       = ent["end"]
        span_text = ent["word"]
        label     = ent["entity_group"]

        item = ResultItem(
            from_name="label",   # must match <Labels name="label">
            to_name="text",      # must match <Text name="text">
            type="labels",
            value=EntitySpan(
                start=start,
                end=end,
                text=span_text,
                labels=[label]
            )
        )
        ls_results.append(item)

    return ls_results

# ── 6. The /predict endpoint that Label Studio will POST to ──
@app.post("/predict", response_model=NERResponse)
async def predict(request: Request):
    """
    Expects JSON: { "data": { "text": "<some Persian text>" }, "id": ... }
    Returns: { "predictions": [ { "result": [ ... ] } ] }
    """
    body = await request.json()
    text = body.get("data", {}).get("text", "") or ""
    ls_entities = parsbert_to_ls_entities(text)
    return NERResponse(predictions=[Prediction(result=ls_entities)])


if __name__ == "__main__":
    # Run with: python NER_server.py
    uvicorn.run("NER_server:app", host="127.0.0.1", port=5001, reload=True)
