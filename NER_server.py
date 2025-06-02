# NER_server.py

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines import pipeline

app = FastAPI(title="Label Studio NER Server")

# ── 2.1 Load the correct ParsBERT‐NER model and tokenizer ──
MODEL_NAME = "HooshvareLab/bert-base-parsbert-ner-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# We want Aggregation so that multi‐word entities (e.g. "بانک ملت") come back as one span
ner_pipe = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"  
)

# ── 2.2 Define Pydantic schemas matching Label Studio’s expected I/O ──
class NERRequest(BaseModel):
    data: dict  # Label Studio sends {"data": {"text": "..."}}

class EntitySpan(BaseModel):
    start: int
    end:   int
    text:  str
    labels: list[str]

class ResultItem(BaseModel):
    from_name: str  # must match <Labels name="label">
    to_name:   str  # must match <Text name="text">
    type:      str  # always "labels"
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


# ── 2.3 Convert ParsBERT‐NER output into Label Studio’s “predictions” format ──
def parsbert_to_ls_entities(text: str) -> list[ResultItem]:
    """
    Call ner_pipe(text) → a list of dicts like:
      {'entity_group':'ORG', 'score':0.98, 'word':'بانک ملت', 'start':10, 'end':18, ...}
    Filter by score ≥ 0.5, then wrap into ResultItem.
    """
    ls_results: list[ResultItem] = []
    preds = ner_pipe(text) or []

    for ent in preds:
        if not isinstance(ent, dict) or ent.get("score", 0) < 0.5:
            continue

        start = ent["start"]
        end   = ent["end"]
        span_text = ent["word"]
        label = ent["entity_group"]

        item = ResultItem(
            from_name="label",   # must exactly match your <Labels name="label">
            to_name="text",      # must match your <Text name="text">
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


# ── 2.4 The /predict endpoint Label Studio will POST to ──
@app.post("/predict", response_model=NERResponse)
async def predict(request: Request):
    """
    Expects JSON: { "data": { "text": "<some Persian sentence>" }, "id": ... }
    Returns:
      {
        "predictions": [
          {
            "result": [
              {
                "from_name":"label","to_name":"text","type":"labels",
                "value":{
                  "start":10,"end":18,
                  "text":"بانک ملت",
                  "labels":["ORG"]
                }
              },
              ...
            ]
          }
        ]
      }
    """
    body = await request.json()
    text = body.get("data", {}).get("text", "") or ""
    ls_entities = parsbert_to_ls_entities(text)
    return NERResponse(predictions=[Prediction(result=ls_entities)])


if __name__ == "__main__":
    # Run with: python NER_server.py
    uvicorn.run("NER_server:app", host="127.0.0.1", port=5001, reload=True)
