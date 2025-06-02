# NER_server.py

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.pipelines import pipeline

app = FastAPI(title="Label Studio NER Server")

# ── 1. Load ParsBERT‐NER model and pipeline ──
MODEL_NAME = "HooshvareLab/bert-base-parsbert-ner-uncased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
ner_pipe   = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# ── 2. Health and Setup endpoints (GET/POST) ──
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

# ── 3. Auxiliary for mapping and formatting ──
def map_label(pars_label: str, span_text: str) -> str:
    """
    If ParsBERT outputs 'organization', you want Label Studio to treat it as 'organization'
    (or map to 'Bank_Name' if you prefer). Since you added <Label value="organization"/> in LS,
    we can return pars_label unchanged. Add more rules here if needed.
    """
    return pars_label

def parsbert_to_ls_entities(text: str) -> List[Dict]:
    """
    Run ner_pipe(text) ⇒ a list of dicts:
      {'entity_group':'organization','score':0.975,'word':'بانک ملت','start':…,'end':…}
    For each, if score ≥ threshold, wrap into Label Studio's ResultItem format.
    Return a list of ResultItem‐dicts.
    """
    results = []
    preds = ner_pipe(text)
    if preds is None:
        preds = []
    for ent in preds:
        # Lower the threshold so "بانک ملت" (score ≈0.975) definitely passes
        if not isinstance(ent, dict) or ent.get("score") is None or ent["score"] < 0.3:
            continue
        start = ent["start"]
        end = ent["end"]
        span_text = ent["word"]
        orig_label = ent["entity_group"]
        mapped_label = map_label(orig_label, span_text)
        # Build exactly the dict structure Label Studio expects:
        result_item = {
            "from_name": "label",        # must match <Labels name="label">
            "to_name":   "text",         # must match <Text name="text">
            "type":      "labels",
            "value": {
                "start":  start,
                "end":    end,
                "text":   span_text,
                "labels": [mapped_label]
            }
        }
        results.append(result_item)
    return results

# ── 4. The /predict endpoint (handles Label Studio’s "tasks" list) ──
@app.post("/predict")
async def predict(request: Request):
    """
    Label Studio sends:
      {
        "tasks": [
          {
            "id": <integer>,
            "data": {
              "id": <internal_id>,
              "text": "<the document text>"
            },
            … more metadata …
          },
          … possibly more tasks …
        ],
        "project": "...",
        "label_config": "<View>…</View>",
        "params": { … }
      }
    We must iterate over body["tasks"], run NER on each task["data"]["text"], 
    then return:
      {
        "predictions": [
          { "result": [ ResultItem, ResultItem, … ] },
          { "result": [ … ] },
          …
        ]
      }
    where len(predictions) == len(tasks).
    """
    body = await request.json()
    # Log entire payload for debugging
    print("\n=== /predict RECEIVED ===")
    print(body)

    tasks = body.get("tasks", [])
    predictions = []

    for task in tasks:
        text = task.get("data", {}).get("text", "") or ""
        # Run NER and format results
        result_items = parsbert_to_ls_entities(text)
        predictions.append({"result": result_items})

    response = {"predictions": predictions}
    print("\n=== /predict RESPONSE ===")
    print(response)
    return response

# ── 5. Run Uvicorn if invoked directly ──
if __name__ == "__main__":
    uvicorn.run("NER_server:app", host="127.0.0.1", port=5001, reload=True)
