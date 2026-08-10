# ruff: noqa: E501
"""Vercel serverless entrypoint: letter-derived financial sentiment demo.

Run locally with:  uvicorn api.index:app --reload
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.classify import classify, vader_score  # noqa: E402

app = FastAPI(title="Letter Valence Research", version="1.0.0")

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Letter Valence Research</title>
<style>
  :root { --ink:#1d2433; --accent:#4f46e5; --bg:#f6f7fb; --ok:#16a34a; --bad:#dc2626; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:Georgia,serif; color:var(--ink); background:var(--bg); }
  header { background:linear-gradient(135deg,#1d2433,#4f46e5); color:#fff; padding:3rem 1.5rem; text-align:center; }
  header h1 { margin:0 0 .4rem; font-size:2rem; }
  header p { margin:0 auto; max-width:620px; opacity:.85; line-height:1.5; }
  main { max-width:720px; margin:2rem auto; padding:0 1.25rem 4rem; }
  .card { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:1.25rem; margin-bottom:1.25rem; box-shadow:0 1px 3px rgba(0,0,0,.06); }
  textarea { width:100%; min-height:110px; padding:.8rem; font:15px/1.5 Georgia,serif; border:1px solid #d1d5db; border-radius:8px; resize:vertical; }
  .row { display:flex; gap:.75rem; align-items:center; margin-top:.9rem; flex-wrap:wrap; }
  button { background:var(--accent); color:#fff; border:0; padding:.7rem 1.4rem; border-radius:8px; font:600 15px system-ui; cursor:pointer; }
  button:disabled { opacity:.55; cursor:wait; }
  #result { display:none; }
  #result.show { display:block; }
  .label { font-size:1.4rem; font-weight:bold; }
  .pos { color:var(--ok); } .neg { color:var(--bad); }
  .bar { background:#e5e7eb; border-radius:6px; height:14px; overflow:hidden; margin-top:.35rem; }
  .bar > div { height:100%; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:1.25rem; margin-top:1rem; }
  @media (max-width:560px){ .grid { grid-template-columns:1fr; } }
  table { width:100%; border-collapse:collapse; font:13px system-ui; }
  td, th { padding:.4rem .5rem; border-bottom:1px solid #eee; text-align:left; }
  th { color:#6b7280; font-weight:600; }
  code { background:#f1f3f9; padding:.1rem .35rem; border-radius:4px; font-size:.9em; }
</style>
</head>
<body>
<header>
  <h1>Letter Valence Research</h1>
  <p>A random-forest model trained on 57 letter-derived features of words (alphabet positions,
  phonetics, spectra, gematria-like statistics) to classify the sentiment of financial news.</p>
</header>
<main>
  <div class="card">
    <textarea id="text" placeholder="Paste a financial news headline or paragraph here...">The company announced record profits and strong growth this quarter.</textarea>
    <div class="row">
      <button id="run" onclick="classify()">Analyze sentiment</button>
      <span id="note" style="color:#6b7280;font-size:13px;"></span>
    </div>
  </div>

  <div id="result" class="card">
    <div id="verdict" class="label"></div>
    <div style="margin-top:.9rem;font-size:14px;color:#374151;">
      <div>Positive</div>
      <div class="bar"><div id="bar-pos" style="background:var(--ok);width:50%;"></div></div>
      <div style="margin-top:.6rem;">Negative</div>
      <div class="bar"><div id="bar-neg" style="background:var(--bad);width:50%;"></div></div>
    </div>

    <div class="grid">
      <div>
        <div style="font-weight:600;margin-bottom:.5rem;font-size:15px;">VADER baseline</div>
        <table id="vader"></table>
      </div>
      <div>
        <div style="font-weight:600;margin-bottom:.5rem;font-size:15px;">Top letter features</div>
        <table id="features"></table>
      </div>
    </div>
  </div>

  <p style="color:#6b7280;font-size:13px;line-height:1.6;">
    Letter-feature model: <code>letter_sentiment_rf.pkl</code> (RandomForest, 272-dim
    mean/max/min/std aggregation). VADER is shown as an independent lexical baseline.
    Source: <a href="https://github.com/1AL1-DATA/letter-valence-research">github.com/1AL1-DATA/letter-valence-research</a>
  </p>
</main>
<script>
async function classify() {
  const btn = document.getElementById('run');
  const note = document.getElementById('note');
  const res = document.getElementById('result');
  btn.disabled = true; note.textContent = 'Running...';
  try {
    const r = await fetch('/api/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: document.getElementById('text').value })
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    if (d.error) throw new Error(d.error);

    const verdict = document.getElementById('verdict');
    verdict.textContent = d.label.toUpperCase() + '  (' + Math.round(d.confidence * 100) + '% confidence)';
    verdict.className = 'label ' + d.label;
    document.getElementById('bar-pos').style.width = Math.round(d.proba.positive * 100) + '%';
    document.getElementById('bar-neg').style.width = Math.round(d.proba.negative * 100) + '%';

    const vt = document.getElementById('vader');
    vt.innerHTML = '<tr><th>Compound</th><td>' + d.vader.compound.toFixed(3) + '</td></tr>' +
      '<tr><th>Label</th><td>' + d.vader.label_default + '</td></tr>' +
      '<tr><th>pos / neu / neg</th><td>' + d.vader.pos.toFixed(2) + ' / ' + d.vader.neu.toFixed(2) + ' / ' + d.vader.neg.toFixed(2) + '</td></tr>';

    const ft = document.getElementById('features');
    ft.innerHTML = (d.top_features || []).map(function (f) {
      return '<tr><td>' + f.name + '</td><td style="font-variant-numeric:tabular-nums">' + f.value.toFixed(3) + '</td></tr>';
    }).join('');

    res.className = 'card show';
    note.textContent = d.n_words + ' words analyzed';
  } catch (e) {
    note.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


class ClassifyRequest(BaseModel):
    text: str


class TopFeature(BaseModel):
    name: str
    value: float


class ClassifyResponse(BaseModel):
    text: str
    n_words: int
    label: str
    label_code: int
    proba: dict[str, float]
    confidence: float
    vader: dict
    top_features: list[TopFeature]


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify_endpoint(req: ClassifyRequest) -> ClassifyResponse:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    try:
        r = classify(text, return_proba=True, return_features=True)
        v = vader_score(text)
    except Exception as exc:  # model load / prediction failure
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}") from exc
    top_features = [TopFeature(name=name, value=float(value)) for name, value in r.get("top_features", [])]
    return ClassifyResponse(
        text=r["text"],
        n_words=r["n_words"],
        label=r["label"],
        label_code=r["label_code"],
        proba={k: float(v) for k, v in r["proba"].items()},
        confidence=float(r["confidence"]),
        vader=v,
        top_features=top_features,
    )
