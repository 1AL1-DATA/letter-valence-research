# ruff: noqa: E501
"""Vercel serverless entrypoint: letter-derived financial sentiment demo.

Run locally with:  uvicorn api.index:app --reload
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from api.cheap_tier import NEG_WORDS as CHEAP_KEYWORDS_NEG  # noqa: E402
from api.cheap_tier import POS_WORDS as CHEAP_KEYWORDS_POS  # noqa: E402
from api.cheap_tier import predict as cheap_predict  # noqa: E402
from api.shap_features import top_features as shap_top_features  # noqa: E402
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
  .meter { position:relative; height:18px; border-radius:9px; background:linear-gradient(90deg,#dc2626 0%,#fca5a5 26%,#f3f4f6 50%,#86efac 74%,#16a34a 100%); box-shadow:inset 0 1px 2px rgba(0,0,0,.18); margin:.8rem 0 .15rem; }
  .meter .mid { position:absolute; left:50%; top:0; bottom:0; width:2px; background:rgba(0,0,0,.25); }
  .meter .pin { position:absolute; top:-5px; width:5px; height:28px; background:#111827; border-radius:3px; transform:translateX(-50%); box-shadow:0 1px 3px rgba(0,0,0,.35); transition:left .5s ease; }
  .scale { display:flex; justify-content:space-between; font:12px system-ui; color:#6b7280; }
  .hist { display:flex; align-items:flex-end; justify-content:space-around; height:130px; margin-top:.6rem; padding:0 .25rem; }
  .hcol { display:flex; flex-direction:column; align-items:center; justify-content:flex-end; gap:2px; width:100%; }
  .hbar { width:100%; max-width:64px; height:0; transition:height .5s ease; border-radius:5px 5px 0 0; }
  .hbar.neg { background:var(--bad); } .hbar.pos { background:var(--ok); } .hbar.neu { background:#9ca3af; }
  .hval { font:700 14px system-ui; color:#1d2433; }
  .hlabel { font:600 12px system-ui; color:#6b7280; }
  .shap { margin-top:1rem; }
  .shap-title { font-weight:600; margin-bottom:.4rem; font-size:15px; }
  .shap-note { font-size:12px; color:#6b7280; margin-bottom:.5rem; }
  .shap-row { display:flex; align-items:center; gap:.6rem; font:13px system-ui; margin:.35rem 0; }
  .shap-name { width:180px; text-align:right; color:#374151; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .shap-track { flex:1; display:flex; align-items:center; }
  .shap-bar { height:12px; border-radius:3px; min-width:3px; transition:width .5s ease; }
  .shap-bar.pos { background:var(--ok); } .shap-bar.neg { background:var(--bad); }
  .shap-val { width:72px; font-variant-numeric:tabular-nums; color:#6b7280; }
  .chip { display:inline-block; padding:.15rem .5rem; margin:.15rem .2rem 0 0; border-radius:999px; font:12px system-ui; }
  .chip.pos { background:#dcfce7; color:#15803d; } .chip.neg { background:#fee2e2; color:#b91c1c; }
  .chips-label { font-size:12px; color:#6b7280; margin-top:.5rem; }
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
    <textarea id="text" placeholder="Paste a financial news headline or paragraph here...">Berkshire Hathaway shares rose on Monday to their highest level since Warren Buffett announced his departure as chief executive in May 2025, after his successor Greg Abel began spending the conglomerate's huge cash pile and financial results topped analysts' expectations.</textarea>
    <div class="row">
      <button id="run" onclick="classify()">Analyze sentiment</button>
      <span id="note" style="color:#6b7280;font-size:13px;"></span>
    </div>
  </div>

  <div id="result" class="card">
    <div style="font-weight:700;font-size:16px;margin-bottom:.4rem;">Word-level model <span style="font-weight:400;font-size:13px;color:#6b7280;">(TF-IDF + VADER + keywords)</span></div>
    <div id="cheap-verdict" class="label" style="font-size:1.2rem;"></div>
    <div class="meter"><div class="mid"></div><div class="pin" id="cheap-pin" style="left:50%"></div></div>
    <div class="scale"><span>negative</span><span>neutral</span><span>positive</span></div>
    <div class="hist">
      <div class="hcol"><div class="hbar neg" id="cn"></div><div class="hval" id="cnv"></div><div class="hlabel">Negative</div></div>
      <div class="hcol"><div class="hbar neu" id="cu"></div><div class="hval" id="cuv"></div><div class="hlabel">Neutral</div></div>
      <div class="hcol"><div class="hbar pos" id="cp"></div><div class="hval" id="cpv"></div><div class="hlabel">Positive</div></div>
    </div>

    <hr style="border:none;border-top:1px solid #e5e7eb;margin:1.1rem 0;">

    <div style="font-weight:700;font-size:16px;margin-bottom:.4rem;">Letter-feature model <span style="font-weight:400;font-size:13px;color:#6b7280;">(RandomForest)</span></div>
    <div id="verdict" class="label"></div>
    <div class="meter"><div class="mid"></div><div class="pin" id="letter-pin" style="left:50%"></div></div>
    <div class="scale"><span>negative</span><span>neutral</span><span>positive</span></div>
    <div class="hist">
      <div class="hcol"><div class="hbar neg" id="ln"></div><div class="hval" id="lnv"></div><div class="hlabel">Negative</div></div>
      <div class="hcol"><div class="hbar pos" id="lp"></div><div class="hval" id="lpv"></div><div class="hlabel">Positive</div></div>
    </div>

    <div class="shap">
      <div class="shap-title">Why this prediction — SHAP attribution</div>
      <div class="shap-note">Signed per-word contribution of each letter feature (green = pushes positive, red = pushes negative).</div>
      <div id="shap"></div>
    </div>
  </div>

  <div id="vader-card" class="card" style="display:none;">
    <div style="font-weight:700;font-size:16px;margin-bottom:.4rem;">VADER baseline <span style="font-weight:400;font-size:13px;color:#6b7280;">(lexical rule-based)</span></div>
    <div id="vader-verdict" class="label" style="font-size:1.2rem;"></div>
    <div class="meter"><div class="mid"></div><div class="pin" id="vader-pin" style="left:50%"></div></div>
    <div class="scale"><span>negative</span><span>neutral</span><span>positive</span></div>
    <div class="hist">
      <div class="hcol"><div class="hbar neg" id="vn"></div><div class="hval" id="vnv"></div><div class="hlabel">Negative</div></div>
      <div class="hcol"><div class="hbar neu" id="vu"></div><div class="hval" id="vuv"></div><div class="hlabel">Neutral</div></div>
      <div class="hcol"><div class="hbar pos" id="vp"></div><div class="hval" id="vpv"></div><div class="hlabel">Positive</div></div>
    </div>
    <div class="chips-label">Lexicon words matched:</div>
    <div id="vader-chips"></div>
  </div>

  <p style="color:#6b7280;font-size:13px;line-height:1.6;">
    Word-level model: TF-IDF (1-2 grams) + VADER + keyword lexicons through a 3-class logistic
    regression, the cascade's cheap tier. Letter model: <code>letter_sentiment_rf.pkl</code>
    (RandomForest, 272-dim mean/max/min/std aggregation). VADER is shown as an independent
    lexical baseline. Source:
    <a href="https://github.com/1AL1-DATA/letter-valence-research">github.com/1AL1-DATA/letter-valence-research</a>
  </p>
</main>
<script>
function setHist(barId, valId, p) {
  document.getElementById(barId).style.height = Math.round(p * 80) + 'px';
  document.getElementById(valId).textContent = Math.round(p * 100) + '%';
}
function setPin(id, valence) {
  document.getElementById(id).style.left = ((valence + 1) / 2 * 100) + '%';
}

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

    // ---- letter-feature model ----
    const verdict = document.getElementById('verdict');
    verdict.textContent = d.label.toUpperCase() + '  (' + Math.round(d.confidence * 100) + '% confidence)';
    verdict.className = 'label ' + d.label;
    const lv = d.proba.positive - d.proba.negative;
    setPin('letter-pin', lv);
    setHist('ln', 'lnv', d.proba.negative);
    setHist('lp', 'lpv', d.proba.positive);

    // ---- word-level (TF-IDF + VADER) model ----
    const cv = document.getElementById('cheap-verdict');
    cv.textContent = d.cheap.label.toUpperCase() +
      '  (valence ' + d.cheap.valence.toFixed(3) + ')  ·  ' +
      (d.cheap.decided ? 'decides at |v|≥0.6' : 'below |v|≥0.6 — falls through');
    cv.className = 'label ' + d.cheap.label;
    setPin('cheap-pin', d.cheap.valence);
    setHist('cn', 'cnv', d.cheap.proba.negative);
    setHist('cu', 'cuv', d.cheap.proba.neutral);
    setHist('cp', 'cpv', d.cheap.proba.positive);

    // ---- VADER baseline ----
    document.getElementById('vader-card').style.display = 'block';
    const vv = document.getElementById('vader-verdict');
    vv.textContent = d.vader.label_default.toUpperCase() +
      '  (compound ' + d.vader.compound.toFixed(3) + ')';
    vv.className = 'label ' + d.vader.label_default;
    setPin('vader-pin', Math.max(-1, Math.min(1, d.vader.compound)));
    setHist('vn', 'vnv', d.vader.neg);
    setHist('vu', 'vuv', d.vader.neu);
    setHist('vp', 'vpv', d.vader.pos);

    const chips = [];
    (d.vader.matched_pos || []).forEach(function (w) { chips.push('<span class="chip pos">' + w + '</span>'); });
    (d.vader.matched_neg || []).forEach(function (w) { chips.push('<span class="chip neg">' + w + '</span>'); });
    document.getElementById('vader-chips').innerHTML =
      chips.length ? chips.join('') : '<span style="font-size:12px;color:#9ca3af;">none from the lexicon</span>';

    // ---- SHAP attribution for the letter model ----
    const maxAbs = Math.max.apply(null, (d.shap_features || []).map(function (f) { return Math.abs(f.value); }).concat([1e-9]));
    const ft = document.getElementById('shap');
    ft.innerHTML = (d.shap_features || []).map(function (f) {
      const pct = Math.max(4, Math.round(Math.abs(f.value) / maxAbs * 100));
      const pos = f.value >= 0;
      const align = pos ? '' : ' justify-content:flex-end;';
      return '<div class="shap-row">' +
        '<div class="shap-name" title="' + f.name + '">' + f.name + '</div>' +
        '<div class="shap-track" style="' + align + '">' +
          '<div class="shap-bar ' + (pos ? 'pos' : 'neg') + '" style="width:' + pct + '%"></div>' +
        '</div>' +
        '<div class="shap-val">' + (pos ? '+' : '') + f.value.toFixed(4) + '</div>' +
      '</div>';
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


class CheapResult(BaseModel):
    label: str
    valence: float
    score: float
    proba: dict[str, float]
    decided: bool


class ClassifyResponse(BaseModel):
    text: str
    n_words: int
    label: str
    label_code: int
    proba: dict[str, float]
    confidence: float
    vader: dict
    top_features: list[TopFeature]
    shap_features: list[TopFeature]
    cheap: CheapResult


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
        v["matched_pos"] = sorted(
            set(re.findall(r"[a-z']+", text.lower())) & CHEAP_KEYWORDS_POS
        )
        v["matched_neg"] = sorted(
            set(re.findall(r"[a-z']+", text.lower())) & CHEAP_KEYWORDS_NEG
        )
        c = cheap_predict(text)
        sf = shap_top_features(text)
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
        shap_features=[TopFeature(**f) for f in sf],
        cheap=CheapResult(**c),
    )
