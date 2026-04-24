import os
import json
import numpy as np
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import joblib
import spacy

load_dotenv()

# ── Filesystem Persistence ─────────────────────────────────────────────────
DATA_DIR = "data"
MACROS_FILE = os.path.join(DATA_DIR, "macros.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")

def save_macros():
    with open(MACROS_FILE, "r+") if os.path.exists(MACROS_FILE) else open(MACROS_FILE, "w") as f:
        json.dump(macros, f, indent=2)

def save_sessions():
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)

# ── In-process stores ──────────────────────────────────────────────────────
# sessions: session_id → list of memory entries
sessions: dict[str, list] = {}
# macros:   name → { keywords: list[str], command: str, mode: str }
macros: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global svc_pipeline, nlp, label_map, sessions, macros
    
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print("[DevFlow] Loading NLP models…")
    svc_pipeline = joblib.load("devflow_artefacts/svc_pipeline.pkl")
    nlp = spacy.load("devflow_artefacts/spacy_ner")
    
    # Load label map
    with open("devflow_artefacts/label_maps.json", "r") as f:
        data = json.load(f)
        label_map = data.get("id2intent", {})
        
    # Load persisted data
    if os.path.exists(MACROS_FILE):
        try:
            with open(MACROS_FILE, "r") as f:
                macros.update(json.load(f))
            print(f"[DevFlow] Loaded {len(macros)} macros.")
        except Exception as e:
            print(f"[DevFlow] Error loading macros: {e}")
            
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r") as f:
                sessions.update(json.load(f))
            print(f"[DevFlow] Loaded {len(sessions)} active sessions.")
        except Exception as e:
            print(f"[DevFlow] Error loading sessions: {e}")
    
    print("[DevFlow] Ready.")
    yield


app = FastAPI(lifespan=lifespan)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
templates = Jinja2Templates(directory="templates")

# ── Pydantic models ────────────────────────────────────────────────────────

class TranscriptInput(BaseModel):
    transcript: str
    mode: str = "auto"
    session_id: Optional[str] = None


class MacroCreate(BaseModel):
    name: str
    keywords: list[str]
    command: str
    mode: str = "auto"


# ── Mode instructions ──────────────────────────────────────────────────────
MODE_INSTRUCTIONS: dict[str, str] = {
    "auto":     "Identify the developer's intent and mode automatically from the transcript.",
    "debug":    "Focus on identifying bugs, error root causes, and reproduction steps.",
    "generate": "Focus on code generation — what needs to be created, its structure and behaviour.",
    "refactor": "Focus on improving structure, readability, and code quality without changing behaviour.",
    "explain":  "Focus on clearly explaining what the code, concept, or system does.",
    "scaffold": "Focus on project scaffolding — file structure, boilerplate, and initial setup.",
    "test":     "Focus on test case design, edge-case coverage, and testing strategy.",
    "document": "Focus on documentation — what needs to be documented, format, and level of detail.",
}

# ── NLP Pipeline ───────────────────────────────────────────────────────────

# Filler words stripped during normalization
_FILLER_WORDS = {
    "um", "uh", "like", "you know", "so", "basically", "literally",
    "actually", "hmm", "er", "ah", "right", "okay", "ok",
}

def _normalize(text: str) -> str:
    """Strip leading/trailing filler words and collapse extra whitespace."""
    import re
    tokens = text.strip().split()
    # Drop leading filler tokens
    while tokens and tokens[0].lower().rstrip(',') in _FILLER_WORDS:
        tokens.pop(0)
    normalized = ' '.join(tokens)
    # Collapse runs of whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def _structure_entities(doc) -> dict:
    """
    Map SpaCy entity labels to structured slots:
      language, framework, error_type, component
    Unmapped labels are folded into 'component' as a last resort.
    """
    slots: dict[str, str | None] = {
        "language": None,
        "framework": None,
        "error_type": None,
        "component": None,
    }
    # Labels that map directly to component if nothing better fits
    _component_labels = {"ORG", "PRODUCT", "GPE", "PERSON", "WORK_OF_ART", "FAC"}

    for ent in doc.ents:
        label = ent.label_.upper()
        val = ent.text
        if label in ("LANG", "LANGUAGE"):
            slots["language"] = val
        elif label == "FRAMEWORK":
            slots["framework"] = val
        elif label == "ERROR":
            slots["error_type"] = val
        elif label in _component_labels and slots["component"] is None:
            slots["component"] = val

    return slots


def run_local_nlp(text: str) -> dict:
    """Run LinearSVC intent classification and SpaCy NER. Models are pre-loaded."""
    raw_utterance = text
    normalized = _normalize(text)
    analysis_text = normalized or text  # fall back to original if blank after normalizing

    # ── Intent classification ─────────────────────────────────────────────
    raw_id = str(svc_pipeline.predict([analysis_text])[0])
    intent = label_map.get(raw_id, raw_id)

    # Probability vector — prefer predict_proba; fall back to softmax(decision_function)
    try:
        proba = svc_pipeline.predict_proba([analysis_text])[0]
    except (AttributeError, Exception):
        raw_df = svc_pipeline.decision_function([analysis_text])[0]
        exp_s = np.exp(raw_df - np.max(raw_df))
        proba = exp_s / exp_s.sum()

    top_confidence = round(float(np.max(proba)), 4)
    low_confidence = top_confidence < 0.50

    # Build ranked list — always top 3 intents by confidence
    n_classes = len(proba)
    all_intents = sorted(
        [
            {
                "intent": label_map.get(str(i), str(i)),
                "confidence": round(float(proba[i]), 4),
            }
            for i in range(n_classes)
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )[:3]

    # ── Named entity recognition ──────────────────────────────────────────
    doc = nlp(analysis_text)
    raw_entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    structured_entities = _structure_entities(doc)

    return {
        # Legacy fields (still used by session timeline / compare panel)
        "svc_intent": intent,
        "svc_confidence": round(top_confidence * 100, 1),
        "ner_entities": raw_entities,
        # Rich NLP analysis fields
        "analysis": {
            "raw_utterance": raw_utterance,
            "normalized": analysis_text,
            "intent": intent,
            "entities": structured_entities,
        },
        "all_intents": all_intents,
    }


# ── Macro matching ─────────────────────────────────────────────────────────

def apply_macros(text: str) -> tuple[str, Optional[str]]:
    """
    Scan transcript for any macro keyword match.
    If found, append the macro's command to the transcript (first match wins).
    Returns (augmented_text, macro_name | None).
    """
    lower = text.lower()
    for name, macro in macros.items():
        for kw in macro["keywords"]:
            if kw.lower() in lower:
                augmented = f"{text}\n\n[MACRO CONTEXT — {name}]\n{macro['command']}"
                return augmented, name
    return text, None


# ── GPT Structuring ────────────────────────────────────────────────────────

async def structure_logic(
    transcript: str,
    mode: str = "auto",
    session_history: Optional[list] = None,
) -> dict:
    """
    Call GPT-4.1 to produce structured JSON.
    Injects the last 3 session entries as conversational context so the model
    treats follow-on queries as continuations, not standalone requests.
    """
    mode_instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["auto"])

    system_prompt = f"""You are DevFlow — Developer Intent Structuring Engine.

Mode: {mode}
{mode_instruction}

IMPORTANT: Return STRICT valid JSON only — no markdown fences, no prose, no wrapping quotes.

Use this exact structure:
{{
  "intent": "",
  "mode_identified": "",
  "context": "",
  "problem_definition": {{
    "current_behavior": "",
    "expected_behavior": ""
  }},
  "constraints": [],
  "edge_cases": [],
  "clarity_gaps_detected": [],
  "refined_prompt": ""
}}"""

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # Inject last 3 session entries as prior conversational turns
    if session_history:
        for entry in session_history[-3:]:
            messages.append({"role": "user", "content": entry["transcript"]})
            messages.append({"role": "assistant", "content": json.dumps(entry["gpt_result"])})

    messages.append({"role": "user", "content": transcript})

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            temperature=0.2,
            messages=messages,
        )
        raw = response.choices[0].message.content
        return json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Core processing (shared by /structure and /process) ───────────────────

async def process_transcript(
    transcript: str,
    mode: str,
    session_id: Optional[str],
) -> dict:
    session_history = sessions.get(session_id, []) if session_id else []

    augmented, macro_name = apply_macros(transcript)
    # Pass augmented query to SVC and SpaCy
    nlp_result = run_local_nlp(augmented)
    gpt_result = await structure_logic(augmented, mode, session_history)

    if session_id:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transcript": augmented,  # Store augmented version in history
            "mode": mode,
            "macro_triggered": macro_name,
            "nlp_result": nlp_result,
            "gpt_result": gpt_result,
        }
        bucket = sessions.setdefault(session_id, [])
        bucket.append(entry)
        if len(bucket) > 50:
            sessions[session_id] = bucket[-50:]
        save_sessions()

    return {
        "transcript": augmented,  # Display augmented version in UI
        "macro_triggered": macro_name,
        "nlp_result": nlp_result,
        "gpt_result": gpt_result,
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "running"}


@app.post("/structure")
async def structure_endpoint(data: TranscriptInput):
    """Text → local NLP + GPT structure (with session context)."""
    return await process_transcript(data.transcript, data.mode, data.session_id)


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    session_id: str = Form(None),
):
    """Audio → Whisper transcription → local NLP + GPT structure (with session context)."""
    try:
        audio_bytes = await file.read()
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=(file.filename, audio_bytes, file.content_type),
        )
        transcript = resp.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return await process_transcript(transcript, mode, session_id)


# ── Session routes ─────────────────────────────────────────────────────────

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    return {"session_id": session_id, "history": sessions.get(session_id, [])}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    save_sessions()
    return {"status": "cleared"}


# ── Macro routes ───────────────────────────────────────────────────────────

@app.get("/macros")
async def get_macros():
    return macros


@app.post("/macros")
async def create_macro(macro: MacroCreate):
    macros[macro.name] = {
        "keywords": macro.keywords,
        "command": macro.command,
        "mode": macro.mode,
    }
    save_macros()
    return {"status": "created", "name": macro.name}


@app.delete("/macros/{name}")
async def delete_macro(name: str):
    if name not in macros:
        raise HTTPException(status_code=404, detail="Macro not found")
    del macros[name]
    save_macros()
    return {"status": "deleted"}