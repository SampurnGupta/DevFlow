import os
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from openai import OpenAI
import json
from fastapi import HTTPException


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

templates = Jinja2Templates(directory="templates")


async def transcribe_audio_logic(file: UploadFile):
    try:
        audio_bytes = await file.read()

        transcript_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=(file.filename, audio_bytes, file.content_type),
        )
        return transcript_response.text

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # except Exception as e:
    #     return JSONResponse(status_code=500, content={"error": str(e)})
    

async def structure_logic(transcript_text: str, mode: str = "debug"):
    try:
        mode_instruction = {
            "debug": "Focus on identifying bugs and root causes.",
            "feature": "Focus on defining feature scope and requirements.",
            "refactor": "Focus on improving structure and code quality.",
            "architecture": "Focus on system design and scalability."
        }.get(mode, "")

        system_prompt = f"""
You are DevFlow — Developer Intent Structuring Engine.

Mode: {mode}
{mode_instruction}

Return STRICT valid JSON only.
Do NOT wrap JSON in quotes.
Do NOT add markdown formatting.
Do NOT add commentary.

Use this structure:

{{
  "intent": "",
  "context": "",
  "problem_definition": {{
    "current_behavior": "",
    "expected_behavior": ""
  }},
  "constraints": [],
  "edge_cases": [],
  "clarity_gaps_detected": [],
  "refined_prompt": ""
}}
"""
        response = client.chat.completions.create(
            model="gpt-4.1",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript_text},
            ],
        )

        raw_output = response.choices[0].message.content

        try:
            structured_json = json.loads(raw_output)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid JSON returned by model")

        return structured_json

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "running"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    transcript = await transcribe_audio_logic(file)
    return {"transcript": transcript}

# @app.post("/transcribe")
# async def transcribe_audio(file: UploadFile = File(...)):
#     try:
#         audio_bytes = await file.read()

#         transcript = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=(file.filename, audio_bytes, file.content_type),
#             # file=("audio.mp3", audio_bytes, file.content_type),   // audio.mp3 or audio.wav : Is just a placeholder filename.
#         )

#         return {"transcript": transcript.text}

#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})


from pydantic import BaseModel
class TranscriptInput(BaseModel):
    transcript: str
    mode: str = "debug"

# class TranscriptInput(BaseModel):
#     transcript: str

@app.post("/structure")
async def structure_prompt(data: TranscriptInput):
    structured_json = await structure_logic(data.transcript, data.mode)
    return structured_json

# @app.post("/structure")
# async def structure_prompt(data: TranscriptInput):
#     try:
#         system_prompt = """
# You are DevFlow — a Developer Intent Structuring Engine.

# Return STRICT valid JSON only.
# Do NOT wrap JSON in quotes.
# Do NOT add markdown formatting.
# Do NOT add commentary.

# Use this structure:

# {
#   "intent": "",
#   "context": "",
#   "problem_definition": {
#     "current_behavior": "",
#     "expected_behavior": ""
#   },
#   "constraints": [],
#   "edge_cases": [],
#   "clarity_gaps_detected": [],
#   "refined_prompt": ""
# }
# """

#         response = client.chat.completions.create(
#             model="gpt-4.1",
#             temperature=0.2,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": data.transcript},
#             ],
#         )

#         raw_output = response.choices[0].message.content

#         try:
#             structured_json = json.loads(raw_output)
#         except json.JSONDecodeError:
#             raise HTTPException(status_code=500, detail="Invalid JSON returned by model")

#         return structured_json

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_audio(file: UploadFile = File(...), mode: str = Form("debug")):
    transcript = await transcribe_audio_logic(file)
    structured_json = await structure_logic(transcript,mode)

    return {
        "transcript": transcript,
        "structured_output": structured_json
    }