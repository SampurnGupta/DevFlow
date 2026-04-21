# DevFlow

**Speak your thoughts. Structure your intent.**

DevFlow transforms developer speech into structured, LLM-ready JSON prompts. It bridges the gap between raw thinking and precise AI interaction using a hybrid local/cloud NLP pipeline.

---

## ✨ Features

- **🎙 Multimodal Input**: Seamless voice-to-text transcription using OpenAI Whisper.
- **🧠 Hybrid Intelligence**:
    - **Local NLP**: Real-time intent classification (LinearSVC) and entity extraction (SpaCy NER).
    - **Cloud LLM**: GPT-4.1 driven structuring with automatic reasoning and problem definition.
- **🔄 Persistent Session Memory**: High-context interactions that remember your last 3 turns even across server restarts.
- **⚡ Keyword Macros**: Define custom keywords that automatically trigger and append long context-setting commands.
- **📊 3-Panel Side-by-Side Comparison**:
    1.  **Transcript**: View and edit the augmented voice results.
    2.  **Local NLP**: Real-time confidence bars and technical entity detection (`LANG`, `FRAMEWORK`, `ERROR`).
    3.  **GPT Structure**: Final structured JSON with clarity gaps and refined prompts.
- **🎛 Intent Modes**: Select from 7 specialized intents (Debug, Generate, Refactor, Explain, Scaffold, Test, Document) or use **Auto**.
- **🎨 Glassmorphic UI**: Modern tabbed interface with parallax motion design and real-time waveforms.

---

## 🏗 Tech Stack

- **Backend**: FastAPI, OpenAI (Whisper + GPT-4.1)
- **Local NLP**: Scikit-Learn (LinearSVC), SpaCy, Joblib, Numpy
- **Frontend**: Tailwind CSS, Vanilla JavaScript
- **Persistence**: File-based JSON storage for local-first reliability.

---

## ⚙️ Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SampurnGupta/DevFlow.git
   cd DevFlow
   ```

2. **Setup Environment**:
   Create a `.env` file and add your OpenAI key:
   ```text
   OPENAI_API_KEY=your_key_here
   ```

3. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   python -m uvicorn main:app --reload
   ```

---

## 📂 Project Structure

- `main.py`: Core logic, NLP pipeline, and persistence handlers.
- `templates/index.html`: The interactive Glassmorphic dashboard.
- `devflow_artefacts/`: Pre-trained local models for intent and entity recognition.
- `data/`: Persistent storage for your macros and session history.

---

## 🚀 Development Mode

- The app uses **StatReload**; any change to `main.py` or the JS will trigger a live refresh.
- Check the **Analyze** tab for the main voice-to-structure flow.
- Use the **Macros** tab to define your library of technical shortcuts.
