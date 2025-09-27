import os
import redis
# Set up Redis client (default: localhost, port 6379)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
import time
import json
import asyncio
import certifi
import pandas as pd
import docx
import pytesseract
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
from fastapi import (
    FastAPI, Request, Form, UploadFile, File, HTTPException,
    WebSocket, WebSocketDisconnect
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path

# LangChain / OpenAI (chat + embeddings)
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

import aiohttp
from patched_edge_tts import edge_tts  # Use patched version to disable SSL verify



## ---- Globals & Utilities (files / OCR / split) ----

# FastAPI app setup
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.urandom(24))
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



UPLOAD_FOLDER = "uploads"
DIAGRAM_FOLDER = "static/diagrams"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DIAGRAM_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'txt', 'jpg', 'jpeg', 'png', 'tif'}

global_embeddings = OpenAIEmbeddings()
global_vector_store = None

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path: str, ext: str) -> str:
    try:
        if ext == "pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            if not text.strip():
                # Image PDF fallback → OCR
                images = convert_from_path(file_path)
                text = "\n".join(pytesseract.image_to_string(img) for img in images)
        elif ext == "docx":
            d = docx.Document(file_path)
            text = "\n".join([p.text for p in d.paragraphs])
        elif ext == "txt":
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
        elif ext in {"jpg", "jpeg", "png", "tif"}:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        elif ext == "xlsx":
            sheets = pd.read_excel(file_path, sheet_name=None)
            text = "\n\n".join(f"{name}:\n{df.to_string(index=False)}" for name, df in sheets.items())
        else:
            return "Unsupported file"
        return text
    except Exception as e:
        return f"Error: {e}"

def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    return splitter.split_text(text)

def extract_images_from_pdf(pdf_path: str):
    pages = convert_from_path(pdf_path)
    paths = []
    for i, page in enumerate(pages):
        out = os.path.join(DIAGRAM_FOLDER, f"page_{i+1}.png")
        page.save(out, "PNG")
        paths.append(out)
    return paths

def reload_documents():
    global global_vector_store
    if not os.path.exists(UPLOAD_FOLDER):
        return
    files = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
    all_chunks = []
    for filename in files:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        ext = filename.rsplit('.', 1)[-1].lower()
        text = extract_text(file_path, ext)
        if text and not text.startswith("Error"):
            chunks = split_text(text)
            all_chunks.extend(chunks)
    if all_chunks:
        global_vector_store = FAISS.from_texts(all_chunks, global_embeddings)

reload_documents()
# -----------------------------
# ElevenLabs TTS
# -----------------------------
async def elevenlabs_tts_to_file(
    text: str,
    voice_id: str = "9BWtsMINqrJLrRacOk9x",
    output_path: str = "static/output.mp3",
    stability: float = 0.4,
    similarity_boost: float = 0.75,
):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": stability, "similarity_boost": similarity_boost}
    }
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        os.remove(output_path)
    except FileNotFoundError:
        pass

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                err = await resp.text()
                raise RuntimeError(f"ElevenLabs TTS failed: HTTP {resp.status} - {err[:200]}")
            with open(output_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(4096):
                    if chunk:
                        f.write(chunk)
    return output_path

# -----------------------------
# Edge TTS
# -----------------------------
async def edge_tts_to_file(
    text: str,
    voice_id: str = "en-US-AriaNeural",
    output_path: str = "static/output.mp3",
    rate: str = "+0%"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        os.remove(output_path)
    except FileNotFoundError:
        pass

    communicate = edge_tts.Communicate(text, voice=voice_id, rate=rate)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            with open(output_path, "ab") as f:
                f.write(chunk["data"])
    return output_path

# -----------------------------
# Conversation session
# -----------------------------
class ConversationSession:

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.voice_settings = {"voice_id": "9BWtsMINqrJLrRacOk9x", "speed": 1.0}
        self.last = time.time()
        self._load_messages()

    def _load_messages(self):
        # Load messages from Redis
        data = redis_client.get(f"session:{self.session_id}:messages")
        if data:
            self.messages = json.loads(data)
        else:
            self.messages = []

    def _save_messages(self):
        # Save messages to Redis (keep only last 10)
        self.messages = self.messages[-10:]
        redis_client.set(f"session:{self.session_id}:messages", json.dumps(self.messages), ex=60*60*24)  # 1 day expiry

    def add(self, role: str, content: str):
        self._load_messages()
        self.messages.append({"role": role, "content": content, "t": time.time()})
        self._save_messages()
        self.last = time.time()

    def recent_context(self, n: int = 5) -> str:
        self._load_messages()
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.messages[-n:])


async def _ws_keepalive(ws: WebSocket, interval: int = 10):
    try:
        while True:
            await asyncio.sleep(interval)
            await ws.send_json({"type": "ping", "t": int(time.time() * 1000)})
    except Exception:
        return

# -----------------------------
# Pages & endpoints
# -----------------------------
@app.get("/voices")
async def get_voices():
    return {
        "voices": [
            {"id": "edge:en-US-AriaNeural", "name": "Edge - Aria (Female)", "description": "US English, Female"},
            {"id": "edge:en-US-GuyNeural", "name": "Edge - Guy (Male)", "description": "US English, Male"},
            {"id": "edge:en-GB-LibbyNeural", "name": "Edge - Libby (Female)", "description": "UK English, Female"},
            {"id": "edge:en-GB-RyanNeural", "name": "Edge - Ryan (Male)", "description": "UK English, Male"},
            {"id": "edge:en-IN-NeerjaNeural", "name": "Edge - Neerja (Female)", "description": "Indian English, Female"},
            {"id": "edge:en-IN-PrabhatNeural", "name": "Edge - Prabhat (Male)", "description": "Indian English, Male"},
            {"id": "edge:en-AU-NatashaNeural", "name": "Edge - Natasha (Female)", "description": "Australian English, Female"},
            {"id": "edge:en-AU-WilliamNeural", "name": "Edge - William (Male)", "description": "Australian English, Male"},
            {"id": "edge:it-IT-ElsaNeural", "name": "Edge - Elsa (Female)", "description": "Italian, Female"},
            {"id": "edge:it-IT-DiegoNeural", "name": "Edge - Diego (Male)", "description": "Italian, Male"},
        ]
    }



# Default route serves speaking (conversation) mode
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# /chat serves interactive chat UI
@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("index_interactive.html", {"request": request})

# Serve the non-conversation chat UI
@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Serve the interactive conversation UI
@app.get("/conversation", response_class=HTMLResponse)
async def conversation(request: Request):
    return templates.TemplateResponse("index_interactive.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        print(f"Upload rejected: unsupported file type: {file.filename}")
        raise HTTPException(400, detail="Unsupported file")

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        with open(file_path, "wb") as f:
            file_bytes = await file.read()
            f.write(file_bytes)
        print(f"File uploaded: {file_path} ({len(file_bytes)} bytes)")
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")
        raise HTTPException(500, detail=f"Failed to save file: {e}")

    ext = filename.rsplit(".", 1)[1].lower()
    text = extract_text(file_path, ext)
    if text.startswith("Error") or not text.strip():
        print(f"Error extracting text from {file_path}: {text}")
        raise HTTPException(400, detail=text)

    chunks = split_text(text)
    global global_vector_store
    if global_vector_store is None:
        global_vector_store = FAISS.from_texts(chunks, global_embeddings)
    else:
        global_vector_store.add_texts(chunks)

    resp = {"message": f"{filename} processed"}
    if ext == "pdf":
        try:
            images = extract_images_from_pdf(file_path)
            resp["diagram_urls"] = [f"/static/diagrams/{os.path.basename(p)}" for p in images]
        except Exception as e:
            print(f"Error extracting images from PDF {file_path}: {e}")
            pass
    print(f"Upload response: {resp}")
    return JSONResponse(content=resp)

# RAG endpoint
@app.post("/ask")
async def ask_question(request: Request, question: str = Form(...)):
    import logging
    start_time = time.time()
    if global_vector_store is None:
        return JSONResponse(content={"answer": "Upload documents first"}, status_code=400)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer based on the context below.
Answer concisely in one sentence without introductory phrases

Context:
{context}

Question: {input}
Answer:
""")

    retriever = global_vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    doc_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    try:
        t0 = time.time()
        result = await asyncio.to_thread(lambda: retrieval_chain.invoke({"input": question}))
        t1 = time.time()
        answer = result.get("answer", "No answer found.")
        logging.info(f"RAG response time: {t1-t0:.2f}s")
    except Exception as e:
        print("❌ Error:", e)
        answer = "An error occurred."

    audio_url = None
    try:
        t2 = time.time()
        await edge_tts_to_file(answer, voice_id="en-US-AriaNeural")
        t3 = time.time()
        audio_url = f"/static/output.mp3?nocache={int(time.time())}"
        logging.info(f"Edge TTS time: {t3-t2:.2f}s")
    except Exception as tts_e:
        print("Edge TTS error:", tts_e)
        try:
            t4 = time.time()
            await elevenlabs_tts_to_file(answer, voice_id="9BWtsMINqrJLrRacOk9x")
            t5 = time.time()
            audio_url = f"/static/output.mp3?nocache={int(time.time())}"
            logging.info(f"ElevenLabs TTS time: {t5-t4:.2f}s")
        except Exception as tts2_e:
            print("ElevenLabs TTS error:", tts2_e)

    end_time = time.time()
    logging.info(f"Total /ask endpoint time: {end_time-start_time:.2f}s")
    return JSONResponse(content={"answer": answer, "audio_url": audio_url})

# -----------------------------
# WebSocket
# -----------------------------
@app.websocket("/ws")
async def ws_conversation(websocket: WebSocket):
    await websocket.accept()
    ka_task = asyncio.create_task(_ws_keepalive(websocket))
    session_id = f"conv_{int(time.time())}_{id(websocket)}"
    session = ConversationSession(session_id)

    await websocket.send_json({
        "type": "session_start",
        "session_id": session_id,
        "message": "🎤 Conversation mode activated!"
    })

    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except Exception:
                # Connection closed or error receiving
                break
            try:
                msg = json.loads(raw)
            except Exception:
                try:
                    await websocket.send_json({"type": "error", "content": "Invalid JSON"})
                except Exception:
                    # Don't write after close
                    break
                continue

            mtype = msg.get("type")
            if mtype == "voice_input":
                user_message = msg.get("content", "").strip()
                if not user_message:
                    try:
                        await websocket.send_json({"type": "error", "content": "Empty message"})
                    except Exception:
                        break
                    continue

                session.voice_settings.update(msg.get("voice_settings", {}))
                session.add("user", user_message)

                try:
                    await websocket.send_json({"type": "processing", "message": "🤔 Thinking..."})
                except Exception:
                    break

                context_prompt = (
                    "You are an intelligent, friendly assistant. "
                    "Answer directly, clearly, and helpfully.\n\n"
                    f"Previous conversation:\n{session.recent_context(5)}\n\n"
                    f"Question: {user_message}\n"
                )

                try:
                    # Use RAG retrieval chain if vector store is available
                    if global_vector_store is not None:
                        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer based on the context below.
Answer concisely in one sentence without introductory phrases

Context:
{context}

Question: {input}
Answer:
""")
                        retriever = global_vector_store.as_retriever()
                        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
                        doc_chain = create_stuff_documents_chain(llm, prompt)
                        retrieval_chain = create_retrieval_chain(retriever, doc_chain)
                        result = await asyncio.to_thread(lambda: retrieval_chain.invoke({"input": user_message}))
                        ai_response = result.get("answer", "No answer found.")
                    else:
                        ai_response = await asyncio.to_thread(
                            lambda: ChatOpenAI(model_name=msg.get("ai_model", "gpt-3.5-turbo"), temperature=0.7).invoke(context_prompt).content
                        )
                except Exception as e:
                    await websocket.send_json({"type": "error", "content": f"LLM error: {e}"})
                    continue

                session.add("assistant", ai_response)
                audio_url = None
                voice_id = session.voice_settings.get("voice_id", "9BWtsMINqrJLrRacOk9x")
                speed = session.voice_settings.get("speed", 1.0)
                # Convert speed to Edge TTS rate string
                try:
                    speed_float = float(speed)
                    if speed_float == 1.0:
                        rate_str = "+0%"
                    else:
                        rate_percent = int((speed_float - 1.0) * 100)
                        rate_str = f"{rate_percent:+d}%"
                except Exception:
                    rate_str = "+0%"

                if voice_id.startswith("edge:"):
                    try:
                        edge_voice = voice_id.split(":", 1)[1]
                        await edge_tts_to_file(ai_response, voice_id=edge_voice, rate=rate_str)
                        audio_url = f"/static/output.mp3?nocache={int(time.time())}"
                    except Exception as tts_e:
                        print("Edge TTS error:", tts_e)
                elif voice_id.startswith("elevenlabs:") or voice_id == "9BWtsMINqrJLrRacOk9x":
                    try:
                        await elevenlabs_tts_to_file(ai_response, voice_id=voice_id)
                        audio_url = f"/static/output.mp3?nocache={int(time.time())}"
                    except Exception as tts_e:
                        print("ElevenLabs TTS error:", tts_e)

                await websocket.send_json({
                    "type": "ai_response",
                    "content": ai_response,
                    "audio_url": audio_url,
                    "voice_used": voice_id,
                    "session_id": session_id
                })

            elif mtype == "voice_settings":
                settings = msg.get("settings", {})
                session.voice_settings.update(settings)
                await websocket.send_json({"type": "settings_updated", "voice_settings": session.voice_settings})

            elif mtype == "end_conversation":
                await websocket.send_json({"type": "conversation_ended", "message": "👋 Conversation ended. Thanks!"})
                break

            else:
                await websocket.send_json({"type": "error", "content": f"Unknown message type: {mtype}"})

    except WebSocketDisconnect:
        pass
    finally:
        try:
            ka_task.cancel()
        except Exception:
            pass
    # No in-memory session cleanup needed with Redis

# -----------------------------
# Transcription
# -----------------------------
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    from openai import OpenAI
    ext = file.filename.split('.')[-1].lower()
    if ext not in {'mp3', 'wav', 'm4a', 'webm', 'ogg'}:
        raise HTTPException(400, detail="Audio format not supported")

    tmp = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    with open(tmp, "wb") as f:
        f.write(await file.read())

    try:
        client = OpenAI()
        with open(tmp, "rb") as af:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=af)
        text = transcript.text if hasattr(transcript, "text") else ""
        return JSONResponse(content={"transcript": text})
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
