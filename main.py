"""
WhatsApp R&D assistant — webhook server.

Receives WhatsApp messages via Twilio, answers them with Claude, and keeps a
small per-user memory plus a searchable notebook of R&D notes.

Run:  uvicorn main:app --reload --port 8000
"""

import os
import re
import sqlite3
import datetime as dt
from xml.sax.saxutils import escape as xml_escape

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, Response
from anthropic import Anthropic

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "bot.db")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
MAX_HISTORY = 20            # how many past messages we replay to the model
WHATSAPP_LIMIT = 1500       # Twilio hard limit is 1600 chars per message
VALIDATE_SIGNATURE = os.getenv("VALIDATE_TWILIO_SIGNATURE", "false").lower() == "true"

# Comma-separated list of allowed numbers, e.g. "whatsapp:+60123456789"
ALLOWED = {n.strip() for n in os.getenv("ALLOWED_NUMBERS", "").split(",") if n.strip()}

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
app = FastAPI(title="WhatsApp R&D assistant")

SYSTEM_PROMPT = """You are an R&D support assistant reached over WhatsApp.

You help with daily research and development work: debugging experiments,
suggesting test plans, checking calculations, summarising findings, and
recalling past lab notes the user saved.

Rules:
- Answer in WhatsApp style: short, plain text, no markdown headers or tables.
- Keep answers under 120 words unless asked for more. Use short line breaks,
  not long paragraphs.
- If notes from the user's notebook are provided, use them and say which note
  you relied on (by its date).
- If a calculation or a number matters, show the working in one line.
- If you are not sure, say so and say what to check next. Never invent data,
  part numbers, or measurement results.
"""


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                wa_id   TEXT NOT NULL,
                role    TEXT NOT NULL,          -- 'user' or 'assistant'
                content TEXT NOT NULL,
                ts      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_wa ON messages(wa_id, id);

            CREATE TABLE IF NOT EXISTS notes (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                wa_id   TEXT NOT NULL,
                content TEXT NOT NULL,
                ts      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_notes_wa ON notes(wa_id, id);
            """
        )


def save_message(wa_id: str, role: str, content: str):
    with db() as conn:
        conn.execute(
            "INSERT INTO messages (wa_id, role, content, ts) VALUES (?,?,?,?)",
            (wa_id, role, content, dt.datetime.now().isoformat(timespec="seconds")),
        )


def get_history(wa_id: str, limit: int = MAX_HISTORY):
    with db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE wa_id=? ORDER BY id DESC LIMIT ?",
            (wa_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def save_note(wa_id: str, content: str) -> str:
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with db() as conn:
        conn.execute(
            "INSERT INTO notes (wa_id, content, ts) VALUES (?,?,?)",
            (wa_id, content, stamp),
        )
    return stamp


def all_notes(wa_id: str):
    with db() as conn:
        return conn.execute(
            "SELECT content, ts FROM notes WHERE wa_id=? ORDER BY id DESC LIMIT 500",
            (wa_id,),
        ).fetchall()


def notes_today(wa_id: str):
    today = dt.date.today().isoformat()
    with db() as conn:
        return conn.execute(
            "SELECT content, ts FROM notes WHERE wa_id=? AND ts LIKE ? ORDER BY id",
            (wa_id, f"{today}%"),
        ).fetchall()


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "for", "and",
    "or", "in", "on", "at", "it", "this", "that", "with", "what", "how",
    "why", "did", "do", "does", "my", "we", "i", "you", "about",
}


def search_notes(wa_id: str, query: str, top_k: int = 3):
    """Cheap keyword scoring. Good enough until you add real embeddings."""
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in STOPWORDS and len(w) > 2}
    if not words:
        return []
    scored = []
    for row in all_notes(wa_id):
        text = row["content"].lower()
        score = sum(1 for w in words if w in text)
        if score:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_k]]


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
HELP = (
    "R&D assistant commands:\n"
    "/log <text> — save a lab note\n"
    "/find <words> — search your notes\n"
    "/today — notes saved today\n"
    "/reset — forget this conversation\n"
    "/help — this message\n\n"
    "Anything else is a normal question."
)


def handle_command(wa_id: str, body: str):
    """Return a reply string if this was a command, else None."""
    low = body.strip().lower()

    if low in ("/help", "help", "hi", "hello"):
        return HELP

    if low.startswith("/log"):
        note = body.strip()[4:].strip()
        if not note:
            return "Nothing to save. Try: /log Run 14 gave 82% yield at 60C"
        stamp = save_note(wa_id, note)
        return f"Saved ({stamp[:16].replace('T', ' ')})."

    if low.startswith("/find"):
        q = body.strip()[5:].strip()
        hits = search_notes(wa_id, q, top_k=5)
        if not hits:
            return "No matching notes."
        return "\n\n".join(f"{h['ts'][:16].replace('T', ' ')}\n{h['content']}" for h in hits)

    if low == "/today":
        rows = notes_today(wa_id)
        if not rows:
            return "No notes logged today."
        return "Today's notes:\n\n" + "\n\n".join(
            f"{r['ts'][11:16]} — {r['content']}" for r in rows
        )

    if low == "/reset":
        with db() as conn:
            conn.execute("DELETE FROM messages WHERE wa_id=?", (wa_id,))
        return "Conversation cleared. Your saved notes are untouched."

    return None


# --------------------------------------------------------------------------
# Model call
# --------------------------------------------------------------------------
def ask_claude(wa_id: str, body: str) -> str:
    history = get_history(wa_id)

    hits = search_notes(wa_id, body)
    if hits:
        context = "\n".join(f"[{h['ts'][:16]}] {h['content']}" for h in hits)
        user_content = f"Relevant notes from my notebook:\n{context}\n\nMy question: {body}"
    else:
        user_content = body

    messages = history + [{"role": "user", "content": user_content}]

    resp = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()


# --------------------------------------------------------------------------
# Webhook
# --------------------------------------------------------------------------
def twiml(text: str) -> Response:
    if len(text) > WHATSAPP_LIMIT:
        text = text[:WHATSAPP_LIMIT] + "\n\n[...truncated]"
    xml = f"<?xml version='1.0' encoding='UTF-8'?><Response><Message>{xml_escape(text)}</Message></Response>"
    return Response(content=xml, media_type="application/xml")


async def signature_ok(request: Request, form: dict) -> bool:
    if not VALIDATE_SIGNATURE:
        return True
    from twilio.request_validator import RequestValidator

    validator = RequestValidator(os.environ["TWILIO_AUTH_TOKEN"])
    url = os.getenv("PUBLIC_URL", str(request.url))
    return validator.validate(url, form, request.headers.get("X-Twilio-Signature", ""))


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/whatsapp")
async def whatsapp(request: Request, From: str = Form(""), Body: str = Form("")):
    form = dict(await request.form())

    if not await signature_ok(request, form):
        return Response(status_code=403, content="bad signature")

    wa_id, body = From, (Body or "").strip()

    if ALLOWED and wa_id not in ALLOWED:
        return twiml("This assistant is private. Ask the owner to add your number.")

    if not body:
        return twiml("I can only read text messages for now. Try /help.")

    reply = handle_command(wa_id, body)
    if reply is None:
        save_message(wa_id, "user", body)
        try:
            reply = ask_claude(wa_id, body)
        except Exception as exc:  # keep the user informed instead of timing out
            print("model error:", exc)
            return twiml("Something went wrong reaching the model. Try again in a moment.")
        save_message(wa_id, "assistant", reply)

    return twiml(reply)


init_db()
