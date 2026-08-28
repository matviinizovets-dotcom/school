"""
School Test Archiving Platform (Google Gemini backend)
--------------------------------------------------------
Flask backend that:
  1. Accepts a photo of a school test as a base64-encoded JSON payload
     (the client resizes/compresses the image via canvas before sending).
  2. Sends the image bytes to the Gemini API (vision) as inline data for
     extraction — no Pillow / PIL involved, since the client already did
     any necessary resizing.
  3. Parses the returned JSON and stores/serves it so students can
     take the quiz interactively.

This version switches from multipart file upload to a JSON body containing
a base64 string, to rule out multipart/Pillow file-handling as a source of
server-side memory pressure. templates/index.html has been updated to match
this new request shape.

Environment variables required (see .env.example):
  GEMINI_API_KEY   - your Google AI Studio / Gemini API key
  SECRET_KEY       - Flask session secret (any random string)
  GEMINI_MODEL     - optional, defaults to gemini-3.6-flash
  MAX_CONTENT_MB   - optional, defaults to 8 (overall request body cap)
  MAX_IMAGE_MB     - optional, defaults to 2 (decoded image byte cap)
"""

import os
import re
import json
import uuid
import base64
import logging
from datetime import datetime

from flask import (
    Flask, render_template, request, jsonify, abort
)
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# NOTE: We fetch GEMINI_API_KEY here but do NOT validate/raise on it at
# import time. Render (and some other hosts) import the app to discover it
# before all environment variables are guaranteed to be injected, and a
# top-level `raise` here would crash the boot process. Locally, load_dotenv()
# above pulls it from a .env file if present; on Render it comes from the
# dashboard's environment settings. Validation happens lazily in
# get_model(), the first time the key is actually needed.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# TEMPORARY DEBUG LINE — remove once you've confirmed Render is passing the
# env var correctly. This only prints True/False, never the key itself, so
# it won't leak the secret into your logs.
print("DEBUG: API Key exists?", bool(GEMINI_API_KEY))

# Current recommended vision-capable model (Aug 2026).
# Check https://ai.google.dev/gemini-api/docs/models for the latest list
# before deploying — model names occasionally change.
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
MAX_CONTENT_MB = int(os.environ.get("MAX_CONTENT_MB", "8"))

# Extra safety net specifically for the decoded image bytes, independent of
# MAX_CONTENT_LENGTH above (which caps the raw JSON body, base64 and all).
# Client-side resizing should already keep uploads small, but this guards
# against clients that skip/fail that step (old cached page, JS disabled,
# direct API calls, etc.).
MAX_IMAGE_MB = float(os.environ.get("MAX_IMAGE_MB", "2"))
MAX_IMAGE_BYTES = int(MAX_IMAGE_MB * 1024 * 1024)

# In-memory store for demo purposes. Swap for a real database
# (Postgres/SQLite) before going to production with real users.
TESTS_DB = {}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model = None  # lazily created on first use, see get_model()


def get_model():
    """
    Lazily validates GEMINI_API_KEY (fetched once at import time into the
    module-level GEMINI_API_KEY variable, but not validated until now) and
    builds the GenerativeModel on first use. Cached in _model after that.

    Raises RuntimeError if GEMINI_API_KEY is missing — callers should catch
    this and turn it into a proper JSON error response, not let it crash
    the process.
    """
    global _model
    if _model is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your environment or .env file."
            )
        genai.configure(api_key=GEMINI_API_KEY)
        _model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    return _model

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are an API backend processor for a school test archiving platform.
Analyze the provided image of a school test.

Tasks:
1. Extract metadata: Class/Grade, Subject, Date, and Topic.
2. Transcribe all questions, text, and handwritten answers accurately.
   If any part is illegible, mark it clearly as "[illegible]" rather than guessing.
3. Check for any visible personal student names. If a name appears, replace it
   in the transcription with "[name redacted]" and set privacy_flag to true.

Output ONLY valid JSON, with no extra commentary and no markdown code fences,
in exactly this format:
{
  "metadata": {
    "class": "string",
    "subject": "string",
    "date": "string",
    "topic": "string"
  },
  "transcription": "string",
  "privacy_flag": true/false
}
"""

QUIZ_PROMPT = """You are an educational platform assistant. Based on the test transcription
provided below, build an interactive quiz.

Rules:
1. Extract every distinct question individually.
2. Order them sequentially.
3. Provide the correct answer/solution ONLY when it can be determined with
   confidence from the transcription. If a value is illegible or ambiguous,
   set "correct_answer" to "unknown - manual review required" rather than
   inventing a number, to avoid mis-grading students.

Output ONLY valid JSON, with no extra commentary and no markdown code fences,
in exactly this format:
{
  "quiz_questions": [
    {
      "question_number": 1,
      "text": "Full text of the question...",
      "correct_answer": "The correct answer or 'unknown - manual review required'"
    }
  ]
}

Test transcription:
---
{transcription}
---
"""

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<data>.*)$", re.DOTALL)


def decode_base64_image(image_field: str) -> tuple[bytes, str]:
    """
    Accepts either a full data URL (e.g. "data:image/jpeg;base64,/9j/4AAQ...")
    or a bare base64 string, and returns (raw_bytes, mime_type).

    Raises ValueError on malformed input — callers should catch this and
    return a 400, not let it propagate as a 500.
    """
    if not image_field or not isinstance(image_field, str):
        raise ValueError("image_base64 must be a non-empty string.")

    match = DATA_URL_RE.match(image_field.strip())
    if match:
        mime_type = match.group("mime").lower()
        b64_data = match.group("data")
    else:
        # Bare base64 with no data-URL header; assume JPEG since that's
        # what our client-side canvas resize always outputs.
        mime_type = "image/jpeg"
        b64_data = image_field.strip()

    try:
        raw_bytes = base64.b64decode(b64_data, validate=True)
    except Exception as exc:
        raise ValueError(f"Could not decode base64 image data: {exc}") from exc

    if not raw_bytes:
        raise ValueError("Decoded image data is empty.")

    return raw_bytes, mime_type


def extract_json_block(text: str) -> dict:
    """
    Gemini is instructed to return only JSON, but this defensively strips
    markdown code fences if the model adds them anyway.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


def call_gemini_vision(prompt: str, image_bytes: bytes, mime_type: str) -> str:
    """
    Sends a single image (as inline data) + text prompt to Gemini and
    returns the raw text reply. Passing raw bytes directly avoids an extra
    base64 encode/decode round trip inside the SDK.
    """
    image_part = {"mime_type": mime_type, "data": image_bytes}
    response = get_model().generate_content(
        [prompt, image_part],
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=2000,
        ),
    )
    return response.text


def call_gemini_text(prompt: str) -> str:
    """Sends a text-only prompt to Gemini and returns the raw text reply."""
    response = get_model().generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=2000,
        ),
    )
    return response.text


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_test():
    """
    Accepts a JSON body: { "image_base64": "data:image/jpeg;base64,...",
    "filename": "optional-name.jpg" }.

    Decodes the image in-memory, extracts metadata + transcription via
    Gemini, generates a quiz from the transcription, stores the result,
    and returns it as JSON along with a test_id.
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Expected a JSON body with an 'image_base64' field."}), 400

    image_field = payload.get("image_base64")
    if not image_field:
        return jsonify({"error": "Missing 'image_base64' in request body."}), 400

    raw_filename = payload.get("filename") or "test-photo.jpg"
    filename = secure_filename(raw_filename) or "test-photo.jpg"

    # --- Decode + validate in-memory, before ever touching Gemini ---
    try:
        image_bytes, mime_type = decode_base64_image(image_field)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if mime_type not in ALLOWED_MIME_TYPES:
        return jsonify({
            "error": f"Unsupported image type '{mime_type}'. Allowed: {sorted(ALLOWED_MIME_TYPES)}"
        }), 400

    if len(image_bytes) > MAX_IMAGE_BYTES:
        logger.warning(
            "Rejected upload: %.2f MB exceeds the %.2f MB per-image limit",
            len(image_bytes) / (1024 * 1024), MAX_IMAGE_MB,
        )
        return jsonify({
            "error": (
                f"Image is too large ({len(image_bytes) / (1024 * 1024):.1f} MB). "
                f"Please upload a photo under {MAX_IMAGE_MB:.0f} MB."
            )
        }), 413

    # --- Step 1: extract metadata + transcription ---
    try:
        raw_extraction = call_gemini_vision(EXTRACTION_PROMPT, image_bytes, mime_type)
        extraction = extract_json_block(raw_extraction)
    except RuntimeError as exc:
        # Missing GEMINI_API_KEY, surfaced lazily from get_model().
        logger.error("Gemini not configured: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except json.JSONDecodeError:
        logger.error("Gemini did not return valid JSON for extraction: %s", raw_extraction)
        return jsonify({"error": "The AI response could not be parsed as JSON. Please try again."}), 502
    except Exception as exc:
        logger.exception("Gemini API error during extraction")
        return jsonify({"error": f"Gemini API error: {exc}"}), 502

    # image_bytes is no longer needed past this point — drop the reference
    # explicitly so it can be garbage-collected before the quiz-generation
    # call, rather than lingering for the rest of the request.
    del image_bytes

    # --- Step 2: build the interactive quiz from the transcription ---
    quiz = {"quiz_questions": []}
    transcription = extraction.get("transcription", "")
    if transcription:
        try:
            quiz_prompt_filled = QUIZ_PROMPT.replace("{transcription}", transcription)
            raw_quiz = call_gemini_text(quiz_prompt_filled)
            quiz = extract_json_block(raw_quiz)
        except Exception:
            logger.exception("Quiz generation failed; continuing with empty quiz.")

    test_id = str(uuid.uuid4())
    record = {
        "id": test_id,
        "filename": filename,
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "metadata": extraction.get("metadata", {}),
        "transcription": transcription,
        "privacy_flag": extraction.get("privacy_flag", False),
        "quiz_questions": quiz.get("quiz_questions", []),
    }
    TESTS_DB[test_id] = record

    return jsonify(record), 200


@app.route("/api/tests", methods=["GET"])
def list_tests():
    """Returns a lightweight list of all archived tests (no full transcription)."""
    summary = [
        {
            "id": t["id"],
            "metadata": t["metadata"],
            "uploaded_at": t["uploaded_at"],
            "privacy_flag": t["privacy_flag"],
            "question_count": len(t["quiz_questions"]),
        }
        for t in TESTS_DB.values()
    ]
    return jsonify(summary), 200


@app.route("/api/tests/<test_id>", methods=["GET"])
def get_test(test_id):
    record = TESTS_DB.get(test_id)
    if not record:
        abort(404)
    return jsonify(record), 200


@app.route("/api/tests/<test_id>/check", methods=["POST"])
def check_answers(test_id):
    """
    Accepts {"answers": {"1": "student answer", "2": "..."}} and returns
    a simple correctness report. This is intentionally a plain string
    comparison — swap in fuzzy/semantic matching for production use.
    """
    record = TESTS_DB.get(test_id)
    if not record:
        abort(404)

    submitted = request.get_json(silent=True) or {}
    answers = submitted.get("answers", {})

    results = []
    for q in record["quiz_questions"]:
        qnum = str(q["question_number"])
        student_answer = (answers.get(qnum) or "").strip().lower()
        correct_answer = (q.get("correct_answer") or "").strip().lower()
        is_gradable = "unknown" not in correct_answer
        results.append({
            "question_number": q["question_number"],
            "student_answer": answers.get(qnum, ""),
            "correct_answer": q.get("correct_answer", ""),
            "is_correct": (student_answer == correct_answer) if is_gradable else None,
            "gradable": is_gradable,
        })

    return jsonify({"test_id": test_id, "results": results}), 200


@app.errorhandler(413)
def file_too_large(_e):
    return jsonify({"error": f"Request too large. Max size is {MAX_CONTENT_MB} MB."}), 413


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found."}), 404


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # For local dev only. On Render, gunicorn runs this via the Procfile:
    #   web: gunicorn app:app --workers 1 --threads 2
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")