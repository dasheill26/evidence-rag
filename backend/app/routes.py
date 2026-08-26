"""
routes.py

API surface. Input validation is defined locally in this file - a real
bug found on an earlier project in this portfolio (Market Signal Lab)
came from calling a validation helper that only existed in a different
project's codebase, copy-pasted without checking it was actually
defined here. Every helper this file uses is defined in this file.
"""

import os
import uuid

from flask import Blueprint, jsonify, request, render_template
from pypdf import PdfReader

from app.engine.rag_pipeline import ingest_document, answer_question
from app.engine.evaluation import lexical_faithfulness, llm_judge_faithfulness
from app.engine import vectorstore

bp = Blueprint("main", __name__)


def _get_json_body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": "Request body must be a JSON object."}), 400)
    return data, None


def _extract_pdf_text(file_bytes: bytes) -> str:
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


@bp.route("/")
def index():
    return render_template("index.html", api_key_configured=bool(os.environ.get("ANTHROPIC_API_KEY")))


@bp.route("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


@bp.route("/api/config")
def get_config():
    return jsonify({"api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY"))})


@bp.route("/api/documents", methods=["GET"])
def list_documents():
    return jsonify({"documents": vectorstore.list_documents()})


@bp.route("/api/documents", methods=["POST"])
def upload_document():
    if "file" not in request.files and not (request.get_json(silent=True) or {}).get("text"):
        return jsonify({"error": "Provide either a 'file' upload or a JSON 'text' field."}), 400

    doc_id = uuid.uuid4().hex[:16]  # genuinely unique per upload, regardless of filename reuse

    if "file" in request.files:
        f = request.files["file"]
        filename = f.filename or "uploaded"
        raw = f.read()
        if filename.lower().endswith(".pdf"):
            try:
                text = _extract_pdf_text(raw)
            except Exception as e:
                return jsonify({"error": f"Could not read PDF: {e}"}), 422
        else:
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception as e:
                return jsonify({"error": f"Could not decode file as text: {e}"}), 422
        source = filename
    else:
        data = request.get_json()
        text = data.get("text", "")
        source = data.get("source", "pasted-text")

    if not text.strip():
        return jsonify({"error": "Document appears to be empty after extraction."}), 422

    n_chunks = ingest_document(text, doc_id=doc_id, source=source)
    if n_chunks == 0:
        return jsonify({"error": "No content could be chunked from this document."}), 422

    return jsonify({"doc_id": doc_id, "source": source, "chunks_created": n_chunks})


@bp.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    vectorstore.delete_document(doc_id)
    return jsonify({"deleted": doc_id})


@bp.route("/api/ask", methods=["POST"])
def ask():
    data, err = _get_json_body()
    if err:
        return err

    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "'question' is required and cannot be empty."}), 400

    doc_id = data.get("doc_id")  # optional - None searches across all ingested documents

    try:
        result = answer_question(question, doc_id=doc_id)
    except Exception as e:
        return jsonify({"error": f"Could not answer question: {e}"}), 500

    context_chunks = [{"text": c.text, "source": c.source} for c in result.retrieved_chunks]
    lexical = lexical_faithfulness(result.answer, context_chunks) if context_chunks else None
    llm_judge = llm_judge_faithfulness(question, result.answer, context_chunks) if context_chunks else None

    return jsonify({
        "question": result.question,
        "answer": result.answer,
        "retrieved_chunks": [
            {"text": c.text, "source": c.source, "chunk_index": c.chunk_index, "distance": c.distance}
            for c in result.retrieved_chunks
        ],
        "model": result.model,
        "latency_seconds": result.latency_seconds,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_cost_gbp": result.estimated_cost_gbp,
        "evaluation": {
            "lexical_faithfulness": lexical.__dict__ if lexical else None,
            "llm_judge_faithfulness": llm_judge.__dict__ if llm_judge else None,
        },
    })
