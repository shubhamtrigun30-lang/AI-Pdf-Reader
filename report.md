# AI PDF Question Answering Chatbot — Project Report

## Abstract

This project is a Streamlit application for asking questions about an uploaded PDF. It extracts selectable text page by page with PyMuPDF, divides text into overlapping chunks, creates local sentence embeddings, and stores them in a FAISS index. For a question, it embeds the question and retrieves up to four relevant chunks. The implemented Gemini client sends those excerpts and the question to the Google Gen AI API and asks the model to answer from the document, with source page references. PDF extraction, embeddings, and retrieval were tested in the current environment. A live Gemini answer was not tested because no API key is configured.

## Introduction

Long PDF documents can be difficult to search manually. This project demonstrates a small Retrieval-Augmented Generation (RAG) workflow: search a document for passages related to a question, then use a language model to form an answer from those passages. The application is intended for learning and college demonstration, not as a replacement for checking the cited source pages.

## Problem statement

Users need a simple way to locate information in a PDF and ask natural-language questions about it. Searching only by exact words may miss relevant passages, while sending an entire document to a language model is inefficient. The project uses semantic retrieval over text chunks to provide a small set of relevant excerpts for question answering.

## Objectives

- Provide a browser interface for PDF upload and chat.
- Extract and preview text page by page.
- Split page text into chunks of about 800 characters with 150-character overlap.
- Keep page and source metadata with each chunk.
- Generate embeddings locally with Sentence Transformers.
- Store and search vectors using FAISS.
- Use Gemini with retrieved excerpts to generate document-grounded answers and page references.
- Handle invalid, empty, password-protected, and textless PDFs with readable messages.
- Read API credentials from a local `.env` file rather than hardcoding them.

## System architecture

```text
PDF upload in Streamlit
          |
          v
PyMuPDF extracts text by page
          |
          v
Page-aware chunks (800 characters, 150 overlap)
          |
          v
Sentence Transformers creates local embeddings
          |
          v
FAISS index stores vectors <--- question embedding
          |                              |
          +---- top 4 relevant chunks ---+
                         |
                         v
          Gemini API receives question and excerpts
                         |
                         v
          Answer and retrieved source page labels
```

The Streamlit session keeps the uploaded PDF's extracted text, chunks, FAISS index, and chat messages in memory. The embedding model runs locally. For answer generation, `gemini_client.py` sends the question, recent conversation turns, and retrieved excerpts to Gemini; it does not send the full PDF file.

## Methodology

1. The user uploads a PDF through Streamlit's PDF file uploader.
2. PyMuPDF opens the uploaded bytes and extracts selectable text from each page.
3. The app displays page count and text previews. A document with no selectable text cannot proceed to chunking; OCR is not included.
4. `chunk_pdf_pages` splits each page's text into chunks of up to 800 characters, stepping by 650 characters so adjacent chunks overlap by 150 characters. Every chunk keeps the source filename and one-based page number.
5. Sentence Transformers encodes each chunk with `sentence-transformers/all-MiniLM-L6-v2`. The vectors are normalized and added to a FAISS inner-product index.
6. For a non-empty question, the same local model creates a question embedding. FAISS returns up to four nearest chunks with their page metadata.
7. If `GEMINI_API_KEY` is available, the Google Gen AI SDK sends the excerpts and question to the configured Gemini model. The system instruction tells Gemini to use only the excerpts, abstain when the answer is not present, and cite page numbers.
8. The chat interface displays the response, retrieved sources, and session conversation history.

## Tools and technologies

- **Python 3.14.3** — interpreter in the environment checked for this report.
- **Streamlit** — web interface and session state.
- **PyMuPDF** — PDF loading and text extraction.
- **Sentence Transformers** — local text and question embeddings.
- **FAISS CPU** — vector similarity index and retrieval.
- **Google Gen AI SDK (`google-genai`)** — Gemini answer-generation client.
- **python-dotenv** — loads local Gemini configuration from `.env`.
- **NumPy** — converts embedding vectors to the format used by FAISS.

## Implementation

The main application is `app.py`. It handles upload, extraction, UI messages, loading the embedding model, and the chat flow. `vector_store.py` contains chunking, vector index creation, and retrieval functions. `gemini_client.py` loads the environment configuration, prepares the prompt from retrieved excerpts, makes the Gemini request, and maps API failures to readable messages. `requirements.txt` lists the dependencies. `.env.example` is a template; the actual `.env` file is ignored by `.gitignore` and must be created locally by the user.

The Gemini API key is not stored in the source code. The current workspace did not have a configured key, and no live API request was made during testing.

## Testing and actual results

Tests below were run in the project's existing `.venv` using Python 3.14.3. A two-page sample PDF was generated in memory for the test; page 1 said “The astronomy club meets in Pune every Friday,” and page 2 said “The physics laboratory opens at nine in the morning.”

| Check | Actual result |
|---|---|
| Python syntax compilation for `app.py`, `vector_store.py`, and `gemini_client.py` | Passed. |
| `pip check` | Passed; no broken requirements reported. |
| Streamlit startup | Passed; server started at `http://localhost:8502` during this run. |
| Local HTTP request to the Streamlit page | Passed; returned HTTP 200. |
| Upload and extraction of the two-page sample | Passed; both pages were recognized and page text was displayed by the test harness. |
| Chunking and embedding | Passed; 2 pages produced 2 chunks and 2 embeddings. |
| FAISS index creation | Passed; index reported 2 vectors. |
| Retrieval question: “Where does the astronomy club meet?” | Passed at retrieval level; the first result contained “Pune” and identified page 1. |
| Zero-byte, corrupted, image-only/scanned-like, and password-protected PDF handling | Passed; each displayed its expected error or limitation message. |
| Missing Gemini key handling | Passed; the app displayed setup guidance and the client refused to call the API without a key. |
| Live Gemini question answer and factual grounding | Not tested. No Gemini key is configured, so no model request or generated answer was observed. |

The retrieval result shows that the question matched a relevant page passage. It does not establish that the Gemini-generated response is accurate; live answer quality and abstention behavior still need a real-key test.

## Limitations

- Scanned and image-only pages do not provide selectable text; OCR is not implemented.
- Live question answering requires a valid Gemini API key and internet access. The key must be supplied by the project owner.
- The embedding model may need to download on first use, requiring internet access and local disk space.
- The uploaded PDF processing state, vector index, and chat history are kept in session memory and are not persisted when the session/app ends.
- Retrieval can return imperfect passages. Users should verify answers against the displayed source pages.
- The project was tested with a short synthetic PDF, not with every PDF layout or a large production document.

## Future scope

- Add OCR for scanned PDFs.
- Test answer accuracy and unsupported-question behavior using a configured Gemini key and a small evaluation set.
- Add optional persistence for indexes if a future use case requires it.
- Improve retrieval evaluation while keeping page citations visible.

## Conclusion

The current implementation demonstrates the main local RAG preparation pipeline: PDF text extraction, page-aware chunking, local embeddings, FAISS indexing, and semantic retrieval. The Streamlit server and PDF-processing workflow were executed successfully in the inspected Python environment. Gemini integration is present in code, but its live answer generation remains unverified until a valid API key is configured and a demonstration question is run.
