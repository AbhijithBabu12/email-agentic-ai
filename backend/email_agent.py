import os
import re
import json
import torch
import base64
import pickle
import uuid

DRAFT_STORE = {}
from threading import Thread
from typing import Generator, Optional, Dict

from transformers import (
    AutoTokenizer,  
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig
)

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


# =========================================================
# 🔥 MODEL LOADING (Qwen 7B 4bit)
# =========================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model_id = "Qwen/Qwen2.5-7B-Instruct"

print("Loading Qwen model...")

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
)

print("Model loaded ✅")


# =========================================================
# 🔥 STREAMING GENERATION CORE
# =========================================================

def stream_generate(system_prompt: str, user_prompt: str) -> Generator[str, None, None]:

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=600,
        temperature=0.6,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for token in streamer:
        yield token


# =========================================================
# 🔥 CONTACT VECTOR DATABASE (Qdrant)
# =========================================================

with open("contact.json", "r") as f:
    contacts = json.load(f)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = QdrantClient(path="qdrant_db")

collection_name = "contacts"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embedding_model
)

if client.count(collection_name).count == 0:
    from langchain_core.documents import Document
    docs = []
    for contact in contacts:
        text = f"""
        Name: {contact['name']}
        Email: {contact['email']}
        Phone: {contact['phone']}
        """
        docs.append(Document(page_content=text, metadata=contact))
    vectorstore.add_documents(docs)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# =========================================================
# 🔥 INTENT EXTRACTION
# =========================================================

def extract_intent(user_input: str) -> Dict:

    user_input_lower = user_input.lower()

    email_match = re.search(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        user_input
    )
    recipient_email = email_match.group(0) if email_match else None
    
    name_match = re.search(r"to\s+([a-zA-Z\s]+)", user_input_lower)
    recipient_name = name_match.group(1).strip() if name_match and not recipient_email else None

    tone = "professional"
    tone_map = {
        "friendly": ["friendly", "casual"],
        "technical": ["technical", "detailed"],
        "romantic": ["romantic", "love"],
        "apologetic": ["sorry", "apology"],
        "enthusiastic": ["excited", "thrilled"]
    }

    for t, keywords in tone_map.items():
        if any(word in user_input_lower for word in keywords):
            tone = t
            break

    desc_match = re.search(r"about\s+(.+)", user_input, re.IGNORECASE)
    message_description = desc_match.group(1) if desc_match else user_input

    # Generate subject (non-streaming small call)
    subject_prompt = f"Generate a short email subject:\n{message_description}"

    subject = ""
    for token in stream_generate(
        "You generate concise subject lines.",
        subject_prompt
    ):
        subject += token

    return {
        "recipient_name": recipient_name,
        "recipient_email": recipient_email,
        "subject": subject.strip(),
        "message_description": message_description,
        "tone": tone
    }


# =========================================================
# 🔥 CONTACT LOOKUP
# =========================================================

def lookup_contact(intent: Dict) -> Optional[Dict]:

    if intent.get("recipient_email"):
        return {"email": intent["recipient_email"], "name": "Recipient"}

    name = intent.get("recipient_name")
    if not name:
        return None

    results = retriever.invoke(name)
    if not results:
        return None

    return results[0].metadata


# =========================================================
# 🔥 EMAIL STREAM
# =========================================================

def stream_email(intent: Dict, contact: Dict) -> Generator[str, None, None]:

    prompt = f"""
You are Abhijith Babu.

Write an email to {contact.get('name')}.

Tone: {intent['tone']}
Subject: {intent['subject']}

Purpose:
{intent['message_description']}

Rules:
- Keep tone accurate
- No placeholders
- Sign exactly:

Abhijith Babu
"""

    yield from stream_generate(
        "You are a professional email assistant.",
        prompt
    )

def stream_edit_email(original_body: str, edit_instruction: str, tone: str):

    prompt = f"""
You are Abhijith Babu.

The original email is:

{original_body}

Modify it according to this instruction:
{edit_instruction}

Rules:
- Keep tone: {tone}
- Do not add placeholders
- Sign exactly:

Abhijith Babu
"""

    yield from stream_generate(
        "You are a professional email editor.",
        prompt
    )
# =========================================================
# 🔥 CHAT STREAM
# =========================================================

def stream_chat(user_input: str) -> Generator[str, None, None]:

    yield from stream_generate(
        "You are a helpful AI assistant.",
        user_input
    )


# =========================================================
# 🔥 ROUTER
# =========================================================
def process_message(user_input: str):

    # EMAIL DETECTION
    if any(word in user_input.lower() for word in ["send", "email", "mail"]):

        intent = extract_intent(user_input)
        contact = lookup_contact(intent)

        if not contact:
            return {"type": "error", "message": "Contact not found"}

        # Generate FULL email (no streaming)
        body = ""
        for token in stream_email(intent, contact):
            body += token

        draft_id = str(uuid.uuid4())

        DRAFT_STORE[draft_id] = {
            "to": contact["email"],
            "subject": intent["subject"],
            "body": body
        }

        return {
            "type": "email_draft",
            "draft_id": draft_id,
            "to": contact["email"],
            "subject": intent["subject"],
            "body": body
        }

    # CHAT
    return {
        "type": "chat",
        "generator": stream_chat(user_input)
    }
def handle_email_action(draft_id: str, action: str, edit_text: str = None):

    if draft_id not in DRAFT_STORE:
        return {"type": "error", "message": "Draft not found"}

    draft = DRAFT_STORE[draft_id]

    # CANCEL
    if action == "cancel":
        del DRAFT_STORE[draft_id]
        return {"type": "info", "message": "Email cancelled"}

    # EDIT
    if action == "edit":
        if not edit_text:
            return {"type": "error", "message": "No edit instruction provided"}

        new_body = ""
        for token in stream_edit_email(draft["body"], edit_text, "professional"):
            new_body += token

        draft["body"] = new_body
        DRAFT_STORE[draft_id] = draft

        return {
            "type": "email_draft",
            "draft_id": draft_id,
            "to": draft["to"],
            "subject": draft["subject"],
            "body": new_body
        }

    # SEND
    if action == "send":
        send_email(draft["to"], draft["subject"], draft["body"])
        del DRAFT_STORE[draft_id]
        return {"type": "success", "message": "Email sent successfully"}

    return {"type": "error", "message": "Invalid action"}
# =========================================================
# 🔥 GMAIL API
# =========================================================

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
gmail_service = None

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

def init_gmail():
    global gmail_service

    creds = None

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    # If no valid credentials available
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired token...")
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES,
                redirect_uri="urn:ietf:wg:oauth:2.0:oob"
            )

            auth_url, _ = flow.authorization_url(
                prompt="consent",
                access_type="offline"
            )

            print("\n🔗 Open this URL in browser:")
            print(auth_url)

            code = input("\nPaste authorization code: ").strip()

            flow.fetch_token(code=code)
            creds = flow.credentials

        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    gmail_service = build("gmail", "v1", credentials=creds)
    print("✅ Gmail Authenticated")
def send_email(to: str, subject: str, body: str):

    if gmail_service is None:
        raise Exception("Gmail not initialized")

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    message["from"] = "Abhijith Babu"

    raw_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode()

    gmail_service.users().messages().send(
        userId="me",
        body={"raw": raw_message}
    ).execute()

    return {"status": "sent"}