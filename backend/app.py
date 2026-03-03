from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from email_agent import (
    process_message,
    handle_email_action,
    stream_edit_email,
    send_email,
    init_gmail
)

app = FastAPI(title="Agentic Email AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    message: str


class EditRequest(BaseModel):
    original_body: str
    edit_instruction: str
    tone: Optional[str] = "professional"


class SendEmailRequest(BaseModel):
    to: str
    subject: str
    body: str

@app.post("/message")
async def message(request: MessageRequest):

    result = process_message(request.message)

    if result["type"] == "chat":

        def token_stream():
            for token in result["generator"]:
                yield token

        return StreamingResponse(token_stream(), media_type="text/plain")

    return JSONResponse(content=result)

class EmailActionRequest(BaseModel):
    draft_id: str
    action: str
    edit_text: Optional[str] = None


@app.post("/email-action")
async def email_action(request: EmailActionRequest):

    result = handle_email_action(
        request.draft_id,
        request.action,
        request.edit_text
    )

    return JSONResponse(content=result)
@app.post("/edit-email")
async def edit_email(request: EditRequest):

    def token_stream():
        for token in stream_edit_email(
            request.original_body,
            request.edit_instruction,
            request.tone
        ):
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.post("/send-email")
async def send_email_endpoint(request: SendEmailRequest):

    try:
        result = send_email(
            request.to,
            request.subject,
            request.body
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.on_event("startup")
async def startup_event():
    init_gmail()

@app.get("/")
async def root():
    return {"status": "Running 🚀"}

