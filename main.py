import os
import time
import random
import hashlib
import smtplib
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart   # ← NEW: for HTML emails
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ══════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY",      "")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY",        "")
HF_API_KEY          = os.getenv("HF_API_KEY",          "")
HF_MODEL            = os.getenv("HF_MODEL",            "black-forest-labs/FLUX.1-schnell")
HF_FALLBACK_MODEL   = os.getenv("HF_FALLBACK_MODEL",   "stabilityai/stable-diffusion-xl-base-1.0")
GOOGLE_SCRIPT_URL   = os.getenv("GOOGLE_SCRIPT_URL",   "")
SMTP_EMAIL          = os.getenv("SMTP_EMAIL",          "")
SMTP_APP_PASSWORD   = os.getenv("SMTP_APP_PASSWORD",   "").replace(" ", "")
ALLOWED_ORIGINS     = os.getenv("ALLOWED_ORIGINS",     "*").split(",")

BASE_DIR = Path(__file__).parent.resolve()

_smtp_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="smtp")

# ══════════════════════════════════════════════════════════
#  AI CLIENTS
# ══════════════════════════════════════════════════════════
gemini_client = None
groq_client   = None

if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini client ready")
    except Exception as e:
        print(f"⚠️  Gemini init failed: {e}")

if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client ready")
    except Exception as e:
        print(f"⚠️  Groq init failed: {e}")

if not SMTP_EMAIL or not SMTP_APP_PASSWORD:
    print("⚠️  SMTP not configured — email features disabled")
if not GOOGLE_SCRIPT_URL:
    print("⚠️  GOOGLE_SCRIPT_URL not set — Sheet read/write disabled")
if not HF_API_KEY:
    print("⚠️  HF_API_KEY not set — image generation disabled")

# ══════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════
app = FastAPI(title="NexusAI Backend", version="6.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_MAP = {
    "gemini-2.5-flash":     {"provider": "gemini", "model": "gemini-2.5-flash"},
    "llama-3.1-8b-instant": {"provider": "groq",   "model": "llama-3.1-8b-instant"},
}
DEFAULT_MODEL = "gemini-2.5-flash"

otp_store:   dict = {}
reset_store: dict = {}
OTP_TTL = 300


# ══════════════════════════════════════════════════════════
#  HELPERS — password / sheet
# ══════════════════════════════════════════════════════════

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def sheet_request(data: dict) -> dict:
    if not GOOGLE_SCRIPT_URL:
        return {"success": False, "message": "GOOGLE_SCRIPT_URL not configured"}
    try:
        r = requests.post(GOOGLE_SCRIPT_URL, json=data, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"success": False, "message": "Google Script timed out (15 s)"}
    except Exception as e:
        return {"success": False, "message": f"Sheet error: {e}"}


def sheet_find_user(email: str) -> dict:
    return sheet_request({"action": "login", "email": email.lower().strip()})


def sheet_save_user(name: str, email: str, password_hash: str) -> dict:
    return sheet_request({
        "action":       "saveUser",
        "name":         name,
        "email":        email.lower().strip(),
        "passwordHash": password_hash,
        "status":       "active",
    })


def sheet_update_password(email: str, password_hash: str) -> dict:
    return sheet_request({
        "action":       "updatePassword",
        "email":        email.lower().strip(),
        "passwordHash": password_hash,
    })


# ══════════════════════════════════════════════════════════
#  EMAIL TEMPLATES  (plain-text + HTML)
# ══════════════════════════════════════════════════════════

def _otp_plain(name: str, otp: str) -> str:
    return (
        f"Hi {name},\n\n"
        f"Welcome to NexusAI! 🎉\n\n"
        f"Your One-Time Password (OTP) for email verification is:\n\n"
        f"    {otp}\n\n"
        f"This OTP is valid for 5 minutes only.\n"
        f"Please do not share it with anyone.\n\n"
        f"If you did not sign up for NexusAI, simply ignore this email.\n\n"
        f"Best regards,\nThe NexusAI Team\n"
        f"─────────────────────────────────────\n"
        f"Powered by AI · Built for Everyone"
    )


def _otp_html(name: str, otp: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NexusAI OTP</title>
</head>
<body style="margin:0;padding:0;background:#0f0f1a;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f0f1a;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="540" cellpadding="0" cellspacing="0"
               style="background:linear-gradient(145deg,#1a1a2e,#16213e);
                      border-radius:16px;border:1px solid #2d2d5e;
                      overflow:hidden;max-width:540px;width:100%;">

          <!-- Header -->
          <tr>
            <td align="center"
                style="background:linear-gradient(135deg,#667eea,#764ba2);
                       padding:32px 24px;">
              <div style="font-size:32px;margin-bottom:8px;">🤖</div>
              <h1 style="margin:0;color:#ffffff;font-size:26px;font-weight:700;
                         letter-spacing:1px;">NexusAI</h1>
              <p style="margin:6px 0 0;color:rgba(255,255,255,0.85);font-size:14px;">
                Email Verification
              </p>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:36px 40px 24px;">
              <p style="margin:0 0 8px;color:#a0a0c8;font-size:14px;">Hi <strong style="color:#c8c8ff;">{name}</strong>,</p>
              <p style="margin:0 0 24px;color:#a0a0c8;font-size:14px;line-height:1.6;">
                Welcome to <strong style="color:#667eea;">NexusAI</strong>! 🎉<br/>
                Use the OTP below to verify your email address.
              </p>

              <!-- OTP Box -->
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td align="center" style="padding:8px 0 28px;">
                    <div style="display:inline-block;background:#0d0d1f;
                                border:2px solid #667eea;border-radius:12px;
                                padding:20px 48px;text-align:center;">
                      <p style="margin:0 0 4px;color:#8080b0;font-size:11px;
                                letter-spacing:2px;text-transform:uppercase;">
                        Your OTP Code
                      </p>
                      <p style="margin:0;font-size:40px;font-weight:800;
                                letter-spacing:10px;color:#ffffff;
                                font-family:'Courier New',monospace;">
                        {otp}
                      </p>
                    </div>
                  </td>
                </tr>
              </table>

              <!-- Timer badge -->
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td align="center" style="padding-bottom:28px;">
                    <span style="background:#1e1e3f;border:1px solid #3d3d6b;
                                 border-radius:20px;padding:6px 16px;
                                 color:#f0a500;font-size:12px;font-weight:600;">
                      ⏱ Valid for 5 minutes only
                    </span>
                  </td>
                </tr>
              </table>

              <!-- Warning -->
              <table width="100%" cellpadding="0" cellspacing="0"
                     style="background:#1a1a30;border-left:3px solid #f0a500;
                            border-radius:0 8px 8px 0;margin-bottom:28px;">
                <tr>
                  <td style="padding:12px 16px;color:#c0b060;font-size:12px;line-height:1.5;">
                    🔒 <strong>Never share this OTP.</strong> NexusAI will never ask for it by phone or chat.
                  </td>
                </tr>
              </table>

              <p style="margin:0;color:#606080;font-size:12px;line-height:1.6;">
                If you didn't sign up for NexusAI, you can safely ignore this email.
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td align="center"
                style="border-top:1px solid #2d2d5e;padding:20px 24px;">
              <p style="margin:0;color:#50506a;font-size:11px;">
                Powered by AI · Built for Everyone<br/>
                <span style="color:#404060;">© 2026 NexusAI</span>
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def _reset_plain(otp: str) -> str:
    return (
        f"Hi,\n\n"
        f"Your NexusAI password reset code is:\n\n"
        f"    {otp}\n\n"
        f"This code is valid for 5 minutes only.\n"
        f"If you did not request a password reset, ignore this email.\n\n"
        f"Best regards,\nThe NexusAI Team\n"
        f"─────────────────────────────────────\n"
        f"Powered by AI · Built for Everyone"
    )


def _reset_html(otp: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NexusAI Password Reset</title>
</head>
<body style="margin:0;padding:0;background:#0f0f1a;font-family:'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f0f1a;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="540" cellpadding="0" cellspacing="0"
               style="background:linear-gradient(145deg,#1a1a2e,#16213e);
                      border-radius:16px;border:1px solid #2d2d5e;
                      overflow:hidden;max-width:540px;width:100%;">

          <!-- Header -->
          <tr>
            <td align="center"
                style="background:linear-gradient(135deg,#e44d7b,#764ba2);
                       padding:32px 24px;">
              <div style="font-size:32px;margin-bottom:8px;">🔐</div>
              <h1 style="margin:0;color:#ffffff;font-size:26px;font-weight:700;
                         letter-spacing:1px;">NexusAI</h1>
              <p style="margin:6px 0 0;color:rgba(255,255,255,0.85);font-size:14px;">
                Password Reset Request
              </p>
            </td>
          </tr>

          <!-- Body -->
          <tr>
            <td style="padding:36px 40px 24px;">
              <p style="margin:0 0 24px;color:#a0a0c8;font-size:14px;line-height:1.6;">
                We received a request to reset your NexusAI password.<br/>
                Use the code below to set a new password.
              </p>

              <!-- OTP Box -->
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td align="center" style="padding:8px 0 28px;">
                    <div style="display:inline-block;background:#0d0d1f;
                                border:2px solid #e44d7b;border-radius:12px;
                                padding:20px 48px;text-align:center;">
                      <p style="margin:0 0 4px;color:#8080b0;font-size:11px;
                                letter-spacing:2px;text-transform:uppercase;">
                        Reset Code
                      </p>
                      <p style="margin:0;font-size:40px;font-weight:800;
                                letter-spacing:10px;color:#ffffff;
                                font-family:'Courier New',monospace;">
                        {otp}
                      </p>
                    </div>
                  </td>
                </tr>
              </table>

              <!-- Timer badge -->
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td align="center" style="padding-bottom:28px;">
                    <span style="background:#1e1e3f;border:1px solid #3d3d6b;
                                 border-radius:20px;padding:6px 16px;
                                 color:#f0a500;font-size:12px;font-weight:600;">
                      ⏱ Valid for 5 minutes only
                    </span>
                  </td>
                </tr>
              </table>

              <!-- Warning -->
              <table width="100%" cellpadding="0" cellspacing="0"
                     style="background:#1a1a30;border-left:3px solid #e44d7b;
                            border-radius:0 8px 8px 0;margin-bottom:28px;">
                <tr>
                  <td style="padding:12px 16px;color:#c06080;font-size:12px;line-height:1.5;">
                    🔒 <strong>Never share this code.</strong> NexusAI will never ask for it by phone or chat.
                  </td>
                </tr>
              </table>

              <p style="margin:0;color:#606080;font-size:12px;line-height:1.6;">
                If you didn't request a password reset, your account is safe — just ignore this email.
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td align="center"
                style="border-top:1px solid #2d2d5e;padding:20px 24px;">
              <p style="margin:0;color:#50506a;font-size:11px;">
                Powered by AI · Built for Everyone<br/>
                <span style="color:#404060;">© 2026 NexusAI</span>
              </p>
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


# ══════════════════════════════════════════════════════════
#  FAST ASYNC EMAIL (SMTP in thread-pool)
# ══════════════════════════════════════════════════════════

def _send_email_sync(
    to_email: str,
    subject: str,
    plain_body: str,
    html_body: str = "",
) -> tuple[bool, str]:
    if not SMTP_EMAIL or not SMTP_APP_PASSWORD:
        return False, "SMTP not configured on server"

    # Build multipart/alternative so clients show HTML when supported
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"NexusAI <{SMTP_EMAIL}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    if html_body:
        msg.attach(MIMEText(html_body, "html", "utf-8"))

    # Attempt 1 — SSL port 465
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=8) as smtp:
            smtp.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
            smtp.send_message(msg)
        print(f"📧 [SSL-465] Email sent → {to_email}: {subject}")
        return True, "Email sent"
    except smtplib.SMTPAuthenticationError:
        err = "SMTP auth failed — check your Gmail App Password formatting"
        print(f"❌ {err}")
        return False, err
    except Exception as e1:
        print(f"⚠️  SSL-465 failed ({e1}), trying STARTTLS-587…")

    # Attempt 2 — STARTTLS port 587
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=8) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
            smtp.send_message(msg)
        print(f"📧 [STARTTLS-587] Email sent → {to_email}: {subject}")
        return True, "Email sent"
    except smtplib.SMTPAuthenticationError:
        err = "SMTP auth failed on STARTTLS — check App Password"
        print(f"❌ {err}")
        return False, err
    except Exception as e2:
        err = f"Both SMTP attempts failed. SSL: {e1} | STARTTLS: {e2}"
        print(f"❌ {err}")
        return False, err


async def send_email_async(
    to_email: str,
    subject: str,
    plain_body: str,
    html_body: str = "",
):
    """Offloads execution to thread pool."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _smtp_executor, _send_email_sync, to_email, subject, plain_body, html_body
    )


# ══════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ══════════════════════════════════════════════════════════

class SendOTPRequest(BaseModel):
    name:  str
    email: str

class VerifyOTPRequest(BaseModel):
    email: str
    otp:   str

class ResendOTPRequest(BaseModel):
    email: str
    name:  Optional[str] = ""

class SavePasswordRequest(BaseModel):
    email:    str
    password: str

class LoginRequest(BaseModel):
    email:    str
    password: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    email:    str
    otp:      str
    password: str

class HistoryItem(BaseModel):
    role:    str
    content: str

class ChatRequest(BaseModel):
    message:    str
    model:      Optional[str]               = DEFAULT_MODEL
    history:    Optional[List[HistoryItem]] = []
    web_search: Optional[bool]              = False
    email:      Optional[str]               = "guest"

class ImageGenRequest(BaseModel):
    prompt: str


# ══════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════

@app.post("/send-otp")
async def api_send_otp(data: SendOTPRequest, background_tasks: BackgroundTasks):
    name  = data.name.strip()
    email = data.email.strip().lower()

    if not name or not email:
        return {"success": False, "message": "Name and email are required"}

    if not SMTP_EMAIL or not SMTP_APP_PASSWORD:
        return {"success": False, "message": "Email service is temporarily unavailable on this server"}

    existing = sheet_find_user(email)
    if existing.get("success"):
        return {"success": False, "message": "An account already exists with this email — please sign in"}

    otp = str(random.randint(100000, 999999))
    otp_store[email] = {
        "otp":        otp,
        "name":       name,
        "expires_at": time.time() + OTP_TTL,
        "verified":   False,
    }

    background_tasks.add_task(
        send_email_async,
        email,
        "Your NexusAI Verification Code",
        _otp_plain(name, otp),
        _otp_html(name, otp),
    )

    return {"success": True, "message": "OTP is being processed — check your inbox shortly"}


@app.post("/resend-otp")
async def api_resend_otp(data: ResendOTPRequest, background_tasks: BackgroundTasks):
    email   = data.email.strip().lower()
    pending = otp_store.get(email)

    if not pending:
        return {"success": False, "message": "No pending signup for this email — register again"}

    new_otp = str(random.randint(100000, 999999))
    pending["otp"]        = new_otp
    pending["expires_at"] = time.time() + OTP_TTL
    pending["verified"]   = False

    background_tasks.add_task(
        send_email_async,
        email,
        "Your NexusAI Verification Code",
        _otp_plain(pending["name"], new_otp),
        _otp_html(pending["name"], new_otp),
    )

    return {"success": True, "message": "New OTP is being dispatched"}


@app.post("/verify-otp")
def api_verify_otp(data: VerifyOTPRequest):
    email   = data.email.strip().lower()
    pending = otp_store.get(email)

    if not pending:
        return {"success": False, "message": "No pending signup found — please register again"}
    if time.time() > pending["expires_at"]:
        del otp_store[email]
        return {"success": False, "message": "OTP has expired — click Resend"}
    if data.otp.strip() != pending["otp"]:
        return {"success": False, "message": "Incorrect OTP — try again"}

    otp_store[email]["verified"] = True
    return {"success": True, "message": "OTP verified"}


@app.post("/save-password")
def api_save_password(data: SavePasswordRequest):
    email    = data.email.strip().lower()
    password = data.password
    pending  = otp_store.get(email)

    if not pending:
        return {"success": False, "message": "Session expired — please register again"}
    if not pending.get("verified"):
        return {"success": False, "message": "OTP not yet verified"}
    if len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters"}

    pw_hash = hash_password(password)
    result  = sheet_save_user(pending["name"], email, pw_hash)

    if not result.get("success"):
        return {"success": False, "message": result.get("message", "Failed to save account")}

    name = pending["name"]
    del otp_store[email]
    print(f"✅ Account created: {name} <{email}>")
    return {"success": True, "name": name, "email": email}


@app.post("/login")
def api_login(data: LoginRequest):
    email = data.email.strip().lower()
    user  = sheet_find_user(email)

    if not user.get("success"):
        return {"success": False, "message": user.get("message", "No account found with this email")}

    status = str(user.get("status") or user.get("Status", "")).lower()
    if status != "active":
        return {"success": False, "message": "Account is not active — contact support"}

    sheet_hash = (
        user.get("passwordHash")
        or user.get("passwordhash")
        or user.get("PasswordHash")
        or ""
    )
    if hash_password(data.password) != sheet_hash:
        return {"success": False, "message": "Incorrect password"}

    return {
        "success": True,
        "name":    user.get("name")  or user.get("Name",  email.split("@")[0]),
        "email":   user.get("email") or user.get("Email", email),
    }


@app.post("/forgot-password")
async def api_forgot_password(data: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    email = data.email.strip().lower()
    if not email:
        return {"success": False, "message": "Email is required"}

    user = sheet_find_user(email)
    if not user.get("success"):
        return {"success": True, "message": "If that email is registered, a reset code was sent"}

    otp = str(random.randint(100000, 999999))
    reset_store[email] = {"otp": otp, "expires_at": time.time() + OTP_TTL}

    background_tasks.add_task(
        send_email_async,
        email,
        "NexusAI Password Reset Code",
        _reset_plain(otp),
        _reset_html(otp),
    )

    return {"success": True, "message": "Reset code queued for sending"}


@app.post("/reset-password")
def api_reset_password(data: ResetPasswordRequest):
    email   = data.email.strip().lower()
    pending = reset_store.get(email)

    if not pending:
        return {"success": False, "message": "No reset request found — request a new code"}
    if time.time() > pending["expires_at"]:
        del reset_store[email]
        return {"success": False, "message": "Reset code expired — request a new one"}
    if data.otp.strip() != pending["otp"]:
        return {"success": False, "message": "Incorrect reset code — try again"}
    if len(data.password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters"}

    pw_hash = hash_password(data.password)
    result  = sheet_update_password(email, pw_hash)

    if not result.get("success"):
        return {"success": False, "message": result.get("message", "Failed to update password")}

    del reset_store[email]
    print(f"🔑 Password reset: {email}")
    return {"success": True, "message": "Password updated — you can now sign in"}


# ══════════════════════════════════════════════════════════
#  AI CHAT
# ══════════════════════════════════════════════════════════

def chat_with_ai(data: ChatRequest) -> dict:
    model_key = data.model if data.model in MODEL_MAP else DEFAULT_MODEL
    info      = MODEL_MAP[model_key]
    provider  = info["provider"]
    model_id  = info["model"]

    try:
        if provider == "gemini":
            if not gemini_client:
                msg = "❌ Gemini API key not set. Add GEMINI_API_KEY to your .env file."
                return {"response": msg, "reply": msg, "error": False}

            history_text = ""
            for h in (data.history or []):
                role = "User" if h.role == "user" else "Assistant"
                history_text += f"{role}: {h.content}\n"
            full_prompt = history_text + f"User: {data.message}\nAssistant:"

            result = gemini_client.models.generate_content(
                model=model_id, contents=full_prompt.strip()
            )
            reply = result.text

        elif provider == "groq":
            if not groq_client:
                msg = "❌ Groq API key not set. Add GROQ_API_KEY to your .env file."
                return {"response": msg, "reply": msg, "error": False}

            messages = [{"role": h.role, "content": h.content} for h in (data.history or [])]
            messages.append({"role": "user", "content": data.message})

            result = groq_client.chat.completions.create(
                model=model_id, messages=messages, max_tokens=2048
            )
            reply = result.choices[0].message.content

        else:
            return {"response": f"Unknown provider: {provider}", "error": True}

        return {"response": reply, "reply": reply, "error": False}

    except Exception as e:
        err_msg = f"⚠️ AI Error: {e}"
        print(f"❌ Chat error: {e}")
        return {"error": str(e), "reply": err_msg, "response": err_msg}


@app.post("/chat")
def chat(data: ChatRequest):
    return chat_with_ai(data)


@app.post("/api/chat")
def chat_api(data: ChatRequest):
    return chat_with_ai(data)


# ══════════════════════════════════════════════════════════
#  IMAGE GENERATION
# ══════════════════════════════════════════════════════════

def _hf_post(model: str, payload: dict, headers: dict, timeout: int) -> requests.Response:
    url = f"https://api-inference.huggingface.co/models/{model}"
    return requests.post(url, headers=headers, json=payload, timeout=timeout)


def generate_image_hf(prompt: str, retries: int = 6) -> dict:
    if not HF_API_KEY:
        return {
            "success": False,
            "message": "Image generation not configured — add HF_API_KEY to .env",
        }

    headers = {
        "Authorization":    f"Bearer {HF_API_KEY}",
        "Accept":           "image/*",
        "X-Wait-For-Model": "true",
    }
    payload = {"inputs": prompt}

    def attempt_model(model_id: str, max_tries: int) -> dict | None:
        for attempt in range(1, max_tries + 1):
            print(f"🎨 [{model_id}] attempt {attempt}/{max_tries}: {prompt[:60]}…")
            try:
                r            = _hf_post(model_id, payload, headers, timeout=180)
                content_type = r.headers.get("content-type", "application/octet-stream")

                if r.status_code == 200 and content_type.startswith("image"):
                    b64 = base64.b64encode(r.content).decode()
                    print(f"✅ [{model_id}] generated ({len(r.content) // 1024} KB)")
                    return {"success": True, "image": f"data:{content_type};base64,{b64}"}

                try:
                    err_data = r.json()
                except Exception:
                    return {"success": False, "message": f"HF error HTTP {r.status_code}"}

                if isinstance(err_data, dict) and "estimated_time" in err_data:
                    wait = min(float(err_data.get("estimated_time", 25)), 40)
                    print(f"⏳ [{model_id}] loading, waiting {wait:.0f} s…")
                    time.sleep(wait)
                    continue

                err_msg = (
                    err_data.get("error")
                    if isinstance(err_data, dict)
                    else str(err_data)
                )
                print(f"❌ [{model_id}] HF error: {err_msg}")
                return {"success": False, "message": err_msg or f"HF HTTP {r.status_code}"}

            except requests.exceptions.Timeout:
                if attempt < max_tries:
                    print(f"⏳ [{model_id}] timeout on attempt {attempt}, retrying…")
                    time.sleep(5)
                    continue
                return {
                    "success": False,
                    "message": "Request timed out — the model may be overloaded. Try again.",
                }
            except Exception as e:
                return {"success": False, "message": str(e)}

        return None

    result = attempt_model(HF_MODEL, retries)

    if result is None or (not result.get("success") and HF_FALLBACK_MODEL and HF_FALLBACK_MODEL != HF_MODEL):
        primary_msg = (result or {}).get("message", "unknown")
        print(f"⚠️  Primary model failed ({primary_msg}), trying fallback: {HF_FALLBACK_MODEL}")
        fallback = attempt_model(HF_FALLBACK_MODEL, 3)
        if fallback and fallback.get("success"):
            return fallback
        if result and not result.get("success"):
            return result
        return fallback or {
            "success": False,
            "message": "Both primary and fallback models failed. Try again in 30 s.",
        }

    if result is None:
        return {
            "success": False,
            "message": "Model is still warming up on Hugging Face. Wait 30 s and try again.",
        }

    return result


@app.post("/generate-image")
def api_generate_image(data: ImageGenRequest):
    prompt = (data.prompt or "").strip()
    if not prompt:
        return {"success": False, "message": "Prompt cannot be empty"}
    if len(prompt) > 500:
        return {"success": False, "message": "Prompt too long — keep it under 500 characters"}
    return generate_image_hf(prompt)


# ══════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status":          "ok",
        "version":         "6.0",
        "gemini":          gemini_client is not None,
        "groq":            groq_client   is not None,
        "hf":              bool(HF_API_KEY),
        "hf_model":        HF_MODEL,
        "hf_fallback":     HF_FALLBACK_MODEL,
        "smtp":            bool(SMTP_EMAIL and SMTP_APP_PASSWORD),
        "sheet":           bool(GOOGLE_SCRIPT_URL),
        "pending_otps":    len(otp_store),
        "pending_resets":  len(reset_store),
    }


# ══════════════════════════════════════════════════════════
#  SERVE FRONTEND
# ══════════════════════════════════════════════════════════

@app.get("/")
def serve_frontend():
    p = BASE_DIR / "index.html"
    if p.exists():
        return FileResponse(str(p))
    return JSONResponse({"message": "NexusAI API v6 — frontend not bundled here"})


@app.get("/{file_path:path}")
def serve_static(file_path: str):
    p = BASE_DIR / file_path
    if p.exists() and p.is_file():
        return FileResponse(str(p))
    return JSONResponse({"error": f"Not found: {file_path}"}, status_code=404)


# ══════════════════════════════════════════════════════════
#  ENTRYPOINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n🚀 NexusAI Backend v6")
    print(f"📁 Serving from: {BASE_DIR}")
    print(f"🌐 Open:         http://127.0.0.1:5000\n")
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)