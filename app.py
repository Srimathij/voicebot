import os
import urllib.parse
from flask import Flask, request, jsonify, session
from flask_session import Session
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
from flask_cors import CORS
import requests
from groq import Groq
import re

import openai

# Load .env variables
load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
client = Groq(api_key=groq_api_key)



# Flask setup
app = Flask(__name__)
CORS(app)
app.secret_key = 'outboundcalls'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
Session(app)
from flask import Flask
# from flask_cors import CORS


# Twilio & OpenAI config
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = "+13392373131"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")


client_ai = OpenAI(api_key=OPENAI_API_KEY)
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
MAX_HISTORY_TOKENS = 9000
# call_log = []

from datetime import datetime, timezone

import os
import json
from datetime import datetime

CALL_HISTORY_FILE = 'call_history.json'

def load_call_history():
    if os.path.exists(CALL_HISTORY_FILE):
        with open(CALL_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_call_history(data):
    with open(CALL_HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# ✅ Load saved call history on startup
call_log = load_call_history()


def truncate_history(history):
    total_tokens = 0
    truncated = []
    for msg in reversed(history):
        tokens = len(msg['content'].split())
        total_tokens += tokens
        if total_tokens <= MAX_HISTORY_TOKENS:
            truncated.insert(0, msg)
        else:
            break
    return truncated

def is_before_due(due_date):
    today = datetime.now().date()
    due = datetime.strptime(due_date, "%Y-%m-%d").date()
    return due > today

from flask import request, jsonify
import urllib.parse
from datetime import datetime
@app.route('/trigger-call', methods=['POST'])
def trigger_call():
    data      = request.get_json()
    to_number = data.get('phone')
    name      = data.get('name')
    plan      = data.get('plan')
    due_date  = data.get('dueDate')
    amount    = data.get('amount')
    policy    = data.get('policy')
    currency  = data.get('currency')

    # ensure policy is a string so we can slice
    policy_str = str(policy or "")
    last_4     = policy_str[-4:] if len(policy_str) >= 4 else policy_str or "****"

    # build the voicebot URL
    query = urllib.parse.urlencode({
        "name":           name,
        "plan_name":      plan,
        "due_date":       due_date,
        "premium_amount": amount,
        "last_4_digit":   last_4,
        "currency":       currency
    })
    voicebot_url = f"https://alliance-q1m5.onrender.com/voicebot?{query}"

    try:
        call = twilio_client.calls.create(
            url=voicebot_url,
            to=to_number,
            from_=TWILIO_FROM_NUMBER,
            record=True,
            recording_status_callback="/recording-saved",
            recording_status_callback_event=["completed"],
            status_callback="/call-status",
            status_callback_event=["completed"]
        )

        # 🔥 Record it in call_log for later reporting/transcript lookup
        call_log.append({
            "sid":          call.sid,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "name":         name,
            "plan":         plan,
            "due_date":     due_date,
            "amount":       amount,
            "policy":       policy,
            "currency":     currency,
            "last_4_digit": last_4,
            "phone":        to_number,
            "status":       "queued",
            "duration":     0,
            "recordingUrl": None
        })

        save_call_history(call_log)

        return jsonify({'message': 'Call triggered', 'sid': call.sid})

    except Exception as e:
        print("[ERROR] trigger_call:", e)
        return jsonify({'error': str(e)}), 500

#####s



    

#####pdf

import glob
from datetime import datetime
from flask import jsonify, send_file
# from fpdf import FPDF
import os
import glob
from datetime import datetime
from io import BytesIO
from datetime import datetime, date


from flask import jsonify, send_file
import pandas as pd
from openpyxl import Workbook  # for pandas ExcelWriter

from io import BytesIO
import pandas as pd
import glob, os
from datetime import datetime
from flask import jsonify, send_file, session


from datetime import datetime
import glob, os, re
from io import BytesIO
from flask import jsonify, send_file, request, current_app
import pandas as pd

from datetime import datetime, date
import glob, os, re
from io import BytesIO
from flask import jsonify, send_file, request, current_app
import pandas as pd

@app.route("/generate-report", methods=["POST"])
def generate_report():
    try:
        data  = request.get_json()
        start = datetime.strptime(data['startDate'], '%Y-%m-%d').date()
        end   = datetime.strptime(data['endDate'],   '%Y-%m-%d').date()
        rpt_type = data.get("reportType", "Report")

        rows = []
        rec_dir = "recordings"

        for path in glob.glob(os.path.join(rec_dir, "*.*")):
            ext = os.path.splitext(path)[1].lower()
            if ext not in (".mp3", ".wav"):
                continue

            fname  = os.path.basename(path)
            parts  = fname.split("_")
            if len(parts) < 3:
                continue

            sid, date_str, _ = parts
            try:
                file_date = datetime.strptime(date_str, "%m-%d-%Y").date()
            except ValueError:
                continue

            if not (start <= file_date <= end):
                continue

            # Transcribe:
            transcript_obj = transcribe_audio(path)
            if not transcript_obj:
                continue
            transcript_text = build_labeled_dialog(transcript_obj)

            # Prompt for Name / Policy / Due Date
            extraction_prompt = f"""
You are an information extraction assistant. Carefully review the transcript below and extract the following customer details with high accuracy:

1. The customer's **full name**  
2. The **last 4 digits** of their **policy number** (e.g., 5678)  
3. The **due date** for their next payment (acceptable formats: YYYY-MM-DD or MM/DD/YYYY)

Only return what's clearly stated in the transcript. If any detail is missing or ambiguous, mark it as "Not Found".

TRANSCRIPT:
{transcript_text}

Respond in the following format (no additional commentary):

Name: <Full Name or Not Found>  
Policy Number: <4 digits or Not Found>  
Due Date: <Date or Not Found>
"""
            resp = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=1,
                max_completion_tokens=500,
            )
            extracted = resp.choices[0].message.content.strip()

            # Regex out all three fields
            match = re.search(
                r"Name:\s*(.+?)\s*Policy Number:\s*(\*{0,3}\d{1,4}|Not Found)\s*Due Date:\s*([0-9/\-]+|Not Found)",
                extracted, re.IGNORECASE
            )
            if match:
                name    = match.group(1).strip()
                last_4  = match.group(2).strip()
                due_str = match.group(3).strip()

                if name.lower()   == "not found": name   = None
                if last_4.lower() == "not found": last_4 = None

                # Try parsing due date
                due_date = None
                if due_str.lower() != "not found":
                    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
                        try:
                            due_date = datetime.strptime(due_str, fmt).date()
                            break
                        except ValueError:
                            continue
            else:
                name, last_4, due_date = None, None, None

            # Fallback on call_log
            if not name or not last_4 or not due_date:
                entry = next((c for c in call_log if c.get("sid") == sid), {})
                name     = name     or entry.get("name", "Customer")
                last_4   = last_4   or entry.get("policy", "****")
                due_date = due_date or entry.get("due_date")

            # Summarize actions
            actions = summarize_with_groq(transcript_text, name, last_4)

            # Format due date for Excel
            if isinstance(due_date, date):
                due_val = due_date.strftime("%Y-%m-%d")
            elif isinstance(due_date, str):
                due_val = due_date
            else:
                due_val = "Not Found"

            rows.append({
                "Customer Name": name,
                "Policy Number": last_4,
                "Due Date": due_val,
                "Actions": actions
            })

        if not rows:
            return jsonify({"error": "No recordings found"}), 404

        # Build Excel
        df = pd.DataFrame(rows, columns=["Customer Name","Policy Number","Due Date","Actions"])
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=rpt_type)
        output.seek(0)

        filename = f"{rpt_type}_{start}_{end}.xlsx"
        return send_file(
            output,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        current_app.logger.error("Report generation failed", exc_info=e)
        return jsonify({"error": str(e)}), 500
#####llm####

import os
import glob
import io           # ← Add this line
import zipfile      # ← Add this line


@app.route("/download-recordings", methods=["POST"])
def download_recordings():
    data = request.get_json() or {}
    start = datetime.strptime(data.get('startDate',''), '%Y-%m-%d').date()
    end   = datetime.strptime(data.get('endDate',''),   '%Y-%m-%d').date()

    # gather .mp3s
    files = []
    for path in glob.glob("recordings/*.mp3"):
        fname     = os.path.basename(path)
        sid, datep, _ = fname.split("_", 2)
        file_date = datetime.strptime(datep, "%m-%d-%Y").date()
        if start <= file_date <= end:
            files.append(path)

    if not files:
        return jsonify({"error": "No recordings found"}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for mp3 in files:
            zf.write(mp3, arcname=os.path.basename(mp3))
    buf.seek(0)

    return send_file(
        buf,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"recordings_{start}_{end}.zip"
    )


#####trans

import os
import glob
import io
import zipfile
from datetime import datetime
from flask import jsonify, send_file
from fpdf import FPDF

import os
import glob
import io
import zipfile
from datetime import datetime
from flask import jsonify, send_file
from fpdf import FPDF

import os
import glob
import io
import zipfile
from datetime import datetime
from flask import jsonify, send_file, request, session
from fpdf import FPDF

# ── 1) Verbose transcription helper ────────────────────────────────────────────

def transcribe_audio(filepath):
    """
    Transcribe an MP3 file using Groq Whisper (verbose JSON).
    """
    try:
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            resp = client.audio.transcriptions.create(
                file=(filename, f.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json"
            )
        return resp
    except Exception as e:
        print(f"[❌] Transcription failed for {filepath}: {e}")
        return None

# ── Extract Customer Name ───────────────────────────────────────────────────────

def extract_customer_name(transcript_text):
    """
    Extract customer name from lines like: "May I speak to Srimathi please?"
    """
    pattern = re.compile(
        r"may\s+i\s+speak\s+to\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)(?:,?\s+please)?\??",
        re.IGNORECASE
    )
    for line in transcript_text.splitlines():
        if line.strip().startswith("Ava:"):
            match = pattern.search(line)
            if match:
                return match.group(1).strip().title()
    return None


##
def build_labeled_dialog(whisper_json):
    """
    Label each segment as Ava vs. Customer based on:
      • AI‐style keywords (explicit markers)
      • OR segment duration >= 4 seconds
      • OR text length > 8 words
    Otherwise it’s Customer.
    """
    segments = getattr(whisper_json, "segments", []) or []
    lines = []

    # Phrases definitely from the bot
    ai_indicators = [
        "greetings", "this is eva", "virtual assistant",
        "you're speaking with an ai", "may i speak to",
        "for quality and training purposes",
        "please be aware that this call may be recorded",
        "thank you for choosing allianz", "customercare@allianzpnblife.ph",
        "we will call back another time"
    ]

    for seg in segments:
        text       = seg.get("text", "").strip()
        if not text:
            continue

        lower      = text.lower()
        word_count = len(text.split())
        # Whisper “verbose_json” gives start/end timestamps
        start      = seg.get("start", 0.0)
        end        = seg.get("end", 0.0)
        duration   = end - start

        # Decide speaker
        if (
            any(k in lower for k in ai_indicators)
            or duration >= 4.0
            or word_count > 8
        ):
            speaker = "Ava"
        else:
            speaker = "Customer"

        lines.append(f"{speaker}: {text}")

    return "\n".join(lines)

# ── Transcript Download Endpoint ────────────────────────────────────────────────

import glob, io, zipfile
from datetime import datetime
from flask import jsonify, send_file, request
from fpdf import FPDF

@app.route("/download-transcript", methods=["POST"])
def download_transcripts():
    data = request.get_json() or {}
    start = datetime.strptime(data.get('startDate',''), '%Y-%m-%d').date()
    end   = datetime.strptime(data.get('endDate',''),   '%Y-%m-%d').date()

    # gather matching .mp3 files
    mp3_files = []
    for path in glob.glob("recordings/**/*.mp3", recursive=True):
        fname = os.path.basename(path)
        try:
            sid, datep, _ = fname.split("_", 2)
            file_date = datetime.strptime(datep, "%m-%d-%Y").date()
            if start <= file_date <= end:
                mp3_files.append(path)
        except ValueError:
            continue

    if not mp3_files:
        return jsonify({"error": "No recordings found for transcripts"}), 404

    # build a ZIP of PDFs
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for mp3_path in mp3_files:
            # 1) Transcribe to verbose JSON
            whisper_json = transcribe_audio(mp3_path)
            if whisper_json:
                # 2) Label each segment as Ava vs Customer
                transcript_text = build_labeled_dialog(whisper_json)
            else:
                transcript_text = "[No transcript available]"

            # 3) Pull real customer name from the “May I speak to …” line
            customer_name = extract_customer_name(transcript_text) or "Customer"

            # 4) Safely replace "Customer:" → "<Name>:"
            if customer_name.lower() != "ava":
                personalized = transcript_text.replace("Customer:", f"{customer_name}:")
            else:
                personalized = transcript_text

            # 5) Use these labels directly—no more turn alternation
            final_transcript = personalized

            # 6) Render into a one‐page PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            for line in final_transcript.splitlines():
                pdf.multi_cell(0, 8, line)

            pdf_bytes = pdf.output(dest='S').encode('latin1')
            pdf_name = os.path.basename(mp3_path).rsplit(".", 1)[0] + ".pdf"
            zf.writestr(pdf_name, pdf_bytes)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"transcripts_{start}_to_{end}.zip"
    )
def summarize_with_groq(text, name, last_4_digit):
    prompt = f"""
You are a call summarization assistant.

You will receive a transcript of a customer support call, along with the customer’s name and the last 4 digits of their policy number.

🎯 Your task is to extract **only the final action or intention expressed by the customer** at the end of the call.

✅ Focus on clearly identifying **what the customer decided, requested, or intended**, such as:

- intends to pay
- already paid
- declined to pay
- asked for more information
- not the correct person
- did not respond or engage

📌 Output a single, short sentence describing the **next action or the customer’s final intent** — no summaries of the entire call, no extra commentary.

🛑 If the customer said nothing or didn’t participate, respond exactly with:  
**Customer did not respond or engage.**

🧾 Customer Name: {name}  
🔐 Policy Number: {last_4_digit}  

📞 Transcript:  
{text}

🎬 Based on this, respond with one single sentence describing only the final action point or intent.
"""
    resp = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7,
        max_completion_tokens=500,
        top_p=1.0,
    )
    return resp.choices[0].message.content.strip()

#####
@app.route('/recording-saved', methods=['POST'])
def recording_saved():
    call_sid      = request.form['CallSid']
    recording_sid = request.form['RecordingSid']
    recording_url = request.form['RecordingUrl']
    duration      = request.form['RecordingDuration']

    # download MP3
    os.makedirs("recordings", exist_ok=True)
    resp = requests.get(recording_url)
    fname = f"recordings/{call_sid}_{datetime.now():%m-%d-%Y}.mp3"
    with open(fname,"wb") as f: f.write(resp.content)

    # update log
    for c in call_log:
        if c['sid']==call_sid:
            c.update(recordingUrl=recording_url, duration=int(duration), recording_sid=recording_sid)
            break
    return ('',204)
######
#    
# 
# 
@app.route('/call-status', methods=['POST'])
def call_status():
    call_sid    = request.form['CallSid']
    call_status = request.form['CallStatus']
    for c in call_log:
        if c['sid'] == call_sid:
            c['status'] = call_status
            if call_status == 'completed':
                tw_call = twilio_client.calls(call_sid).fetch()
                c['duration'] = int(tw_call.duration or 0)
            break
    return ('', 204)


from datetime import datetime, timezone

@app.route('/call-stats', methods=['GET'])
def call_stats():
    today = datetime.now(timezone.utc).date()  # Use UTC to match stored timestamps

    # Filter calls triggered today (by UTC date)
    today_calls = [
        c for c in call_log
        if datetime.fromisoformat(c["timestamp"]).date() == today
    ]

    total = len(today_calls)

    # Successful calls are those that have a recording URL
    success_count = sum(1 for c in today_calls if c.get("recordingUrl"))

    # Failed calls are the rest
    success_count = total - success_count

    # Calculate average duration (only for successful/recorded calls)
    durations = [c.get("duration", 0) for c in today_calls if c.get("duration")]
    avg_secs = int(sum(durations) / len(durations)) if durations else 0
    avg_duration = f"{avg_secs // 60}:{avg_secs % 60:02d}"

    return jsonify({
        "today": total,
        "successful": success_count,
        "failed": 0,
        "duration": avg_duration
    })
 


######
@app.route('/recent-activity', methods=['GET'])
def recent_activity():
    sorted_log = sorted(call_log, key=lambda x: x["timestamp"], reverse=True)
    return jsonify([
        {
            "phone": entry.get("phone", "Unknown"),
            "timestamp": entry.get("timestamp"),
            "status": entry.get("status", "queued")
        }
        for entry in sorted_log[:5]
    ])

@app.route('/call-history', methods=['GET'])
def call_history():
    return jsonify(sorted(call_log, key=lambda x: x["timestamp"], reverse=True))


@app.route('/voicebot', methods=['POST'])
def voicebot():
    session['name'] = request.args.get('name', 'Customer')
    session['plan_name'] = request.args.get('plan_name', 'Plan')
    session['premium_amount'] = request.args.get('premium_amount', '0')
    session['due_date'] = request.args.get('due_date', '2025-01-01')
    session['last_4_digit'] = request.args.get('last_4_digit', '****')
    session['currency'] = request.args.get("currency", "PHP")
    session['fallback_count'] = 0
    session['history'] = []

    print(f"[INFO] Voicebot started for: {session['name']}, Plan: {session['plan_name']}, Due: {session['due_date']}, Amount: {session['premium_amount']} {session['currency']}")

    return str(welcome())

@app.route("/welcome", methods=['POST'])
def welcome():
    response = VoiceResponse()
    name = session.get('name', 'Customer')
    response.say(
        f"Greetings. This is Eva, your virtual assistant from Allianz PNB Life. "
        f"Just to let you know, you’re speaking with an AI, and this call may be recorded for quality and training purposes. "
        f"May I speak to {name} please?"
    )




    gather = Gather(
        action='/openaires',
        input='speech',
        speech_model='phone_call',
        speechTimeout=3,
        actionOnEmptyResult=True
    )
    response.append(gather)

    # Debug logging to validate TwiML output
    print("[DEBUG] TwiML from /welcome:")
    print(str(response))

    return str(response)  # ✅ Required for Twilio to parse XML properly

@app.route("/fallback", methods=['POST'])
def fallback():
    response = VoiceResponse()
    session['fallback_count'] += 1
    if session['fallback_count'] >= 100:
        response.say("We're unable to hear you. We'll try again later. Goodbye.")
        response.hangup()
    else:
        gather = Gather(action='/openaires', input='speech', speech_model='phone_call',
                        speechTimeout=0.1, actionOnEmptyResult=True)
        response.append(gather)
    return str(response)

@app.route("/openaires", methods=['POST'])
def chatbot_res():
    response = VoiceResponse()
    speech_result = request.values.get('SpeechResult', '').strip()

    if not speech_result:
        return str(response.redirect('/fallback'))

    # Get session data
    name = session.get('name', 'Customer')
    last_4_digit = session.get('last_4_digit', '****')
    plan_name = session.get('plan_name', 'your plan')
    premium_amount = session.get('premium_amount', '0')
    cur = session.get("currency", "currency")
    due_date = session.get('due_date', 'Unknown')

    # Log what the user said
    print(f"[USER] {name} said: {speech_result}")

    # End if user wants to exit
    # Exit keywords—including “thank you”/“thanks”
    exit_keywords = [
        'bye',
        'goodbye',
        'exit',
        'hang up',
        'nothing',
        'thank you',
        'thanks',
        'no thanks'
    ]

    # If user said any exit keyword, say goodbye and hang up immediately
    if any(kw in speech_result for kw in exit_keywords):
        name = session.get('name', 'Customer')
        goodbye_msg = (
            f"My pleasure speaking with you, {name}. "
            "For other concerns, feel free to reach out to us via email at "
            "customercare@allianzpnblife.ph or call us at 8818-4357. "
            "Thank you for choosing Allianz PNB Life as your insurance partner. "
            "Have a good day ahead!"
        )
        response.say(goodbye_msg)
        response.hangup()
        return str(response)


    history = session.get('history', [])    
    prompt = f"""

        You are a voice assistant for Allianz PNB Life, and your name is Ava. You assist users with their queries in a professional, natural, and dynamic manner based on the script provided. You act confidently and intelligently to interpret user responses and provide relevant information, especially about their premium payment status.
    🧩 IMPORTANT FORMAT NOTE:
        When reading out policy numbers (or any numeric code), say each digit individually.  
        For example, “5678” should be spoken as “five, six, seven, eight.”

    🧩 NAME PRONUNCIATION RULES:

    - Always pronounce names as **whole names**, never letter-by-letter unless the user spells it themselves.
    - Avoid interpreting alphabetic names (e.g., "Christiane") as acronyms. Do not say: “C-H-R-I-S-T-I-A-N-E”.
    - Speak names naturally and smoothly, preserving respectful intonation.

        🔊 Examples:
        - “Christiane” → say “Chris-tee-ahn” (not “C-H-R-I-S-T-I-A-N-E”)
        - “Joaquin” → say “Wah-keen”
        - “Ma. Theresa” → say “Ma Theresa” as “Mah Teh-reh-sah”
        - “Juan_Dela_Cruz” → convert to “Juan Dela Cruz” using:
            spoken_name = name.replace("_", " ")

        
    🧩 DATA FORMATTING (APPLY BEFORE SPEAKING):
    - Format plan name to avoid underscores and ensure natural speech:
        spoken_plan_name = plan_name.replace("_", " ").title()

        🔊 Examples:
        - "allianz_score" → "Allianz Score"
        - "wealth_accumulator" → "Wealth Accumulator"

    
    🧩 MISIDENTIFICATION HANDLING:
    If the user says they are **not the intended person**, or says anything like:
    ["not me", "wrong person", "i'm not that person", "you have the wrong number", "that's not me"]

    → Respond:
    "Sorry for the confusion. I won’t share any personal details. We’ll call back another time. Thank you for your time!"

    → If user says similar phrases repeatedly, vary the response:
    - "Apologies for the mix-up. We’ll try to reach the correct person later. Have a good day!"
    - "Thank you for letting us know. No worries — we’ll follow up with the correct contact another time."

    → **Never mention the user's name, plan, or policy information** in any "wrong person" scenario.


    
        
    🧩 VOICE PRONUNCIATION INSTRUCTIONS:
        - Speak naturally, with a calm, respectful, and warm tone.
        - Ensure smooth phrasing and emphasize important words clearly.


    🧩 CURRENCY PRONUNCIATION:
    - Always say "Philippine Peso" instead of the currency code "PHP".
    - For US Dollar, say "US Dollar" instead of "USD".

    - Format plan name:
        spoken_plan_name = plan_name.replace("_", " ").title()


    🧩 GENERAL PRINCIPLES:
    - Always be concise, friendly, and professional.
    - If the user indicates they are unavailable, busy, or it is a wrong number, respond with:
        "Thank you for your time. We will call back another time."
    - If the user states, "I'm not that person" or similar, respond:
        "Thank you for letting us know. I won’t share any confidential policy details. We will call back another time."
    - If the user repeatedly says they are the wrong person, vary the response slightly to maintain a natural tone.
    - Do not share any confidential policy or account details until identity is confirmed.

    
    🧩 GREETING LOGIC:
    Start by saying : If user says any of the following phrases to indicate they are speaking:
    ["speaking", "yes speaking", "yes, speaking", "this is speaking", "i am speaking"]

    Respond:
    "Hi {name}. Please be aware that this call may be recorded for security and quality assurance purposes. We wish to remind you of your premium payment for your {plan_name} policy with policy number ending in {last_4_digit} (read as individual digits).  
    May we ask you to kindly pay your premium of {premium_amount} {cur} on or before {due_date} to keep your policy active and enjoy continuous coverage? Would you like to know more about payment options?"


    If speaking to the policyowner:
    - Continue based on due date status (before or after due date).

    🧩 DUE DATE STATUS:

    If BEFORE the due date:
    - Respond:
        "Hi {name}. Please be aware that this call may be recorded for security and quality assurance purposes. We wish to remind you of your premium payment for your {plan_name} policy with policy number ending in {last_4_digit}.
        May we ask you to kindly pay your premium of {premium_amount} {cur} on or before {due_date} to keep your policy active and enjoy continuous coverage.Would you like to know more about payment options?"

    If AFTER the due date:
    - Respond:
        "Hi {name}. Please be aware that this call may be recorded for security and quality assurance purposes. We wish to remind you of your premium payment for your {plan_name} policy with policy number ending in {last_4_digit}.
        You have missed your payment of {premium_amount} {cur} which was due last {due_date}.
        To keep your policy active and enjoy continuous coverage, may we kindly ask you to pay your premium on or before [31 days after due date – actual date]Would you like to know more about payment options??"

    🧩 CUSTOMER RESPONSES:

    If user acknowledges payment delay:
    - Provide payment options:
        "You may pay your premium over-the-counter through our partner banks PNB, Metrobank, BDO, or Cebuana Lhuillier for Philippine Peso policies.
        Online payments are available through BDO Online, PNB Mobile Banking, PNB Internet Banking, and Metrobank Direct Online (for both Peso and Dollar policies).
        For Philippine Peso policies, you may also use Metrobank Mobile App, GCash, Maya, or Bancnet Online.
        You may also log into your Allianz Touch account and pay via PayNow.
        Kindly remember: our agents and financial advisors are not authorized to receive payments directly."

    If user informs payment has been made:
    - Acknowledge:
        "Thank you for that information. Kindly disregard this reminder if payment has already been made.
        For policies under auto-pay, an automatic rebilling will occur in the following days. Kindly ensure your account is active and well-funded."

    If user asks about Allianz Touch:
    - Respond:
        "Simply go to www.touch.allianzpnblife.ph and use the email address you provided during your policy application when creating or logging into your account."

    If user asks to repeat the Allianz Touch link:
    - Repeat:
        "It’s www.touch.allianzpnblife.ph. Reminder to use the email address you provided during your policy application."

    🧩 END-OF-CALL DETECTION:

    - If user clearly wants to end the call (e.g., says "no", "no thanks", "nothing", etc. when asked if they want more info):
        → "My pleasure speaking with you, {name}. For other concerns, feel free to reach out to us via email at customercare@allianzpnblife.ph or call us at 8818-4357.  
        Thank you for choosing Allianz PNB Life as your insurance partner. Have a good day ahead!"

    - If user says thank you, goodbye, or confirms completion:
        → "My pleasure speaking with you, {name}. For other concerns, feel free to reach out to us via email at customercare@allianzpnblife.ph or call us at 8818-4357.  
        Thank you for choosing Allianz PNB Life as your insurance partner. Have a good day ahead!"

    Always end the call gracefully and professionally.

    📝 NOTE: IMPORTANT BEHAVIORS
    - Act like an intelligent voice agent—analyze, decide, and respond smartly.
    - Recognize English phrases accurately.
    - If a user speaks Tagalog or an unfamiliar language, respond politely without admitting inability to understand: 
        "Thank you for your response. As a reminder, we are calling regarding your policy. If you have any questions, feel free to reach out to our customer service channels."
    - If user says they cannot pay now:
        - Respond:
            "I understand. May I ask why you’re unable to make the payment today? This will help me direct you to the right assistance."

    - Understand what user is speaking and respond accordingly.

    - Always normalize speech:
        - Never pronounce special characters like underscores or dashes.
        - plan_name like "allianz_score" → "Allianz Score"
        - cur like "PHP" → "Philippine Peso", "USD" → "US Dollar"
        - Speak policy digits clearly: "five, six, seven, eight"

        
    - Recognize English phrases accurately, especially greeting confirmations like:", "this is speaking", etc.

     """

    history.insert(0, {"role": "system", "content": prompt})
    history.append({"role": "user", "content": speech_result})

    completion = client_ai.chat.completions.create(
        model="gpt-4o",
        messages=truncate_history(history),
        temperature=0.7,
        max_tokens=1024,
    )

    reply = completion.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    session['history'] = history[-6:]

    response.say(reply)
    gather = Gather(action='/openaires', input='speech', speech_model='phone_call',
                    speechTimeout=0.1, actionOnEmptyResult=True)
    response.append(gather)
    return str(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
