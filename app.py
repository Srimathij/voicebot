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


# Load .env variables
load_dotenv()

load_dotenv()
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

# Twilio & OpenAI config
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = "+13392373131"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")


client_ai = OpenAI(api_key=OPENAI_API_KEY)
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
MAX_HISTORY_TOKENS = 9000
call_log = []

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

@app.route('/trigger-call', methods=['POST'])
def trigger_call():
    data = request.get_json()
    to_number = data.get('phone')
    name = data.get('name')
    plan = data.get('plan')
    due_date = data.get('dueDate')
    amount = data.get('amount')
    policy = data.get('policy')
    currency = data.get('currency')
    last_4 = policy[-4:] if policy else "****"

    query = urllib.parse.urlencode({
        "name": name,
        "plan_name": plan,
        "due_date": due_date,
        "premium_amount": amount,
        "last_4_digit": last_4,
        "currency": currency
    })
    voicebot_url = f"https://7081-157-49-97-197.ngrok-free.app/voicebot?{query}"

    print("[INFO] Attempting call to:", to_number)
    print("[INFO] Voicebot URL:", voicebot_url)

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



        print("[SUCCESS] Call triggered. SID:", call.sid)

        call_log.append({
            "sid": call.sid,
            "timestamp": datetime.now().isoformat(),
            "phone": to_number,
            "policy": policy,
            "status": "completed",
            "duration": 1,
            "recordingUrl": None,
            "name": name  # ✅ Add this line

        })


        return jsonify({'message': 'Call triggered', 'sid': call.sid})

    except Exception as e:
        print("[ERROR] Twilio call failed:", e)
        return jsonify({'error': str(e)}), 500
    

#####status

@app.route('/call-status', methods=['POST'])
def call_status():
    call_sid    = request.form['CallSid']
    call_status = request.form['CallStatus']
    # update status
    for c in call_log:
        if c['sid'] == call_sid:
            c['status'] = call_status
            # if it just completed, fetch the real duration
            if call_status == 'completed':
                tw_call = twilio_client.calls(call_sid).fetch()
                # tw_call.duration is in seconds
                c['duration'] = int(tw_call.duration or 0)
            break
    return ('', 204)


    

#####pdf

import glob
from datetime import datetime
from flask import jsonify, send_file
from fpdf import FPDF

@app.route("/generate-report", methods=["POST"])
def generate_report():
    data = request.get_json()
    start = datetime.strptime(data['startDate'], '%Y-%m-%d').date()
    end = datetime.strptime(data['endDate'], '%Y-%m-%d').date()
    rpt_type = data["reportType"]

    # collect recordings in range
    recs = []
    for path in glob.glob("recordings/*.mp3"):
        fname = os.path.basename(path)
        sid, datepart, _ = fname.split("_", 2)
        file_date = datetime.strptime(datepart, "%m-%d-%Y").date()
        if start <= file_date <= end:
            entry = next((c for c in call_log if c["sid"] == sid), {})
            recs.append({**entry, "file": path})

    if not recs:
        return jsonify({"error": "No recordings found"}), 404

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"{rpt_type.capitalize()} from {start} to {end}", ln=True, align='C')
    pdf.ln(5)

    customer_idx = 1
    for r in recs:
        # Build your display label
        display_label = r.get('name') or r.get('phone') or r.get('sid') 

        # Print “Customer 1:”, “Customer 2:”, etc.
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, f"Customer {customer_idx}: {display_label}", ln=True)

        # … rest of loop …
        customer_idx += 1

        pdf.set_font("Arial", '', 10)

        # Using transcription for summaries
        transcript = transcribe_audio(r["file"])
        summary = summarize_with_groq(transcript)

        # Adding the formatted summary
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(0, 8, "Summary:", ln=True)
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 6, summary)
        
        pdf.ln(5)

    # Save PDF
    os.makedirs("reports", exist_ok=True)
    out = f"reports/report_{start}_{end}.pdf"
    pdf.output(out)

    return send_file(out, as_attachment=True)

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


# ————— Helpers —————
def transcribe_audio(filepath):
    """Transcribe an MP3 file using Groq Whisper."""
    try:
        # use basename directly
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            resp = client.audio.transcriptions.create(
                file=(filename, f.read()),
                model="whisper-large-v3",
                response_format="verbose_json"
            )
        return resp.text
    except Exception as e:
        print(f"[❌] Transcription failed for {filepath}: {e}")
        return ""

def summarize_with_groq(text):
    prompt = f"""
    You are a professional call summarizer.  You will be given a transcript (which may be brief).  Produce a concise, polished summary in four sections:

    1. Opening – State the purpose of the call.  
    2. Customer perspective – Describe what the customer said or asked.  
    3. Key points & decisions – Summarize any main points, decisions or commitments.  
    4. Action items – List any follow‑up steps, with responsible parties and due dates if known.

    If the transcript is empty or contains no customer speech, simply note that the customer did not engage, but still fill out all four sections.  Do not ask for more transcript or repeat instructions.  

    Transcript:

{text}
"""
    resp = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7,
        max_completion_tokens=512,
        top_p=1,
    )
    return resp.choices[0].message.content.strip()


#####
@app.route('/recording-saved', methods=['POST'])
def recording_saved():
    call_sid      = request.form['CallSid']
    recording_url = request.form['RecordingUrl']
    duration      = request.form['RecordingDuration']    # seconds as string

    for c in call_log:
        if c['sid'] == call_sid:
            c['recordingUrl'] = recording_url
            c['duration']     = int(duration or 0)
            break

    return ('', 204)

######
#     
    
@app.route('/call-stats', methods=['GET'])
def call_stats():
    today = datetime.now().date()

    # only calls triggered today
    today_calls = [
        c for c in call_log
        if datetime.fromisoformat(c["timestamp"]).date() == today
    ]

    total = len(today_calls)

    # “successful” — we have a recordingUrl (i.e. bot spoke & recorded)
    successful = sum(1 for c in today_calls if c.get("recordingUrl"))

    # everything else is “failed”
    successful = total - successful

    # average duration of those that did record
    durations = [c.get("duration", 0) for c in today_calls if c.get("duration")]
    avg_secs = int(sum(durations) / len(durations)) if durations else 0
    avg_duration = f"{avg_secs // 60}:{avg_secs % 60:02d}"

    return jsonify({
        "today":      total,
        "successful": successful,
        "failed":     0,
        "duration":   avg_duration
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
        speechTimeout=1,
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
    if session['fallback_count'] >= 50:
        response.say("We're unable to hear you. We'll try again later. Goodbye.")
        response.hangup()
    else:
        gather = Gather(action='/openaires', input='speech', speech_model='phone_call',
                        speechTimeout=0.5, actionOnEmptyResult=True)
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
    if any(kw in speech_result.lower() for kw in ['bye', 'goodbye', 'exit', 'hang up', 'nothing']):
        goodbye_msg = f"My pleasure speaking with you, Mr./Ms. {name}. For other concerns, feel free to reach out to us via email at customercare@allianzpnblife.ph or call us at 8818-4357.Thank you for choosing Allianz PNB Life as your insurance partner. Have a good day ahead!"
        print(f"[BOT] Ava replied: {goodbye_msg}")
        response.say(goodbye_msg)
        response.hangup()
        return str(response)

    history = session.get('history', [])    
    prompt = f"""
    You are a voice assistant for Allianz PNB Life, and your name is Ava. You assist users with their queries in a professional, natural, and dynamic manner based on the script provided. You act confidently and intelligently to interpret user responses and provide relevant information, especially about their premium payment status.

    🧩 GENERAL PRINCIPLES:
    - Always be concise, friendly, and professional.
    - If the user indicates they are unavailable, busy, or it is a wrong number, respond with:
        "Thank you for your time. We will call back another time."
    - If the user states, "I'm not that person" or similar, respond:
        "Thank you for letting us know. I won’t share any confidential policy details. We will call back another time."
    - If the user repeatedly says they are the wrong person, vary the response slightly to maintain a natural tone.
    - Do not share any confidential policy or account details until identity is confirmed.

    🧩 GREETING:
    - Start by saying
        "Hi {name}. Please be aware that this call may be recorded for security and quality assurance purposes. We wish to remind you of your premium payment for your {plan_name} policy with policy number ending in {last_4_digit}.
        May we ask you to kindly pay your premium of {premium_amount} {cur} on or before {due_date} to keep your policy active and enjoy continuous coverage.Would you like to know more about payment options?"

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

    🧩 CLOSING THE CALL:

    If user expresses thanks , bye , ending the call or confirms action:
    - Respond:
        "My pleasure speaking with you, Mr./Ms. {name}. For other concerns, feel free to reach out to us via email at customercare@allianzpnblife.ph or call us at 8818-4357.
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
                    speechTimeout=1, actionOnEmptyResult=True)
    response.append(gather)
    return str(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
