import os
import time
import requests
from datetime import datetime
from twilio.rest import Client

# Twilio credentials
account_sid = 'AC16d7b9cf372c665e28e6f581b6a97690'
auth_token = '90d7076e7df9f3fcec3f874bbb39bebf'
client = Client(account_sid, auth_token)

# Recording save path
download_dir = os.path.abspath("recordings")
os.makedirs(download_dir, exist_ok=True)

# Keep track of already downloaded SIDs
downloaded_sids = set()

def download_today_recordings():
    today = datetime.now().date()
    print(f"📦 Checking recordings for {today}...")

    recordings = client.recordings.list(date_created=today)
    print(f"🟡 Found {len(recordings)} recordings")

    for recording in recordings:
        if recording.sid in downloaded_sids:
            continue

        timestamp = recording.date_created.strftime('%m-%d-%Y_%H-%M-%S')
        filename = f"{recording.sid}_{timestamp}.mp3"
        file_path = os.path.join(download_dir, filename)

        url = f"https://api.twilio.com{recording.uri.replace('.json', '.mp3')}"
        response = requests.get(url, auth=(account_sid, auth_token))

        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            downloaded_sids.add(recording.sid)
            print(f"✅ Saved: {file_path}")
        else:
            print(f"❌ Failed: {recording.sid} – HTTP {response.status_code}")

# Run this every 5 minutes in background
if __name__ == "__main__":
    while True:
        download_today_recordings()
        print("🔁 Waiting 5 minutes before next check...\n")
        time.sleep(3)  # Wait 5 minutes (300 seconds)

# recordings = client.calls("CAb8cb70f4c97b39a58f4f67684c067529").recordings.list()
# if recordings:
#     print("Recordings:")
#     for recording in recordings:
#         print(f"Recording SID: {recording.sid}")
#         print(f"Recording URL: https://api.twilio.com{recording.uri.replace('.json', '.mp3')}")
#         print(f"Duration: {recording.duration} seconds")
#         print("===================================")
# call_id = ""
# transcriptions = client.transcriptions.list()
# if transcriptions:
#     print("Transcriptions:")
#     for transcription in transcriptions:
#         if transcription.call_sid == call_sid:
#             print(f"Transcription SID: {transcription.sid}")
#             print(f"Transcription Text: {transcription.transcription_text}")
#             print("===================================")

# response = requests.get('https://api.twilio.com/2010-04-01/Accounts/AC12229307f8a979cfed90a909a8c4c05e/Recordings/REc023badd86690e02a0b6d4264e656f52.mp3', auth=(account_sid, auth_token))

# # Check if the request was successful
# if response.status_code == 200:
#     with open('downloaded_recording.mp3', 'wb') as file:
#         file.write(response.content)
#     print('Recording downloaded successfully as downloaded_recording.mp3')
# else:
#     print(f'Failed to download recording: {response.status_code} - {response.text}')






###
# # for to download recordings of today
# import os
# import requests
# from twilio.rest import Client
# from datetime import datetime, timedelta

# # Your Twilio account SID and Auth Token
# account_sid = 'AC16d7b9cf372c665e28e6f581b6a97690'
# auth_token = '05d3b43f0c0644ed02dca1379f0bf92f'

# # Initialize Twilio client
# client = Client(account_sid, auth_token)

# # Set the download directory
# download_dir = '/recordings'
# if not os.path.exists(download_dir):
#     os.makedirs(download_dir)

# # Get today's date
# today = datetime.now().date()

# # Fetch all recordings created today
# recordings = client.recordings.list(today)
# print(len(recordings))
# # Function to download recording
# def download_recording(recording_sid, recording_url):
#     response = requests.get(recording_url, auth=(account_sid, auth_token))
#     if response.status_code == 200:
#         file_path = os.path.join(download_dir, f"{recording_sid}.mp3")
#         with open(file_path, 'wb') as f:
#             f.write(response.content)
#         print(f"Downloaded: {file_path}")
#     else:
#         print(f"Failed to download: {recording_sid} - Status Code: {response.status_code}")

# # Download each recording
# for recording in recordings:
#     recording_url = f"https://api.twilio.com{recording.uri.replace('.json', '.mp3')}"
#     print(recording_url)
#     download_recording(recording.sid, recording_url)

# print("Download completed!")

