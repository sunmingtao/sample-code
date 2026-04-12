from google_auth_oauthlib.flow import InstalledAppFlow
import os.path
import pickle
import base64
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText

# Email variables
SENDER = "admin@gmail.com"
RECIPIENT = "msun@nla.gov.au"
SUBJECT = "Hello from Python with OAuth"
BODY_TEXT = "This is an email sent by Python using OAuth 2.0. It's secure!"

# Path to your credentials JSON file
CREDENTIALS_FILE = 'gmail-auth.json'
TOKEN_FILE = 'token.pickle'

def send_email(recipient, subject, body_text):
    # Load credentials or get new ones if they don't exist
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE,
                                                         scopes=['https://www.googleapis.com/auth/gmail.send'])
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    try:
        # Build the service from the credentials
        service = build('gmail', 'v1', credentials=creds)
        message = MIMEText(body_text)
        message['to'] = recipient
        message['from'] = SENDER
        message['subject'] = subject
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        raw_message = {'raw': raw}
        message = service.users().messages().send(userId='me', body=raw_message).execute()
        print('Message Id: %s' % message['id'])
    except HttpError as error:
        print(f'An error occurred: {error}')

if __name__ == "__main__":
    send_email(RECIPIENT, SUBJECT, BODY_TEXT)


