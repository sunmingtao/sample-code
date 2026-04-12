from google_auth_oauthlib.flow import InstalledAppFlow
import os.path
import pickle
import base64
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders

# Email variables
SENDER = "admin@gmail.com"
RECIPIENT = "msun@nla.gov.au"
SUBJECT = "Hello from Python with an Attachment"
BODY_TEXT = "This is an email sent by Python using OAuth 2.0 with an image attachment."
IMAGE_PATH = 'shangtou.jpg'

# Path to your credentials JSON file
CREDENTIALS_FILE = 'gmail-auth.json'
TOKEN_FILE = 'token.pickle'

def send_email_with_attachment(recipient, subject, body_text, attachment_path):
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
        # Create a multipart email message
        message = MIMEMultipart()
        message['to'] = recipient
        message['from'] = SENDER
        message['subject'] = subject

        # Add text content
        body = MIMEText(body_text, 'plain')
        message.attach(body)

        # Attach the image
        with open(attachment_path, 'rb') as file:
            img = MIMEImage(file.read())
            img.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(attachment_path)}"')
            message.attach(img)

        # Encode the message to base64
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        raw_message = {'raw': raw}

        # Send the email via Gmail API
        service = build('gmail', 'v1', credentials=creds)
        message = service.users().messages().send(userId='me', body=raw_message).execute()
        print('Message Id: %s' % message['id'])

    except HttpError as error:
        print(f'An error occurred: {error}')

if __name__ == "__main__":
    send_email_with_attachment(RECIPIENT, SUBJECT, BODY_TEXT, IMAGE_PATH)
