import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import traceback
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

def test_email():
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    email_address = os.getenv("EMAIL_ADDRESS")
    email_password = os.getenv("EMAIL_PASSWORD")
    to_email = email_address # Send to self
    
    if not email_address or not email_password:
        print("ERROR: EMAIL_ADDRESS or EMAIL_PASSWORD not found in .env file.")
        return False
    
    print(f"Attempting to send test email to {to_email}...")
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = email_address
        msg['To'] = to_email
        msg['Subject'] = "Email Service Test"
        
        html_part = MIMEText("<h1>Testing Email Service</h1><p>If you see this, the SMTP connection succeeded.</p>", 'html')
        msg.attach(html_part)
        
        print("Connecting to server...")
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
        print("TLS start...")
        server.starttls()
        print("Logging in...")
        server.login(email_address, email_password)
        
        print("Sending mail...")
        server.sendmail(email_address, to_email, msg.as_string())
        server.quit()
        
        print("SUCCESS!")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_email()
