from email_service import EmailService
import os
from dotenv import load_dotenv

load_dotenv()

def test_service():
    service = EmailService()
    print(f"Service SMTP Server: {service.smtp_server}")
    print(f"Service Email: {service.email_address}")
    # Don't print password for security, but check if it's set
    print(f"Service Password set: {'Yes' if service.email_password else 'No'}")
    
    # Try sending a test notification
    analysis_data = {
        'filename': 'test_video.mp4',
        'result': 'Fake',
        'confidence': 0.85,
        'trust_score': 0.12,
        'file_type': 'video',
        'id': 123
    }
    user_email = os.environ.get("EMAIL_ADDRESS")
    
    # Try sending a report with attachment
    report_filename = "test_report.pdf"
    report_path = os.path.join("reports", report_filename)
    
    # Create a dummy pdf if it doesn't exist
    if not os.path.exists("reports"):
        os.makedirs("reports")
    with open(report_path, "w") as f:
        f.write("Dummy PDF content")
    
    print(f"Sending report with attachment to {user_email}...")
    success = service.send_report_with_attachment(
        user_email,
        "Test Report with Attachment",
        "<h3>Test Content</h3><p>Attached is a test report.</p>",
        report_path
    )
    
    if success:
        print("SUCCESS! Report with attachment sent.")
    else:
        print(f"FAILED to send report with attachment. Last error: {service.last_error}")

if __name__ == "__main__":
    test_service()
