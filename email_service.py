import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime
import json

class EmailService:
    """Email service for sending notifications and reports"""
    
    def __init__(self):
        # Email configuration (in production, use environment variables)
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_address = "thulasitmk181@gmail.com"
        self.email_password = "tbex lpqr rorf zvxc"  # Use app password for Gmail
        
    def send_analysis_complete_notification(self, user_email, analysis_data):
        """Send notification when analysis is complete"""
        try:
            subject = f"Deepfake Analysis Complete - {analysis_data['filename']}"
            
            # Create HTML email content
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                        Analysis Complete
                    </h2>
                    
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                        <h3 style="margin-top: 0; color: #495057;">File: {analysis_data['filename']}</h3>
                        <p><strong>Result:</strong> 
                            <span style="color: {'#dc3545' if analysis_data['result'] == 'Fake' else '#28a745'}; font-weight: bold;">
                                {analysis_data['result']}
                            </span>
                        </p>
                        <p><strong>Confidence:</strong> {analysis_data['confidence']:.1%}</p>
                        <p><strong>Trust Score:</strong> {analysis_data['trust_score']:.1%}</p>
                        <p><strong>File Type:</strong> {analysis_data['file_type'].title()}</p>
                    </div>
                    
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0;">
                        <h4 style="margin-top: 0; color: #1976d2;">Key Findings:</h4>
                        <ul>
                            {"<li>High confidence detection - result is reliable</li>" if analysis_data['confidence'] > 0.8 else ""}
                            {"<li>Multiple suspicious regions detected</li>" if analysis_data.get('suspicious_regions') else ""}
                            {"<li>Temporal inconsistencies found</li>" if analysis_data.get('temporal_inconsistencies') else ""}
                            {"<li>Artifacts detected in the media</li>" if analysis_data.get('artifacts_detected') else ""}
                        </ul>
                    </div>
                    
                    <p style="margin-top: 30px;">
                        <a href="https://deepfake-analyzer.com/analysis/{analysis_data['id']}" 
                           style="background: #007bff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">
                            View Detailed Results
                        </a>
                    </p>
                    
                    <hr style="margin: 30px 0; border: none; border-top: 1px solid #dee2e6;">
                    <p style="font-size: 12px; color: #6c757d; text-align: center;">
                        This is an automated message from Deepfake Analyzer.<br>
                        Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </div>
            </body>
            </html>
            """
            
            return self._send_email(user_email, subject, html_content)
            
        except Exception as e:
            print(f"Error sending analysis notification: {e}")
            return False
    
    def send_weekly_report(self, user_email, report_data):
        """Send weekly analysis report"""
        try:
            subject = f"Weekly Deepfake Analysis Report - {report_data['week_ending']}"
            
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                        Weekly Analysis Report
                    </h2>
                    <p style="color: #6c757d;">Week ending {report_data['week_ending']}</p>
                    
                    <div style="display: flex; gap: 20px; margin: 20px 0;">
                        <div style="flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;">
                            <h3 style="margin: 0; color: #007bff; font-size: 2em;">{report_data['total_analyses']}</h3>
                            <p style="margin: 5px 0 0 0; color: #6c757d;">Total Analyses</p>
                        </div>
                        <div style="flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;">
                            <h3 style="margin: 0; color: #dc3545; font-size: 2em;">{report_data['fake_count']}</h3>
                            <p style="margin: 5px 0 0 0; color: #6c757d;">Fake Detected</p>
                        </div>
                        <div style="flex: 1; background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center;">
                            <h3 style="margin: 0; color: #28a745; font-size: 2em;">{report_data['real_count']}</h3>
                            <p style="margin: 5px 0 0 0; color: #6c757d;">Real Content</p>
                        </div>
                    </div>
                    
                    <div style="background: #e8f5e8; padding: 20px; border-radius: 8px; margin: 20px 0;">
                        <h4 style="margin-top: 0; color: #155724;">Weekly Insights</h4>
                        <ul>
                            <li>Average confidence score: {report_data['avg_confidence']:.1%}</li>
                            <li>Most analyzed file type: {report_data['top_file_type']}</li>
                            <li>Peak analysis day: {report_data['peak_day']}</li>
                            <li>Detection accuracy: {report_data['accuracy']:.1%}</li>
                        </ul>
                    </div>
                    
                    <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0;">
                        <h4 style="margin-top: 0; color: #856404;">Recommendations</h4>
                        <p>{report_data['recommendations']}</p>
                    </div>
                    
                    <p style="margin-top: 30px;">
                        <a href="https://deepfake-analyzer.com/analytics" 
                           style="background: #007bff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">
                            View Full Analytics
                        </a>
                    </p>
                    
                    <hr style="margin: 30px 0; border: none; border-top: 1px solid #dee2e6;">
                    <p style="font-size: 12px; color: #6c757d; text-align: center;">
                        Deepfake Analyzer Weekly Report<br>
                        Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </div>
            </body>
            </html>
            """
            
            return self._send_email(user_email, subject, html_content)
            
        except Exception as e:
            print(f"Error sending weekly report: {e}")
            return False
    
    def send_admin_alert(self, alert_type, alert_data):
        """Send admin alerts for system events"""
        try:
            admin_email = "admin@deepfake-analyzer.com"
            
            if alert_type == "high_fake_detection":
                subject = "Alert: High Fake Content Detection Rate"
                content = f"""
                <h3>High Fake Detection Alert</h3>
                <p>The system has detected an unusually high rate of fake content:</p>
                <ul>
                    <li>Detection rate: {alert_data['fake_rate']:.1%}</li>
                    <li>Time period: {alert_data['time_period']}</li>
                    <li>Total analyses: {alert_data['total_analyses']}</li>
                </ul>
                <p>This may indicate a coordinated disinformation campaign or system anomaly.</p>
                """
            elif alert_type == "system_overload":
                subject = "Alert: System Overload Detected"
                content = f"""
                <h3>System Overload Alert</h3>
                <p>The system is experiencing high load:</p>
                <ul>
                    <li>Requests per hour: {alert_data['requests_per_hour']}</li>
                    <li>Queue length: {alert_data['queue_length']}</li>
                    <li>Average response time: {alert_data['avg_response_time']:.2f}s</li>
                </ul>
                <p>Consider scaling resources or implementing rate limiting.</p>
                """
            
            elif alert_type == "new_user_spike":
                subject = "Alert: Unusual User Registration Activity"
                content = f"""
                <h3>User Registration Spike Alert</h3>
                <p>Unusual user registration activity detected:</p>
                <ul>
                    <li>New registrations: {alert_data['new_users']}</li>
                    <li>Time period: {alert_data['time_period']}</li>
                    <li>Increase from average: {alert_data['increase_percentage']:.1%}</li>
                </ul>
                <p>Monitor for potential abuse or bot activity.</p>
                """
            
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <div style="background: #dc3545; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <h2 style="margin: 0;">🚨 System Alert</h2>
                    </div>
                    {content}
                    <hr style="margin: 30px 0;">
                    <p style="font-size: 12px; color: #6c757d;">
                        Automated alert from Deepfake Analyzer<br>
                        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </div>
            </body>
            </html>
            """
            return self._send_email(admin_email, subject, html_content)
            
        except Exception as e:
            print(f"Error sending admin alert: {e}")
            return False
    
    def _send_email(self, to_email, subject, html_content):
        """Send email using SMTP"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_address
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Connect to server and send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_address, self.email_password)
            
            text = msg.as_string()
            server.sendmail(self.email_address, to_email, text)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def send_report_with_attachment(self, to_email, subject, content, attachment_path):
        """Send email with PDF attachment"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_address
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add content
            msg.attach(MIMEText(content, 'html'))
            
            # Add attachment
            if os.path.exists(attachment_path):
                with open(attachment_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(attachment_path)}'
                )
                msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_address, self.email_password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Error sending email with attachment: {e}")
            return False
