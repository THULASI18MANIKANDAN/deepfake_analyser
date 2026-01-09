from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import json
import os
import io
import base64

class ReportGenerator:
    """Generate PDF reports for deepfake analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50'),
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#34495e'),
            borderWidth=1,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='HighlightBox',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=10,
            backColor=colors.HexColor('#f8f9fa'),
            borderWidth=1,
            borderColor=colors.HexColor('#dee2e6'),
            borderPadding=10
        ))
    
    def generate_analysis_report(self, analysis_data, output_path):
        """Generate detailed analysis report"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Title
            title = Paragraph("Deepfake Analysis Report", self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Analysis Summary
            story.append(Paragraph("Analysis Summary", self.styles['SectionHeader']))
            
            summary_data = [
                ['File Name', analysis_data['filename']],
                ['File Type', analysis_data['file_type'].title()],
                ['Analysis Date', analysis_data['created_at']],
                ['Result', analysis_data['result']],
                ['Confidence Score', f"{analysis_data['confidence']:.1%}"],
                ['Trust Score', f"{analysis_data['trust_score']:.1%}"],
                ['Accuracy', f"{analysis_data['accuracy']:.1%}"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Analysis Details
            analysis_details = json.loads(analysis_data['analysis_data'])
            
            story.append(Paragraph("Technical Analysis", self.styles['SectionHeader']))
            
            if 'models_used' in analysis_details:
                models_text = f"<b>Models Used:</b> {', '.join(analysis_details['models_used'])}"
                story.append(Paragraph(models_text, self.styles['Normal']))
                story.append(Spacer(1, 10))
            
            if 'cnn_confidence' in analysis_details:
                model_scores = f"""
                <b>Individual Model Scores:</b><br/>
                • CNN Model: {analysis_details['cnn_confidence']:.1%}<br/>
                • XceptionNet: {analysis_details['xception_confidence']:.1%}<br/>
                • Model Agreement: {analysis_details['model_agreement']:.1%}
                """
                story.append(Paragraph(model_scores, self.styles['HighlightBox']))
                story.append(Spacer(1, 15))
            
            # Suspicious Regions
            if analysis_details.get('suspicious_regions'):
                story.append(Paragraph("Suspicious Regions Detected", self.styles['SectionHeader']))
                regions_text = "The following regions showed signs of manipulation:<br/>"
                for region in analysis_details['suspicious_regions']:
                    regions_text += f"• {region.replace('_', ' ').title()}<br/>"
                story.append(Paragraph(regions_text, self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Artifacts and Inconsistencies
            story.append(Paragraph("Detection Findings", self.styles['SectionHeader']))
            findings = []
            
            if analysis_details.get('artifacts_detected'):
                findings.append("Digital artifacts detected in the media")
            if analysis_details.get('temporal_inconsistencies'):
                findings.append("Temporal inconsistencies found in video frames")
            if analysis_details.get('exif_inconsistencies'):
                findings.append("EXIF metadata inconsistencies detected")
            
            if not findings:
                findings.append("No significant artifacts or inconsistencies detected")
            
            findings_text = "<br/>".join([f"• {finding}" for finding in findings])
            story.append(Paragraph(findings_text, self.styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Educational Tips
            if analysis_details.get('tips'):
                story.append(Paragraph("Educational Insights", self.styles['SectionHeader']))
                tips_text = "<br/>".join([f"• {tip}" for tip in analysis_details['tips']])
                story.append(Paragraph(tips_text, self.styles['HighlightBox']))
                story.append(Spacer(1, 15))
            
            # Timeline Analysis (for videos)
            if analysis_data['file_type'] == 'video' and 'timeline' in analysis_details:
                story.append(PageBreak())
                story.append(Paragraph("Video Timeline Analysis", self.styles['SectionHeader']))
                
                # Generate timeline chart
                timeline_chart_path = self._generate_timeline_chart(analysis_details['timeline'], output_path)
                if timeline_chart_path and os.path.exists(timeline_chart_path):
                    story.append(Image(timeline_chart_path, width=6*inch, height=3*inch))
                    story.append(Spacer(1, 15))
            
            # Heatmap Analysis (for images)
            if analysis_data['file_type'] == 'image' and 'heatmap' in analysis_details:
                story.append(Paragraph("Attention Heatmap Analysis", self.styles['SectionHeader']))
                
                heatmap_chart_path = self._generate_heatmap_chart(analysis_details['heatmap'], output_path)
                if heatmap_chart_path and os.path.exists(heatmap_chart_path):
                    story.append(Image(heatmap_chart_path, width=4*inch, height=4*inch))
                    story.append(Spacer(1, 15))
            
            # Disclaimer
            story.append(Spacer(1, 30))
            disclaimer = """
            <b>Disclaimer:</b> This analysis is generated by AI models and should be used as a tool to assist 
            in media verification. Results should be interpreted by qualified professionals and considered 
            alongside other evidence. The accuracy of detection may vary based on the quality and type of content analyzed.
            """
            story.append(Paragraph(disclaimer, self.styles['HighlightBox']))
            
            # Footer
            story.append(Spacer(1, 20))
            footer = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Deepfake Analyzer"
            story.append(Paragraph(footer, self.styles['Normal']))
            
            # Build PDF
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"Error generating analysis report: {e}")
            return False
    
    def generate_weekly_report(self, user_data, analyses_data, output_path):
        """Generate weekly summary report"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Title
            week_ending = (datetime.now() - timedelta(days=datetime.now().weekday())).strftime('%Y-%m-%d')
            title = Paragraph(f"Weekly Analysis Report - Week Ending {week_ending}", self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
            
            total_analyses = len(analyses_data)
            fake_count = sum(1 for a in analyses_data if a['result'] == 'Fake')
            real_count = total_analyses - fake_count
            avg_confidence = np.mean([a['confidence'] for a in analyses_data]) if analyses_data else 0
            
            summary_stats = [
                ['Total Analyses', str(total_analyses)],
                ['Fake Content Detected', str(fake_count)],
                ['Authentic Content', str(real_count)],
                ['Average Confidence', f"{avg_confidence:.1%}"],
                ['Detection Rate', f"{fake_count/total_analyses:.1%}" if total_analyses > 0 else "0%"]
            ]
            
            summary_table = Table(summary_stats, colWidths=[2.5*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
            
            # Daily Analysis Trend
            if analyses_data:
                story.append(Paragraph("Daily Analysis Trend", self.styles['SectionHeader']))
                
                # Generate daily trend chart
                trend_chart_path = self._generate_daily_trend_chart(analyses_data, output_path)
                if trend_chart_path and os.path.exists(trend_chart_path):
                    story.append(Image(trend_chart_path, width=6*inch, height=3*inch))
                    story.append(Spacer(1, 15))
            
            # File Type Distribution
            if analyses_data:
                story.append(Paragraph("File Type Analysis", self.styles['SectionHeader']))
                
                file_types = {}
                for analysis in analyses_data:
                    file_type = analysis['file_type']
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                
                file_type_data = [[file_type.title(), str(count)] for file_type, count in file_types.items()]
                
                file_type_table = Table(file_type_data, colWidths=[2*inch, 1*inch])
                file_type_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(file_type_table)
                story.append(Spacer(1, 20))
            
            # Recommendations
            story.append(Paragraph("Recommendations", self.styles['SectionHeader']))
            
            recommendations = []
            if fake_count > real_count:
                recommendations.append("High fake content detection rate - consider reviewing sources")
            if avg_confidence < 0.7:
                recommendations.append("Lower average confidence - consider higher quality source material")
            if total_analyses > 50:
                recommendations.append("High analysis volume - consider batch processing for efficiency")
            
            if not recommendations:
                recommendations.append("Analysis patterns appear normal - continue current practices")
            
            rec_text = "<br/>".join([f"• {rec}" for rec in recommendations])
            story.append(Paragraph(rec_text, self.styles['HighlightBox']))
            
            # Build PDF
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"Error generating weekly report: {e}")
            return False
    
    def _generate_timeline_chart(self, timeline_data, base_path):
        """Generate timeline chart for video analysis"""
        try:
            plt.figure(figsize=(10, 5))
            
            times = [point['time'] for point in timeline_data]
            confidences = [point['ensemble_confidence'] for point in timeline_data]
            
            plt.plot(times, confidences, 'b-', linewidth=2, label='Confidence Score')
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Detection Threshold')
            
            plt.xlabel('Time (seconds)')
            plt.ylabel('Confidence Score')
            plt.title('Deepfake Detection Confidence Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            chart_path = base_path.replace('.pdf', '_timeline.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            print(f"Error generating timeline chart: {e}")
            return None
    
    def _generate_heatmap_chart(self, heatmap_data, base_path):
        """Generate heatmap visualization"""
        try:
            plt.figure(figsize=(8, 8))
            
            heatmap_array = np.array(heatmap_data)
            
            plt.imshow(heatmap_array, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Attention Intensity')
            plt.title('Model Attention Heatmap')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            
            chart_path = base_path.replace('.pdf', '_heatmap.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            print(f"Error generating heatmap chart: {e}")
            return None
    
    def _generate_daily_trend_chart(self, analyses_data, base_path):
        """Generate daily analysis trend chart"""
        try:
            plt.figure(figsize=(10, 5))
            
            # Group analyses by day
            daily_counts = {}
            daily_fake_counts = {}
            
            for analysis in analyses_data:
                date = analysis['created_at'][:10]  # Extract date part
                daily_counts[date] = daily_counts.get(date, 0) + 1
                if analysis['result'] == 'Fake':
                    daily_fake_counts[date] = daily_fake_counts.get(date, 0) + 1
            
            dates = sorted(daily_counts.keys())
            total_counts = [daily_counts[date] for date in dates]
            fake_counts = [daily_fake_counts.get(date, 0) for date in dates]
            
            plt.bar(dates, total_counts, alpha=0.7, label='Total Analyses', color='skyblue')
            plt.bar(dates, fake_counts, alpha=0.9, label='Fake Detected', color='salmon')
            
            plt.xlabel('Date')
            plt.ylabel('Number of Analyses')
            plt.title('Daily Analysis Activity')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            chart_path = base_path.replace('.pdf', '_daily_trend.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            print(f"Error generating daily trend chart: {e}")
            return None
