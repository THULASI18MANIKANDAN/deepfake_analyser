# Deepfake Analyzer

A comprehensive web-based application designed to detect and analyze deepfake content in images and videos using advanced machine learning techniques.

## 🚀 Features

- **Deepfake Detection**: Upload images or videos to analyze them for deepfake manipulation.
- **Detailed Analysis**: Get confidence scores, accuracy ratings, and "trust scores" for analyzed content.
- **User Dashboard**: personalized dashboard to track your recent analyses and statistics.
- **Analytics**: Visual insights into analysis trends, fake vs. real ratios, and confidence distributions over time.
- **History & Bookmarks**: Searchable history of past analyses and the ability to bookmark important results.
- **Report Generation**: Generate detailed reports for your analyses.
- **Admin Panel**: Administrative dashboard for user management and system-wide statistics.
- **Secure Authentication**: User registration and login system with role-based access control.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, OpenCV
- **Database**: SQLite
- **Image Processing**: Pillow
- **Data Visualization**: Matplotlib
- **Frontend**: HTML5, CSS3, JavaScript (Jinja2 Templates)

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🔧 Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd deepfake-analyzer
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

1.  **Start the application**
    ```bash
    python app.py
    ```

2.  **Access the application**
    Open your web browser and go to:
    `http://localhost:5000`

3.  **Default Admin Credentials**
    *   **Email**: `admin@deepfake-analyzer.com`
    *   **Password**: `admin123`

    *Note: It is highly recommended to change the admin password after the first login.*

## 📂 Project Structure

- `app.py`: Main Flask application entry point and route definitions.
- `analysis_engine.py`: Core logic for deepfake detection and media processing.
- `report_generator.py`: Generates PDF/detailed reports of analyses.
- `email_service.py`: Handles email notifications (if configured).
- `templates/`: HTML templates for the user interface.
- `static/`: Static assets (CSS, JS, images).
- `uploads/`: Directory where uploaded files are temporarily stored.
- `deepfake_analyzer.db`: SQLite database storing user data and analysis results.

