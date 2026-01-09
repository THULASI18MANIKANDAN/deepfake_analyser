from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file, g
import sqlite3
import hashlib
import os
import uuid
from datetime import datetime, timedelta
import json
from functools import wraps
import secrets
from analysis_engine import DeepfakeAnalyzer
from werkzeug.utils import secure_filename
import mimetypes
import csv
import io
import time

from email_service import EmailService
from report_generator import ReportGenerator

import struct

def clean_trust_scores(rows):
    clean = []
    for row in rows:
        row_dict = dict(row)
        ts = row_dict.get('trust_score')
        if isinstance(ts, (bytes, bytearray)):
            try:
                row_dict['trust_score'] = struct.unpack('d', ts[:8])[0]
            except Exception:
                try:
                    row_dict['trust_score'] = float(ts.decode(errors='ignore'))
                except Exception:
                    row_dict['trust_score'] = 0.0
        clean.append(row_dict)
    return clean

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

email_service = EmailService()
report_generator = ReportGenerator()

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect('deepfake_analyzer.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with required tables"""
    conn = get_db_connection()
    
    # Users table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Analysis results table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            accuracy REAL NOT NULL,
            result TEXT NOT NULL,
            trust_score REAL NOT NULL,
            analysis_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # API keys table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            api_key TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Bookmarks table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            analysis_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (analysis_id) REFERENCES analyses (id)
        )
    ''')
    
    # API requests table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS api_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            api_key_id INTEGER,
            endpoint TEXT NOT NULL,
            method TEXT NOT NULL,
            status_code INTEGER NOT NULL,
            response_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (api_key_id) REFERENCES api_keys (id)
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            report_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            analysis_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (analysis_id) REFERENCES analyses (id)
        )
    ''')
    
    # Create default admin user if none exists
    admin_exists = conn.execute('SELECT id FROM users WHERE is_admin = 1').fetchone()
    if not admin_exists:
        admin_password = hash_password('admin123')
        conn.execute('''
            INSERT INTO users (email, password_hash, is_admin) 
            VALUES (?, ?, ?)
        ''', ('admin@deepfake-analyzer.com', admin_password, True))
        conn.commit()
    
    conn.close()

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to require admin access"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        user = conn.execute('SELECT is_admin FROM users WHERE id = ?', (session['user_id'],)).fetchone()
        conn.close()
        
        if not user or not user['is_admin']:
            flash('Admin access required', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    """Check if file is allowed"""
    return analyzer.is_supported_format(filename)

# Initialize analyzer
analyzer = DeepfakeAnalyzer()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        if not email or not password:
            flash('Email and password are required', 'error')
            return render_template('register.html')
        
        conn = get_db_connection()
        
        # Check if user already exists
        existing_user = conn.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
        if existing_user:
            flash('Email already registered', 'error')
            conn.close()
            return render_template('register.html')
        
        # Create new user
        password_hash = hash_password(password)
        conn.execute('INSERT INTO users (email, password_hash) VALUES (?, ?)', 
                    (email, password_hash))
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT id, password_hash FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        
        if user and user['password_hash'] == hash_password(password):
            session['user_id'] = user['id']
            session['user_email'] = email
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    conn = get_db_connection()
    
    # Get recent analyses
    analyses = conn.execute('''
        SELECT * FROM analyses 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 10
    ''', (session['user_id'],)).fetchall()
    
    # Get statistics
    stats = conn.execute('''
        SELECT 
            COUNT(*) as total_analyses,
            AVG(confidence) as avg_confidence,
            SUM(CASE WHEN result = 'Fake' THEN 1 ELSE 0 END) as fake_count,
            SUM(CASE WHEN result = 'Real' THEN 1 ELSE 0 END) as real_count
        FROM analyses 
        WHERE user_id = ?
    ''', (session['user_id'],)).fetchone()
    
    # Get weekly analysis trend
    weekly_stats = conn.execute('''
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as count,
            SUM(CASE WHEN result = 'Fake' THEN 1 ELSE 0 END) as fake_count
        FROM analyses 
        WHERE user_id = ? AND created_at >= date('now', '-7 days')
        GROUP BY DATE(created_at)
        ORDER BY date
    ''', (session['user_id'],)).fetchall()
    
    conn.close()
    
    # --- 🔧 Fix trust_score conversion ---
    import struct
    clean_analyses = []
    for analysis in analyses:
        analysis_dict = dict(analysis)
        ts = analysis_dict['trust_score']
        if isinstance(ts, (bytes, bytearray)):
            try:
                analysis_dict['trust_score'] = struct.unpack('d', ts[:8])[0]
            except Exception:
                try:
                    analysis_dict['trust_score'] = float(ts.decode(errors='ignore'))
                except Exception:
                    analysis_dict['trust_score'] = 0.0
        clean_analyses.append(analysis_dict)
    
    return render_template(
        'dashboard.html',
        analyses=clean_analyses,
        stats=stats,
        weekly_stats=weekly_stats
    )


@app.route('/history')
@login_required
def history():
    """Analysis history with filtering and search"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Get filter parameters
    result_filter = request.args.get('result', '')
    file_type_filter = request.args.get('file_type', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    search_query = request.args.get('search', '')
    sort_by = request.args.get('sort', 'created_at')
    sort_order = request.args.get('order', 'desc')
    
    # Build SQL query
    where_conditions = ['user_id = ?']
    params = [session['user_id']]
    
    if result_filter:
        where_conditions.append('result = ?')
        params.append(result_filter)
    
    if file_type_filter:
        where_conditions.append('file_type = ?')
        params.append(file_type_filter)
    
    if date_from:
        where_conditions.append('DATE(created_at) >= ?')
        params.append(date_from)
    
    if date_to:
        where_conditions.append('DATE(created_at) <= ?')
        params.append(date_to)
    
    if search_query:
        where_conditions.append('filename LIKE ?')
        params.append(f'%{search_query}%')
    
    where_clause = ' AND '.join(where_conditions)
    
    # Validate sort parameters
    valid_sorts = ['created_at', 'filename', 'confidence', 'result']
    if sort_by not in valid_sorts:
        sort_by = 'created_at'
    
    if sort_order not in ['asc', 'desc']:
        sort_order = 'desc'
    
    conn = get_db_connection()
    
    # Get total count
    count_query = f'SELECT COUNT(*) as total FROM analyses WHERE {where_clause}'
    total = conn.execute(count_query, params).fetchone()['total']
    
    # Get analyses with pagination
    offset = (page - 1) * per_page
    query = f'''
        SELECT * FROM analyses 
        WHERE {where_clause}
        ORDER BY {sort_by} {sort_order.upper()}
        LIMIT ? OFFSET ?
    '''
    params.extend([per_page, offset])
    analyses = conn.execute(query, params).fetchall()
    
    conn.close()
    
    # Calculate pagination info
    total_pages = (total + per_page - 1) // per_page
    has_prev = page > 1
    has_next = page < total_pages
    
    analyses = clean_trust_scores(analyses)
    return render_template('history.html', 
                         analyses=analyses,
                         total=total,
                         page=page,
                         total_pages=total_pages,
                         has_prev=has_prev,
                         has_next=has_next,
                         result_filter=result_filter,
                         file_type_filter=file_type_filter,
                         date_from=date_from,
                         date_to=date_to,
                         search_query=search_query,
                         sort_by=sort_by,
                         sort_order=sort_order)

@app.route('/bookmarks')
@login_required
def bookmarks():
    """User bookmarks"""
    conn = get_db_connection()
    
    bookmarked_analyses = conn.execute('''
        SELECT a.*, b.created_at as bookmarked_at
        FROM analyses a
        JOIN bookmarks b ON a.id = b.analysis_id
        WHERE b.user_id = ?
        ORDER BY b.created_at DESC
    ''', (session['user_id'],)).fetchall()
    
    conn.close()
    
    bookmarked_analyses = clean_trust_scores(bookmarked_analyses)
    return render_template('bookmarks.html', analyses=bookmarked_analyses)
@app.route('/analytics')
@login_required
def analytics():
    """User analytics and insights"""
    conn = get_db_connection()
    
    # Monthly analysis trend (last 12 months)
    monthly_data = conn.execute('''
        SELECT 
            strftime('%Y-%m', created_at) as month,
            COUNT(*) as total,
            SUM(CASE WHEN result = 'Fake' THEN 1 ELSE 0 END) as fake_count,
            SUM(CASE WHEN result = 'Real' THEN 1 ELSE 0 END) as real_count,
            AVG(confidence) as avg_confidence
        FROM analyses 
        WHERE user_id = ? AND created_at >= date('now', '-12 months')
        GROUP BY strftime('%Y-%m', created_at)
        ORDER BY month
    ''', (session['user_id'],)).fetchall()
    # Weekly analysis trend (last 12 weeks)
    weekly_data = conn.execute('''
        SELECT 
            strftime('%W', created_at) as week,
            COUNT(*) as total,
            SUM(CASE WHEN result = 'Fake' THEN 1 ELSE 0 END) as fake,
            SUM(CASE WHEN result = 'Real' THEN 1 ELSE 0 END) as real
        FROM analyses
        WHERE user_id = ? AND created_at >= date('now', '-12 weeks')
        GROUP BY strftime('%W', created_at)
        ORDER BY week
    ''', (session['user_id'],)).fetchall()

# Yearly analysis trend (last 3 years)
    yearly_data = conn.execute('''
        SELECT 
            strftime('%Y', created_at) as year,
            COUNT(*) as total,
            SUM(CASE WHEN result = 'Fake' THEN 1 ELSE 0 END) as fake,
            SUM(CASE WHEN result = 'Real' THEN 1 ELSE 0 END) as real
        FROM analyses
        WHERE user_id = ? AND created_at >= date('now', '-3 years')
        GROUP BY strftime('%Y', created_at)
        ORDER BY year
    ''', (session['user_id'],)).fetchall()
    
    # File type distribution
    file_type_data = conn.execute('''
        SELECT 
            file_type,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence
        FROM analyses 
        WHERE user_id = ?
        GROUP BY file_type
    ''', (session['user_id'],)).fetchall()
    
    # Confidence distribution
    confidence_ranges = conn.execute('''
        SELECT 
            CASE 
                WHEN confidence < 0.3 THEN 'Low (0-30%)'
                WHEN confidence < 0.7 THEN 'Medium (30-70%)'
                ELSE 'High (70-100%)'
            END as confidence_range,
            COUNT(*) as count
        FROM analyses 
        WHERE user_id = ?
        GROUP BY confidence_range
    ''', (session['user_id'],)).fetchall()
    
    # Recent activity (last 30 days)
    recent_activity = conn.execute('''
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as analyses_count,
            SUM(CASE WHEN result = 'Fake' THEN 1 ELSE 0 END) as fake_count
        FROM analyses 
        WHERE user_id = ? AND created_at >= date('now', '-30 days')
        GROUP BY DATE(created_at)
        ORDER BY date DESC
    ''', (session['user_id'],)).fetchall()

    # Overall stats
    stats = conn.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN result = 'Fake' THEN 1 ELSE 0 END) as fake,
            SUM(CASE WHEN result = 'Real' THEN 1 ELSE 0 END) as real,
            AVG(confidence) as avg_confidence
        FROM analyses 
        WHERE user_id = ?
    ''', (session['user_id'],)).fetchone()
    
    conn.close()

    # Convert sqlite3.Row -> dict
    monthly_data = [dict(row) for row in monthly_data]
    # Convert all rows → dicts
    weekly_data = [dict(row) for row in weekly_data]
    yearly_data = [dict(row) for row in yearly_data]
    file_type_data = [dict(row) for row in file_type_data]
    confidence_ranges = [dict(row) for row in confidence_ranges]
    recent_activity = [dict(row) for row in recent_activity]
    stats = dict(stats) if stats else {}

    return render_template(
    'analytics.html',
    stats=stats,
    monthly_data=monthly_data,
    weekly_data=weekly_data,   # ✅ now passed
    yearly_data=yearly_data,   # ✅ now passed
    file_type_data=file_type_data,
    confidence_ranges=confidence_ranges,
    recent_activity=recent_activity
)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """File upload and analysis"""
    if request.method == 'POST':
        if 'files' not in request.files:
            flash('No files selected', 'error')
            return redirect(request.url)
        
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            if file.filename == '':
                continue
                
            if file and allowed_file(file.filename):
                # Secure filename and save
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                
                # Analyze file
                file_type = analyzer.get_file_type(filename)
                
                if file_type == 'image':
                    analysis_result = analyzer.analyze_image(file_path, filename)
                elif file_type == 'video':
                    analysis_result = analyzer.analyze_video(file_path, filename)
                else:
                    flash(f'Unsupported file type: {filename}', 'error')
                    continue
                
                # Save to database
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO analyses (user_id, filename, file_path, file_type, confidence, accuracy, result, trust_score, analysis_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                    session['user_id'],
                    filename,
                    file_path,
                    file_type,
                    float(analysis_result['confidence']),
                    float(analysis_result['accuracy']),
                    analysis_result['result'],
                    float(analysis_result['trust_score']),
                    analysis_result['analysis_data']
                ))
                conn.commit()
                analysis_id = cursor.lastrowid
                cursor.close()
                conn.close()
                
                results.append({
                    'id': analysis_id,
                    'filename': filename,
                    'result': analysis_result['result'],
                    'confidence': float(analysis_result['confidence']),
                    'trust_score': float(analysis_result['trust_score'])
                })
            else:
                flash(f'Invalid file type: {filename}', 'error')
        
        if results:
            flash(f'Successfully analyzed {len(results)} files', 'success')
            return render_template('upload_results.html', results=results)
        else:
            flash('No valid files were processed', 'error')
    
    return render_template('upload.html')
@app.route('/analysis/<int:analysis_id>')
@login_required
def view_analysis(analysis_id):
    """View detailed analysis results"""
    conn = get_db_connection()
    
    # Fetch analysis
    analysis = conn.execute('''
        SELECT * FROM analyses 
        WHERE id = ? AND user_id = ?
    ''', (analysis_id, session['user_id'])).fetchone()
    
    # Check if bookmarked
    is_bookmarked = conn.execute('''
        SELECT id FROM bookmarks 
        WHERE user_id = ? AND analysis_id = ?
    ''', (session['user_id'], analysis_id)).fetchone() is not None
    
    conn.close()
    
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard'))
    
    # Convert Row object to dict to allow modifications
    analysis = dict(analysis)
    
    # Function to safely convert bytes/strings to float
    def safe_float(value, default=0.0):
        if isinstance(value, bytes):
            try:
                return float(value.decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    # Automatically convert all numeric fields from bytes to float
    numeric_fields = ['accuracy', 'trust_score']  # Add more fields if needed
    for field in numeric_fields:
        if field in analysis:
            analysis[field] = safe_float(analysis[field])
    
    # Parse analysis_data safely
    try:
        analysis_data = json.loads(analysis.get('analysis_data', '{}'))
    except (json.JSONDecodeError, TypeError):
        analysis_data = {}
    
    return render_template(
        'analysis_detail.html', 
        analysis=analysis, 
        analysis_data=analysis_data,
        is_bookmarked=is_bookmarked
    )

@app.route('/bookmark/<int:analysis_id>', methods=['POST'])
@login_required
def bookmark_analysis(analysis_id):
    """Bookmark an analysis"""
    conn = get_db_connection()
    
    # Check if already bookmarked
    existing = conn.execute('''
        SELECT id FROM bookmarks 
        WHERE user_id = ? AND analysis_id = ?
    ''', (session['user_id'], analysis_id)).fetchone()
    
    if existing:
        # Remove bookmark
        conn.execute('DELETE FROM bookmarks WHERE user_id = ? AND analysis_id = ?', 
                    (session['user_id'], analysis_id))
        message = 'Bookmark removed'
        bookmarked = False
    else:
        # Add bookmark
        conn.execute('INSERT INTO bookmarks (user_id, analysis_id) VALUES (?, ?)', 
                    (session['user_id'], analysis_id))
        message = 'Analysis bookmarked'
        bookmarked = True
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': message, 'bookmarked': bookmarked})

@app.route('/bulk_action', methods=['POST'])
@login_required
def bulk_action():
    """Perform bulk actions on analyses"""
    action = request.json.get('action')
    analysis_ids = request.json.get('analysis_ids', [])
    
    if not analysis_ids:
        return jsonify({'success': False, 'message': 'No analyses selected'})
    
    conn = get_db_connection()
    
    if action == 'bookmark':
        # Bookmark selected analyses
        for analysis_id in analysis_ids:
            # Check if already bookmarked
            existing = conn.execute('''
                SELECT id FROM bookmarks 
                WHERE user_id = ? AND analysis_id = ?
            ''', (session['user_id'], analysis_id)).fetchone()
            
            if not existing:
                conn.execute('INSERT INTO bookmarks (user_id, analysis_id) VALUES (?, ?)', 
                           (session['user_id'], analysis_id))
        
        conn.commit()
        message = f'Bookmarked {len(analysis_ids)} analyses'
    
    elif action == 'delete':
        # Delete selected analyses (and their files)
        for analysis_id in analysis_ids:
            analysis = conn.execute('''
                SELECT file_path FROM analyses 
                WHERE id = ? AND user_id = ?
            ''', (analysis_id, session['user_id'])).fetchone()
            
            if analysis:
                # Delete file
                try:
                    os.remove(analysis['file_path'])
                except OSError:
                    pass
                
                # Delete from database
                conn.execute('DELETE FROM bookmarks WHERE analysis_id = ?', (analysis_id,))
                conn.execute('DELETE FROM analyses WHERE id = ? AND user_id = ?', 
                           (analysis_id, session['user_id']))
        
        conn.commit()
        message = f'Deleted {len(analysis_ids)} analyses'
    
    else:
        conn.close()
        return jsonify({'success': False, 'message': 'Invalid action'})
    
    conn.close()
    return jsonify({'success': True, 'message': message})

@app.route('/admin')
@admin_required
def admin_dashboard():
    """Admin dashboard with system overview"""
    conn = get_db_connection()
    
    # System statistics
    system_stats = conn.execute('''
        SELECT 
            (SELECT COUNT(*) FROM users) as total_users,
            (SELECT COUNT(*) FROM users WHERE is_admin = 1) as admin_users,
            (SELECT COUNT(*) FROM analyses) as total_analyses,
            (SELECT COUNT(*) FROM analyses WHERE created_at >= date('now', '-7 days')) as analyses_this_week,
            (SELECT COUNT(*) FROM analyses WHERE result = 'Fake') as total_fake,
            (SELECT COUNT(*) FROM analyses WHERE result = 'Real') as total_real,
            (SELECT AVG(confidence) FROM analyses) as avg_confidence
    ''').fetchone()
    
    # Recent user registrations
    recent_users = conn.execute('''
        SELECT email, created_at, is_admin
        FROM users 
        ORDER BY created_at DESC 
        LIMIT 10
    ''').fetchall()
    
    # Daily analysis trend (last 30 days)
    daily_trend = conn.execute('''
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as total,
            SUM(CASE WHEN result = 'Fake' THEN 1 ELSE 0 END) as fake_count,
            COUNT(DISTINCT user_id) as unique_users
        FROM analyses 
        WHERE created_at >= date('now', '-30 days')
        GROUP BY DATE(created_at)
        ORDER BY date
    ''').fetchall()
    
    # Top users by analysis count
    top_users = conn.execute('''
        SELECT 
            u.email,
            COUNT(a.id) as analysis_count,
            SUM(CASE WHEN a.result = 'Fake' THEN 1 ELSE 0 END) as fake_count,
            AVG(a.confidence) as avg_confidence
        FROM users u
        LEFT JOIN analyses a ON u.id = a.user_id
        WHERE u.is_admin = 0
        GROUP BY u.id, u.email
        HAVING analysis_count > 0
        ORDER BY analysis_count DESC
        LIMIT 10
    ''').fetchall()
    
    conn.close()
    
    return render_template('admin/dashboard.html',
                         system_stats=system_stats,
                         recent_users=recent_users,
                         daily_trend=daily_trend,
                         top_users=top_users)

@app.route('/admin/users')
@admin_required
def admin_users():
    """Admin user management"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    search_query = request.args.get('search', '')
    
    conn = get_db_connection()
    
    # Build query with search
    where_clause = "1=1"
    params = []
    
    if search_query:
        where_clause += " AND email LIKE ?"
        params.append(f'%{search_query}%')
    
    # Get total count
    total = conn.execute(f'SELECT COUNT(*) as total FROM users WHERE {where_clause}', params).fetchone()['total']
    
    # Get users with pagination
    offset = (page - 1) * per_page
    users = conn.execute(f'''
        SELECT 
            u.*,
            COUNT(a.id) as analysis_count,
            MAX(a.created_at) as last_analysis
        FROM users u
        LEFT JOIN analyses a ON u.id = a.user_id
        WHERE {where_clause}
        GROUP BY u.id
        ORDER BY u.created_at DESC
        LIMIT ? OFFSET ?
    ''', params + [per_page, offset]).fetchall()
    
    conn.close()
    
    # Calculate pagination info
    total_pages = (total + per_page - 1) // per_page
    has_prev = page > 1
    has_next = page < total_pages
    
    return render_template('admin/users.html',
                         users=users,
                         total=total,
                         page=page,
                         total_pages=total_pages,
                         has_prev=has_prev,
                         has_next=has_next,
                         search_query=search_query)

@app.route('/admin/analyses')
@admin_required
def admin_analyses():
    """Admin analysis management"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Filters
    result_filter = request.args.get('result', '')
    user_filter = request.args.get('user', '')
    date_from = request.args.get('date_from', '')
    date_to = request.args.get('date_to', '')
    
    conn = get_db_connection()
    
    # Build query
    where_conditions = ['1=1']
    params = []
    
    if result_filter:
        where_conditions.append('a.result = ?')
        params.append(result_filter)
    
    if user_filter:
        where_conditions.append('u.email LIKE ?')
        params.append(f'%{user_filter}%')
    
    if date_from:
        where_conditions.append('DATE(a.created_at) >= ?')
        params.append(date_from)
    
    if date_to:
        where_conditions.append('DATE(a.created_at) <= ?')
        params.append(date_to)
    
    where_clause = ' AND '.join(where_conditions)
    
    # Get total count
    total = conn.execute(f'''
        SELECT COUNT(*) as total 
        FROM analyses a
        JOIN users u ON a.user_id = u.id
        WHERE {where_clause}
    ''', params).fetchone()['total']
    
    # Get analyses with pagination
    offset = (page - 1) * per_page
    analyses = conn.execute(f'''
        SELECT 
            a.*,
            u.email as user_email
        FROM analyses a
        JOIN users u ON a.user_id = u.id
        WHERE {where_clause}
        ORDER BY a.created_at DESC
        LIMIT ? OFFSET ?
    ''', params + [per_page, offset]).fetchall()
    
    conn.close()
    
    # Calculate pagination info
    total_pages = (total + per_page - 1) // per_page
    has_prev = page > 1
    has_next = page < total_pages
    
    analyses = clean_trust_scores(analyses)
    return render_template('admin/analyses.html',
                         analyses=analyses,
                         total=total,
                         page=page,
                         total_pages=total_pages,
                         has_prev=has_prev,
                         has_next=has_next,
                         result_filter=result_filter,
                         user_filter=user_filter,
                         date_from=date_from,
                         date_to=date_to)

@app.route('/admin/analytics')
@admin_required
def admin_analytics():
    """Admin system analytics"""
    conn = get_db_connection()
    
    # Monthly statistics (last 12 months)
    monthly_stats = conn.execute('''
        SELECT 
            strftime('%Y-%m', created_at) as month,
            COUNT(*) as total_analyses,
            COUNT(DISTINCT user_id) as unique_users,
            SUM(CASE WHEN result = 'Fake' THEN 1 ELSE 0 END) as fake_count,
            AVG(confidence) as avg_confidence
        FROM analyses 
        WHERE created_at >= date('now', '-12 months')
        GROUP BY strftime('%Y-%m', created_at)
        ORDER BY month
    ''').fetchall()
    
    # File type distribution
    file_type_stats = conn.execute('''
        SELECT 
            file_type,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence,
            SUM(CASE WHEN result = 'Fake' THEN 1 ELSE 0 END) as fake_count
        FROM analyses
        GROUP BY file_type
    ''').fetchall()
    
    # User activity distribution
    user_activity = conn.execute('''
        SELECT 
            CASE 
                WHEN analysis_count = 0 THEN 'No Activity'
                WHEN analysis_count <= 5 THEN 'Low (1-5)'
                WHEN analysis_count <= 20 THEN 'Medium (6-20)'
                WHEN analysis_count <= 50 THEN 'High (21-50)'
                ELSE 'Very High (50+)'
            END as activity_level,
            COUNT(*) as user_count
        FROM (
            SELECT 
                u.id,
                COUNT(a.id) as analysis_count
            FROM users u
            LEFT JOIN analyses a ON u.id = a.user_id
            WHERE u.is_admin = 0
            GROUP BY u.id
        ) user_stats
        GROUP BY activity_level
    ''').fetchall()
    
    # Detection accuracy trends
    accuracy_trends = conn.execute('''
        SELECT 
            DATE(created_at) as date,
            AVG(confidence) as avg_confidence,
            AVG(accuracy) as avg_accuracy,
            AVG(trust_score) as avg_trust_score
        FROM analyses 
        WHERE created_at >= date('now', '-30 days')
        GROUP BY DATE(created_at)
        ORDER BY date
    ''').fetchall()
    
    conn.close()
    
    return render_template('admin/analytics.html',
                         monthly_stats=monthly_stats,
                         file_type_stats=file_type_stats,
                         user_activity=user_activity,
                         accuracy_trends=accuracy_trends)

@app.route('/admin/export')
@admin_required
def admin_export():
    """Export system data"""
    export_type = request.args.get('type', 'analyses')
    format_type = request.args.get('format', 'csv')
    
    conn = get_db_connection()
    
    if export_type == 'analyses':
        data = conn.execute('''
            SELECT 
                a.id,
                u.email as user_email,
                a.filename,
                a.file_type,
                a.result,
                a.confidence,
                a.accuracy,
                a.trust_score,
                a.created_at
            FROM analyses a
            JOIN users u ON a.user_id = u.id
            ORDER BY a.created_at DESC
        ''').fetchall()
        
        filename = f'analyses_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
    elif export_type == 'users':
        data = conn.execute('''
            SELECT 
                u.id,
                u.email,
                u.is_admin,
                u.created_at,
                COUNT(a.id) as analysis_count,
                MAX(a.created_at) as last_analysis
            FROM users u
            LEFT JOIN analyses a ON u.id = a.user_id
            GROUP BY u.id
            ORDER BY u.created_at DESC
        ''').fetchall()
        
        filename = f'users_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    
    conn.close()
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    if data:
        # Write header
        writer.writerow(data[0].keys())
        
        # Write data
        for row in data:
            writer.writerow(row)
    
    # Create response
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

@app.route('/admin/user/<int:user_id>/toggle_admin', methods=['POST'])
@admin_required
def toggle_user_admin(user_id):
    """Toggle user admin status"""
    conn = get_db_connection()
    
    # Get current status
    user = conn.execute('SELECT is_admin FROM users WHERE id = ?', (user_id,)).fetchone()
    
    if not user:
        conn.close()
        return jsonify({'success': False, 'message': 'User not found'})
    
    # Don't allow removing admin from self
    if user_id == session['user_id'] and user['is_admin']:
        conn.close()
        return jsonify({'success': False, 'message': 'Cannot remove admin from yourself'})
    
    # Toggle admin status
    new_status = not user['is_admin']
    conn.execute('UPDATE users SET is_admin = ? WHERE id = ?', (new_status, user_id))
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True, 
        'message': f'User {"promoted to" if new_status else "removed from"} admin',
        'is_admin': new_status
    })

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@admin_required
def delete_user(user_id):
    """Delete user and all their data"""
    if user_id == session['user_id']:
        return jsonify({'success': False, 'message': 'Cannot delete yourself'})
    
    conn = get_db_connection()
    
    # Get user's analyses to delete files
    analyses = conn.execute('SELECT file_path FROM analyses WHERE user_id = ?', (user_id,)).fetchall()
    
    # Delete files
    for analysis in analyses:
        try:
            os.remove(analysis['file_path'])
        except OSError:
            pass
    
    # Delete user data
    conn.execute('DELETE FROM bookmarks WHERE user_id = ?', (user_id,))
    conn.execute('DELETE FROM api_keys WHERE user_id = ?', (user_id,))
    conn.execute('DELETE FROM analyses WHERE user_id = ?', (user_id,))
    conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'User deleted successfully'})

@app.route('/admin/analysis/<int:analysis_id>/delete', methods=['POST'])
@admin_required
def admin_delete_analysis(analysis_id):
    """Delete analysis as admin"""
    conn = get_db_connection()
    
    # Get analysis to delete file
    analysis = conn.execute('SELECT file_path FROM analyses WHERE id = ?', (analysis_id,)).fetchone()
    
    if analysis:
        # Delete file
        try:
            os.remove(analysis['file_path'])
        except OSError:
            pass
        
        # Delete from database
        conn.execute('DELETE FROM bookmarks WHERE analysis_id = ?', (analysis_id,))
        conn.execute('DELETE FROM analyses WHERE id = ?', (analysis_id,))
        conn.commit()
    
    conn.close()
    
    return jsonify({'success': True, 'message': 'Analysis deleted successfully'})

@app.route('/developer')
@login_required
def developer_dashboard():
    """Developer dashboard for API management"""
    conn = get_db_connection()
    
    # Get user's API keys
    api_keys = conn.execute('''
        SELECT * FROM api_keys 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (session['user_id'],)).fetchall()
    
    # Get API usage statistics
    api_stats = conn.execute('''
        SELECT 
            COUNT(*) as total_requests,
            COUNT(CASE WHEN created_at >= date('now', '-7 days') THEN 1 END) as requests_this_week,
            COUNT(CASE WHEN created_at >= date('now', '-1 day') THEN 1 END) as requests_today
        FROM api_requests 
        WHERE user_id = ?
    ''', (session['user_id'],)).fetchone()
    
    # Get recent API requests
    recent_requests = conn.execute('''
        SELECT * FROM api_requests 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 10
    ''', (session['user_id'],)).fetchall()
    
    conn.close()
    
    return render_template('developer.html', 
                         api_keys=api_keys, 
                         api_stats=api_stats,
                         recent_requests=recent_requests)

@app.route('/api/generate_key', methods=['POST'])
@login_required
def generate_api_key():
    """Generate new API key"""
    key_name = request.json.get('name', 'Default Key')
    
    # Generate secure API key
    api_key = f"df_{secrets.token_urlsafe(32)}"
    
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO api_keys (user_id, name, api_key, is_active) 
        VALUES (?, ?, ?, 1)
    ''', (session['user_id'], key_name, api_key))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'api_key': api_key, 'message': 'API key generated successfully'})

@app.route('/api/revoke_key/<int:key_id>', methods=['POST'])
@login_required
def revoke_api_key(key_id):
    """Revoke API key"""
    conn = get_db_connection()
    conn.execute('''
        UPDATE api_keys 
        SET is_active = 0 
        WHERE id = ? AND user_id = ?
    ''', (key_id, session['user_id']))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'API key revoked successfully'})

def require_api_key(f):
    """Decorator to require valid API key for API endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        conn = get_db_connection()
        key_data = conn.execute('''
            SELECT ak.*, u.email 
            FROM api_keys ak
            JOIN users u ON ak.user_id = u.id
            WHERE ak.api_key = ? AND ak.is_active = 1
        ''', (api_key,)).fetchone()
        conn.close()
        
        if not key_data:
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Store user info for the request
        g.api_user_id = key_data['user_id']
        g.api_user_email = key_data['email']
        g.api_key_id = key_data['id']
        
        return f(*args, **kwargs)
    
    return decorated_function

def log_api_request(endpoint, method, status_code, response_time=None):
    """Log API request for analytics"""
    if hasattr(g, 'api_user_id'):
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO api_requests (user_id, api_key_id, endpoint, method, status_code, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (g.api_user_id, g.api_key_id, endpoint, method, status_code, response_time))
        conn.commit()
        conn.close()

# API Endpoints
@app.route('/api/v1/analyze', methods=['POST'])
@require_api_key
def api_analyze():
    """API endpoint for file analysis"""
    start_time = time.time()
    
    try:
        if 'file' not in request.files:
            log_api_request('/api/v1/analyze', 'POST', 400)
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            log_api_request('/api/v1/analyze', 'POST', 400)
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            log_api_request('/api/v1/analyze', 'POST', 400)
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Analyze file
        file_type = analyzer.get_file_type(filename)
        
        if file_type == 'image':
            analysis_result = analyzer.analyze_image(file_path, filename)
        elif file_type == 'video':
            analysis_result = analyzer.analyze_video(file_path, filename)
        else:
            log_api_request('/api/v1/analyze', 'POST', 400)
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Save to database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
                    INSERT INTO analyses (user_id, filename, file_path, file_type, confidence, accuracy, result, trust_score, analysis_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            g.api_user_id,
            filename,
            file_path,
            file_type,
            float(analysis_result['confidence']),
            float(analysis_result['accuracy']),
            analysis_result['result'],
            float(analysis_result['trust_score']),
            analysis_result['analysis_data']
        ))
        conn.commit()
        analysis_id = cursor.lastrowid
        cursor.close()
        conn.close()
        
        # Prepare response
        response_data = {
            'analysis_id': analysis_id,
            'filename': filename,
            'file_type': file_type,
            'result': analysis_result['result'],
            'confidence': float(analysis_result['confidence']),
            'accuracy': float(analysis_result['accuracy']),
            'trust_score': float(analysis_result['trust_score']),
            'analysis_data': json.loads(analysis_result['analysis_data'])
        }
        
        response_time = time.time() - start_time
        log_api_request('/api/v1/analyze', 'POST', 200, response_time)
        
        return jsonify(response_data)
    
    except Exception as e:
        response_time = time.time() - start_time
        log_api_request('/api/v1/analyze', 'POST', 500, response_time)
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/analysis/<int:analysis_id>', methods=['GET'])
@require_api_key
def api_get_analysis(analysis_id):
    """Get analysis results by ID"""
    start_time = time.time()
    
    try:
        conn = get_db_connection()
        analysis = conn.execute('''
            SELECT * FROM analyses 
            WHERE id = ? AND user_id = ?
        ''', (analysis_id, g.api_user_id)).fetchone()
        conn.close()
        
        if not analysis:
            log_api_request(f'/api/v1/analysis/{analysis_id}', 'GET', 404)
            return jsonify({'error': 'Analysis not found'}), 404
        
        response_data = {
            'analysis_id': analysis['id'],
            'filename': analysis['filename'],
            'file_type': analysis['file_type'],
            'result': analysis['result'],
            'confidence': analysis['confidence'],
            'accuracy': analysis['accuracy'],
            'trust_score': analysis['trust_score'],
            'created_at': analysis['created_at'],
            'analysis_data': json.loads(analysis['analysis_data'])
        }
        
        response_time = time.time() - start_time
        log_api_request(f'/api/v1/analysis/{analysis_id}', 'GET', 200, response_time)
        
        return jsonify(response_data)
    
    except Exception as e:
        response_time = time.time() - start_time
        log_api_request(f'/api/v1/analysis/{analysis_id}', 'GET', 500, response_time)
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/analyses', methods=['GET'])
@require_api_key
def api_list_analyses():
    """List user's analyses with pagination"""
    start_time = time.time()
    
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)  # Max 100 per page
        result_filter = request.args.get('result', '')
        file_type_filter = request.args.get('file_type', '')
        
        # Build query
        where_conditions = ['user_id = ?']
        params = [g.api_user_id]
        
        if result_filter:
            where_conditions.append('result = ?')
            params.append(result_filter)
        
        if file_type_filter:
            where_conditions.append('file_type = ?')
            params.append(file_type_filter)
        
        where_clause = ' AND '.join(where_conditions)
        
        conn = get_db_connection()
        
        # Get total count
        total = conn.execute(f'SELECT COUNT(*) as total FROM analyses WHERE {where_clause}', params).fetchone()['total']
        
        # Get analyses
        offset = (page - 1) * per_page
        analyses = conn.execute(f'''
            SELECT id, filename, file_type, result, confidence, accuracy, trust_score, created_at
            FROM analyses 
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        ''', params + [per_page, offset]).fetchall()
        
        conn.close()
        
        response_data = {
            'analyses': [dict(analysis) for analysis in analyses],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        }
        
        response_time = time.time() - start_time
        log_api_request('/api/v1/analyses', 'GET', 200, response_time)
        
        return jsonify(response_data)
    
    except Exception as e:
        response_time = time.time() - start_time
        log_api_request('/api/v1/analyses', 'GET', 500, response_time)
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/batch_analyze', methods=['POST'])
@require_api_key
def api_batch_analyze():
    """Batch analyze multiple files"""
    start_time = time.time()
    
    try:
        if 'files' not in request.files:
            log_api_request('/api/v1/batch_analyze', 'POST', 400)
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if len(files) > 10:  # Limit batch size
            log_api_request('/api/v1/batch_analyze', 'POST', 400)
            return jsonify({'error': 'Maximum 10 files per batch'}), 400
        
        results = []
        
        for file in files:
            if file.filename == '' or not allowed_file(file.filename):
                continue
            
            # Save file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Analyze file
            file_type = analyzer.get_file_type(filename)
            
            if file_type == 'image':
                analysis_result = analyzer.analyze_image(file_path, filename)
            elif file_type == 'video':
                analysis_result = analyzer.analyze_video(file_path, filename)
            else:
                continue
            
            # Save to database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                    INSERT INTO analyses (user_id, filename, file_path, file_type, confidence, accuracy, result, trust_score, analysis_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                g.api_user_id,
                filename,
                file_path,
                file_type,
                float(analysis_result['confidence']),
                float(analysis_result['accuracy']),
                analysis_result['result'],
                float(analysis_result['trust_score']),
                analysis_result['analysis_data']
            ))
            conn.commit()
            analysis_id = cursor.lastrowid
            cursor.close()
            conn.close()
            
            results.append({
                'analysis_id': analysis_id,
                'filename': filename,
                'file_type': file_type,
                'result': analysis_result['result'],
                'confidence': float(analysis_result['confidence']),
                'trust_score': float(analysis_result['trust_score'])
            })
        
        response_time = time.time() - start_time
        log_api_request('/api/v1/batch_analyze', 'POST', 200, response_time)
        
        return jsonify({
            'processed_files': len(results),
            'results': results
        })
    
    except Exception as e:
        response_time = time.time() - start_time
        log_api_request('/api/v1/batch_analyze', 'POST', 500, response_time)
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/stats', methods=['GET'])
@require_api_key
def api_user_stats():
    """Get user statistics via API"""
    start_time = time.time()
    
    try:
        conn = get_db_connection()
        
        stats = conn.execute('''
            SELECT 
                COUNT(*) as total_analyses,
                AVG(confidence) as avg_confidence,
                SUM(CASE WHEN result = 'Fake' THEN 1 ELSE 0 END) as fake_count,
                SUM(CASE WHEN result = 'Real' THEN 1 ELSE 0 END) as real_count,
                COUNT(CASE WHEN created_at >= date('now', '-7 days') THEN 1 END) as analyses_this_week
            FROM analyses 
            WHERE user_id = ?
        ''', (g.api_user_id,)).fetchone()
        
        conn.close()
        
        response_data = dict(stats)
        
        response_time = time.time() - start_time
        log_api_request('/api/v1/stats', 'GET', 200, response_time)
        
        return jsonify(response_data)
    
    except Exception as e:
        response_time = time.time() - start_time
        log_api_request('/api/v1/stats', 'GET', 500, response_time)
        return jsonify({'error': str(e)}), 500

@app.route('/reports')
@login_required
def reports_dashboard():
    """Reports dashboard"""
    conn = get_db_connection()
    
    # Get user's recent reports
    reports = conn.execute('''
        SELECT * FROM reports 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 10
    ''', (session['user_id'],)).fetchall()
    
    conn.close()
    
    return render_template('reports.html', reports=reports)

@app.route('/generate_report', methods=['POST'])
@login_required
def generate_report():
    """Generate analysis report"""
    try:
        report_type = request.json.get('type', 'analysis')
        analysis_id = request.json.get('analysis_id')
        
        if report_type == 'analysis' and analysis_id:
            # Generate single analysis report
            conn = get_db_connection()
            analysis = conn.execute('''
                SELECT * FROM analyses 
                WHERE id = ? AND user_id = ?
            ''', (analysis_id, session['user_id'])).fetchone()
            conn.close()
            
            if not analysis:
                return jsonify({'success': False, 'message': 'Analysis not found'})
            
            # Generate PDF report
            report_filename = f"analysis_report_{analysis_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            report_path = os.path.join('reports', report_filename)
            os.makedirs('reports', exist_ok=True)
            
            success = report_generator.generate_analysis_report(dict(analysis), report_path)
            
            if success:
                # Save report record
                conn = get_db_connection()
                conn.execute('''
                    INSERT INTO reports (user_id, report_type, file_path, analysis_id)
                    VALUES (?, ?, ?, ?)
                ''', (session['user_id'], 'analysis', report_path, analysis_id))
                conn.commit()
                conn.close()
                
                # Send email notification
                email_service.send_analysis_complete_notification(
                    session['user_email'], 
                    dict(analysis)
                )
                
                return jsonify({
                    'success': True, 
                    'message': 'Report generated successfully',
                    'download_url': f'/download_report/{report_filename}'
                })
            else:
                return jsonify({'success': False, 'message': 'Error generating report'})
        
        elif report_type == 'weekly':
            # Generate weekly report
            conn = get_db_connection()
            
            # Get user data
            user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
            
            # Get week's analyses
            week_start = datetime.now() - timedelta(days=7)
            analyses = conn.execute('''
                SELECT * FROM analyses 
                WHERE user_id = ? AND created_at >= ?
                ORDER BY created_at DESC
            ''', (session['user_id'], week_start.isoformat())).fetchall()
            
            conn.close()
            
            # Generate PDF report
            report_filename = f"weekly_report_{session['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            report_path = os.path.join('reports', report_filename)
            
            success = report_generator.generate_weekly_report(
                dict(user), 
                [dict(a) for a in analyses], 
                report_path
            )
            
            if success:
                # Save report record
                conn = get_db_connection()
                conn.execute('''
                    INSERT INTO reports (user_id, report_type, file_path)
                    VALUES (?, ?, ?)
                ''', (session['user_id'], 'weekly', report_path))
                conn.commit()
                conn.close()
                
                return jsonify({
                    'success': True, 
                    'message': 'Weekly report generated successfully',
                    'download_url': f'/download_report/{report_filename}'
                })
            else:
                return jsonify({'success': False, 'message': 'Error generating weekly report'})
        
        else:
            return jsonify({'success': False, 'message': 'Invalid report type'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/download_report/<filename>')
@login_required
def download_report(filename):
    """Download generated report"""
    report_path = os.path.join('reports', filename)
    
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    else:
        flash('Report not found', 'error')
        return redirect(url_for('reports_dashboard'))

@app.route('/email_report', methods=['POST'])
@login_required
def email_report():
    """Email report to user"""
    try:
        report_filename = request.json.get('filename')
        email_address = request.json.get('email', session['user_email'])
        
        report_path = os.path.join('reports', report_filename)
        
        if os.path.exists(report_path):
            subject = f"Deepfake Analysis Report - {report_filename}"
            content = """
            <h3>Your Deepfake Analysis Report</h3>
            <p>Please find your detailed analysis report attached.</p>
            <p>This report contains comprehensive information about your deepfake detection analysis.</p>
            """
            
            success = email_service.send_report_with_attachment(
                email_address, 
                subject, 
                content, 
                report_path
            )
            
            if success:
                return jsonify({'success': True, 'message': 'Report emailed successfully'})
            else:
                return jsonify({'success': False, 'message': 'Error sending email'})
        else:
            return jsonify({'success': False, 'message': 'Report not found'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/live_capture')
@login_required
def live_capture():
    """Live webcam capture and analysis"""
    return render_template('live_capture.html')

@app.route('/api/v1/live_analyze', methods=['POST'])
@require_api_key
def api_live_analyze():
    """API endpoint for live frame analysis (optimized for real-time)"""
    start_time = time.time()
    
    try:
        if 'frame' not in request.files:
            log_api_request('/api/v1/live_analyze', 'POST', 400)
            return jsonify({'error': 'No frame provided'}), 400
        
        frame = request.files['frame']
        if frame.filename == '':
            log_api_request('/api/v1/live_analyze', 'POST', 400)
            return jsonify({'error': 'No frame selected'}), 400
        
        # Save frame temporarily
        filename = f"live_frame_{uuid.uuid4()}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        frame.save(file_path)
        
        # Quick analysis for live feed (reduced processing for speed)
        analysis_result = analyzer.analyze_image_fast(file_path, filename)
        
        # Clean up temporary file
        try:
            os.remove(file_path)
        except OSError:
            pass
        
        # Prepare lightweight response for real-time use
        response_data = {
            'result': analysis_result['result'],
            'confidence': float(analysis_result['confidence']),
            'trust_score': float(analysis_result['trust_score']),
            'processing_time': time.time() - start_time,
            'faces_detected': analysis_result.get('faces_detected', 0),
            'face_regions': analysis_result.get('face_regions', [])
        }
        
        response_time = time.time() - start_time
        log_api_request('/api/v1/live_analyze', 'POST', 200, response_time)
        
        return jsonify(response_data)
    
    except Exception as e:
        response_time = time.time() - start_time
        log_api_request('/api/v1/live_analyze', 'POST', 500, response_time)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
