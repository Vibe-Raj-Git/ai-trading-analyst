# auth.py - User Authentication Module

import re
from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, g
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
import sqlite3
import os
from datetime import datetime, timedelta
import secrets
import string

# Create blueprint
auth_bp = Blueprint('auth', __name__)

# Initialize extensions
bcrypt = Bcrypt()
mail = Mail()

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'users.db')

# ========== Database Setup ==========
def init_user_db():
    """Initialize user database with tables"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            email_verified INTEGER DEFAULT 0,
            verification_token TEXT,
            reset_token TEXT,
            reset_token_expiry TIMESTAMP
        )
    ''')
    
    # Login history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT,
            user_agent TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # User activity log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Check if admin exists, if not create default admin
    cursor.execute("SELECT * FROM users WHERE role = 'admin'")
    if not cursor.fetchone():
        # Create default admin (change credentials after first login!)
        admin_password = bcrypt.generate_password_hash('Admin@123').decode('utf-8')
        cursor.execute('''
            INSERT INTO users (email, username, password_hash, role, email_verified)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin@aisauda.com', 'admin', admin_password, 'admin', 1))
    
    conn.commit()
    conn.close()

# ========== User Class for Flask-Login ==========
# ========== User Class for Flask-Login ==========
class User(UserMixin):
    def __init__(self, id, email, username, role, is_active):
        self.id = id
        self.email = email
        self.username = username
        self.role = role
        self._is_active = bool(is_active)  # Store in private variable
    
    @property
    def is_active(self):
        """Required by Flask-Login - returns True if account is active"""
        return self._is_active
    
    @is_active.setter
    def is_active(self, value):
        """Allow setting is_active"""
        self._is_active = bool(value)
    
    @property
    def is_authenticated(self):
        """Required by Flask-Login"""
        return True
    
    @property
    def is_anonymous(self):
        """Required by Flask-Login"""
        return False
    
    def get_id(self):
        """Required by Flask-Login - returns unique identifier"""
        return str(self.id)

    @staticmethod
    def get(user_id):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, username, role, is_active FROM users WHERE id = ?", (user_id,))
        user_data = cursor.fetchone()
        conn.close()
        if user_data:
            return User(user_data[0], user_data[1], user_data[2], user_data[3], user_data[4])
        return None

    @staticmethod
    def find_by_email(email):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, username, role, is_active FROM users WHERE email = ?", (email,))
        user_data = cursor.fetchone()
        conn.close()
        if user_data:
            return User(user_data[0], user_data[1], user_data[2], user_data[3], user_data[4])
        return None

    def is_admin(self):
        return self.role == 'admin'
# ========== Helper Functions ==========
def is_valid_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_strong_password(password):
    """Check if password is strong"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    return True, "OK"

def generate_verification_token():
    """Generate email verification token"""
    return secrets.token_urlsafe(32)

def log_activity(user_id, action, details=None):
    """Log user activity"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_activity (user_id, action, details)
        VALUES (?, ?, ?)
    ''', (user_id, action, details))
    conn.commit()
    conn.close()

def log_login(user_id, ip_address, user_agent):
    """Log login attempt"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO login_history (user_id, ip_address, user_agent)
        VALUES (?, ?, ?)
    ''', (user_id, ip_address, user_agent))
    conn.commit()
    conn.close()

# ========== Admin Required Decorator ==========
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin():
            flash('Admin access required.', 'danger')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function

# ========== Routes ==========
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'
        
        if not email or not password:
            flash('Please enter both email and password.', 'danger')
            return render_template('login.html')
        
        user = User.find_by_email(email)
        
        if not user:
            flash('Invalid email or password.', 'danger')
            return render_template('login.html')
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE email = ?", (email,))
        stored_hash = cursor.fetchone()
        conn.close()
        
        if stored_hash and bcrypt.check_password_hash(stored_hash[0], password):
            if not user.is_active:
                flash('Your account has been deactivated. Contact admin.', 'danger')
                return render_template('login.html')
            
            login_user(user, remember=remember)
            
            # Log login
            log_login(user.id, request.remote_addr, request.headers.get('User-Agent'))
            log_activity(user.id, 'login', f'Logged in from {request.remote_addr}')
            
            # Update last login
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", 
                          (datetime.now(), user.id))
            conn.commit()
            conn.close()
            
            flash(f'Welcome back, {user.username}!', 'success')
            
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password.', 'danger')
    
    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        if not email or not username or not password:
            flash('All fields are required.', 'danger')
            return render_template('register.html')
        
        if not is_valid_email(email):
            flash('Please enter a valid email address.', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        
        is_strong, msg = is_strong_password(password)
        if not is_strong:
            flash(msg, 'danger')
            return render_template('register.html')
        
        # Check if user exists
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ? OR username = ?", (email, username))
        existing = cursor.fetchone()
        
        if existing:
            flash('Email or username already exists.', 'danger')
            conn.close()
            return render_template('register.html')
        
        # Create user
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        verification_token = generate_verification_token()
        
        cursor.execute('''
            INSERT INTO users (email, username, password_hash, verification_token)
            VALUES (?, ?, ?, ?)
        ''', (email, username, password_hash, verification_token))
        
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        # Send verification email (optional)
        try:
            verification_link = url_for('auth.verify_email', token=verification_token, _external=True)
            msg = Message('Verify Your Email - AI Trading Analyst',
                         sender='noreply@aisauda.com',
                         recipients=[email])
            msg.body = f'''Hello {username},

Thank you for registering with AI Trading Analyst.

Please click the link below to verify your email address:
{verification_link}

If you did not register, please ignore this email.

Best regards,
AI Trading Analyst Team
'''
            mail.send(msg)
        except Exception as e:
            print(f"Email sending failed: {e}")
        
        flash('Registration successful! Please check your email to verify your account.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('register.html')

@auth_bp.route('/verify-email/<token>')
def verify_email(token):
    """Verify user email"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE verification_token = ?", (token,))
    user = cursor.fetchone()
    
    if user:
        cursor.execute("UPDATE users SET email_verified = 1, verification_token = NULL WHERE id = ?", (user[0],))
        conn.commit()
        flash('Email verified successfully! You can now log in.', 'success')
    else:
        flash('Invalid verification token.', 'danger')
    
    conn.close()
    return redirect(url_for('auth.login'))

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout"""
    log_activity(current_user.id, 'logout', 'User logged out')
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/profile')
@login_required
def profile():
    """User profile page"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT email, username, role, created_at, last_login, email_verified 
        FROM users WHERE id = ?
    ''', (current_user.id,))
    user_data = cursor.fetchone()
    
    cursor.execute('''
        SELECT login_time, ip_address FROM login_history 
        WHERE user_id = ? ORDER BY login_time DESC LIMIT 10
    ''', (current_user.id,))
    login_history = cursor.fetchall()
    
    conn.close()
    
    return render_template('profile.html', 
                         user=user_data, 
                         login_history=login_history,
                         current_user=current_user)

@auth_bp.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """Change user password"""
    current_pwd = request.form.get('current_password')
    new_pwd = request.form.get('new_password')
    confirm_pwd = request.form.get('confirm_password')
    
    if new_pwd != confirm_pwd:
        flash('New passwords do not match.', 'danger')
        return redirect(url_for('auth.profile'))
    
    is_strong, msg = is_strong_password(new_pwd)
    if not is_strong:
        flash(msg, 'danger')
        return redirect(url_for('auth.profile'))
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE id = ?", (current_user.id,))
    stored_hash = cursor.fetchone()
    
    if stored_hash and bcrypt.check_password_hash(stored_hash[0], current_pwd):
        new_hash = bcrypt.generate_password_hash(new_pwd).decode('utf-8')
        cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, current_user.id))
        conn.commit()
        log_activity(current_user.id, 'password_change', 'Password changed')
        flash('Password changed successfully!', 'success')
    else:
        flash('Current password is incorrect.', 'danger')
    
    conn.close()
    return redirect(url_for('auth.profile'))

@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Forgot password - send reset link"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if user:
            reset_token = secrets.token_urlsafe(32)
            reset_expiry = datetime.now() + timedelta(hours=1)
            
            cursor.execute('''
                UPDATE users SET reset_token = ?, reset_token_expiry = ? 
                WHERE id = ?
            ''', (reset_token, reset_expiry, user[0]))
            conn.commit()
            
            reset_link = url_for('auth.reset_password', token=reset_token, _external=True)
            
            try:
                msg = Message('Password Reset - AI Trading Analyst',
                             sender='noreply@aisauda.com',
                             recipients=[email])
                msg.body = f'''Hello {user[1]},

You requested a password reset for your AI Trading Analyst account.

Click the link below to reset your password:
{reset_link}

This link will expire in 1 hour.

If you did not request this, please ignore this email.

Best regards,
AI Trading Analyst Team
'''
                mail.send(msg)
                flash('Password reset link sent to your email.', 'success')
            except Exception as e:
                print(f"Email sending failed: {e}")
                flash('Unable to send email. Please try again later.', 'danger')
        else:
            # Don't reveal if email exists
            flash('If an account exists, a reset link will be sent.', 'info')
        
        conn.close()
        return redirect(url_for('auth.login'))
    
    return render_template('forgot_password.html')

@auth_bp.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Reset password with token"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id FROM users 
        WHERE reset_token = ? AND reset_token_expiry > ?
    ''', (token, datetime.now()))
    user = cursor.fetchone()
    
    if not user:
        flash('Invalid or expired reset link.', 'danger')
        conn.close()
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('reset_password.html', token=token)
        
        is_strong, msg = is_strong_password(password)
        if not is_strong:
            flash(msg, 'danger')
            return render_template('reset_password.html', token=token)
        
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        cursor.execute('''
            UPDATE users SET password_hash = ?, reset_token = NULL, reset_token_expiry = NULL 
            WHERE id = ?
        ''', (password_hash, user[0]))
        conn.commit()
        
        flash('Password reset successfully! Please log in.', 'success')
        conn.close()
        return redirect(url_for('auth.login'))
    
    conn.close()
    return render_template('reset_password.html', token=token)