# admin.py - Admin Management Module

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from auth import admin_required, bcrypt, log_activity, DB_PATH
import sqlite3

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route('/dashboard')
@login_required
@admin_required
def dashboard():
    """Admin dashboard"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get statistics
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
    total_admins = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE DATE(created_at) = DATE('now')")
    new_today = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM user_activity WHERE DATE(timestamp) = DATE('now')")
    activity_today = cursor.fetchone()[0]
    
    # Get recent users
    cursor.execute('''
        SELECT id, email, username, role, created_at, last_login, email_verified 
        FROM users ORDER BY created_at DESC LIMIT 10
    ''')
    recent_users = cursor.fetchall()
    
    # Get recent activity
    cursor.execute('''
        SELECT ua.*, u.username 
        FROM user_activity ua
        JOIN users u ON ua.user_id = u.id
        ORDER BY ua.timestamp DESC LIMIT 20
    ''')
    recent_activity = cursor.fetchall()
    
    conn.close()
    
    return render_template('admin_dashboard.html',
                         total_users=total_users,
                         total_admins=total_admins,
                         new_today=new_today,
                         activity_today=activity_today,
                         recent_users=recent_users,
                         recent_activity=recent_activity)

@admin_bp.route('/users')
@login_required
@admin_required
def users():
    """Manage users page"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    offset = (page - 1) * per_page
    search = request.args.get('search', '')
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if search:
        cursor.execute('''
            SELECT COUNT(*) FROM users 
            WHERE email LIKE ? OR username LIKE ?
        ''', (f'%{search}%', f'%{search}%'))
        total = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT id, email, username, role, created_at, last_login, email_verified, is_active
            FROM users 
            WHERE email LIKE ? OR username LIKE ?
            ORDER BY created_at DESC LIMIT ? OFFSET ?
        ''', (f'%{search}%', f'%{search}%', per_page, offset))
    else:
        cursor.execute("SELECT COUNT(*) FROM users")
        total = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT id, email, username, role, created_at, last_login, email_verified, is_active
            FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?
        ''', (per_page, offset))
    
    users_list = cursor.fetchall()
    conn.close()
    
    return render_template('admin_users.html',
                         users=users_list,
                         page=page,
                         total=total,
                         per_page=per_page,
                         search=search)

@admin_bp.route('/user/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_user(user_id):
    """Edit user details"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if request.method == 'POST':
        role = request.form.get('role')
        is_active = 1 if request.form.get('is_active') == 'on' else 0
        
        cursor.execute('''
            UPDATE users SET role = ?, is_active = ? WHERE id = ?
        ''', (role, is_active, user_id))
        conn.commit()
        
        # Get username for log
        cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        username = cursor.fetchone()[0]
        
        log_activity(current_user.id, 'admin_edit_user', f'Edited user: {username}')
        flash(f'User {username} updated successfully.', 'success')
        conn.close()
        return redirect(url_for('admin.users'))
    
    cursor.execute('''
        SELECT id, email, username, role, created_at, last_login, email_verified, is_active
        FROM users WHERE id = ?
    ''', (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin.users'))
    
    return render_template('admin_edit_user.html', user=user)

@admin_bp.route('/user/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    """Delete user (cannot delete self)"""
    if user_id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin.users'))
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
    username = cursor.fetchone()
    
    if username:
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        cursor.execute("DELETE FROM login_history WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM user_activity WHERE user_id = ?", (user_id,))
        conn.commit()
        log_activity(current_user.id, 'admin_delete_user', f'Deleted user: {username[0]}')
        flash(f'User {username[0]} deleted successfully.', 'success')
    
    conn.close()
    return redirect(url_for('admin.users'))

@admin_bp.route('/user/add', methods=['GET', 'POST'])
@login_required
@admin_required
def add_user():
    """Add new user (admin only)"""
    from auth import is_valid_email, is_strong_password
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        role = request.form.get('role', 'user')
        
        # Validation
        if not email or not username or not password:
            flash('All fields are required.', 'danger')
            return render_template('admin_add_user.html')
        
        if not is_valid_email(email):
            flash('Please enter a valid email address.', 'danger')
            return render_template('admin_add_user.html')
        
        is_strong, msg = is_strong_password(password)
        if not is_strong:
            flash(msg, 'danger')
            return render_template('admin_add_user.html')
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ? OR username = ?", (email, username))
        if cursor.fetchone():
            flash('Email or username already exists.', 'danger')
            conn.close()
            return render_template('admin_add_user.html')
        
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        cursor.execute('''
            INSERT INTO users (email, username, password_hash, role, email_verified)
            VALUES (?, ?, ?, ?, ?)
        ''', (email, username, password_hash, role, 1))
        conn.commit()
        conn.close()
        
        log_activity(current_user.id, 'admin_add_user', f'Added user: {username}')
        flash(f'User {username} added successfully.', 'success')
        return redirect(url_for('admin.users'))
    
    return render_template('admin_add_user.html')

@admin_bp.route('/activity')
@login_required
@admin_required
def activity_log():
    """View user activity log"""
    page = request.args.get('page', 1, type=int)
    per_page = 50
    offset = (page - 1) * per_page
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM user_activity")
    total = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT ua.*, u.username 
        FROM user_activity ua
        JOIN users u ON ua.user_id = u.id
        ORDER BY ua.timestamp DESC LIMIT ? OFFSET ?
    ''', (per_page, offset))
    activities = cursor.fetchall()
    
    conn.close()
    
    return render_template('admin_activity.html',
                         activities=activities,
                         page=page,
                         total=total,
                         per_page=per_page)