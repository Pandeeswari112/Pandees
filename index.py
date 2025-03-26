# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 17:49:35 2025

@author: pande
"""

import streamlit as st
import sqlite3
import hashlib

# Initialize user database
def init_user_adb():
    conn = sqlite3.connect("users1.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY, 
                    username TEXT UNIQUE, 
                    password TEXT, 
                    login_count INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

# Initialize admin database
def init_admin_db():
    conn = sqlite3.connect("admin_users1.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS admins (
                    id INTEGER PRIMARY KEY, 
                    username TEXT UNIQUE, 
                    password TEXT)''')
    conn.commit()
    conn.close()

# Hash passwords for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Register user
def register_user(username, password):
    conn = sqlite3.connect("users1.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

# Register admin
def register_admin(username, password):
    conn = sqlite3.connect("admin_users1.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO admins (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

# Validate user login
def validate_user_login(username, password):
    conn = sqlite3.connect("users1.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
    user = c.fetchone()
    if user:
        c.execute("UPDATE users SET login_count = login_count + 1 WHERE username = ?", (username,))
        conn.commit()
    conn.close()
    return user

# Validate admin login
def validate_admin_login(username, password):
    conn = sqlite3.connect("admin_users1.db")
    c = conn.cursor()
    c.execute("SELECT * FROM admins WHERE username = ? AND password = ?", (username, hash_password(password)))
    admin = c.fetchone()
    conn.close()
    return admin

# Main app
def main():
    # Page navigation
    if 'page' not in st.session_state:
        st.session_state['page'] = 'index'

    if st.session_state['page'] == 'index':
        st.title("DDoS Attack Detection System")
        st.header("Welcome to the DDoS Attack Detection System")
        st.write("Your security is our priority.")

        st.subheader("System Status")
        status = "Monitoring..."  # This is a placeholder, replace it with actual system status
        st.text(status)

        st.subheader("Alerts")
        alert = "No current alerts."  # This is a placeholder, replace it with actual alerts
        st.text(alert)

        if st.button('Go to Register'):
            st.session_state['page'] = 'register'

        # Styling
        st.markdown("""
            <style>
                .reportview-container {
                    font-family: Arial, sans-serif;
                    text-align: center;
                    padding: 20px;
                }
                .stAlert, .stHeader, .stText {
                    margin-bottom: 20px;
                }
            </style>
            """, unsafe_allow_html=True)

    elif st.session_state['page'] == 'register':
        st.title("User and Admin Login & Registration System")
        
        init_admin_db()

        menu = ["User Login", "User Register", "Admin Login", "Admin Register"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "User Login":
            st.subheader("Login to Your User Account")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                user = validate_user_login(username, password)
                if user:
                    st.success(f"Welcome {username}!")
                    st.session_state["authenticated"] = True
                    import subprocess
                    subprocess.run(['streamlit','run','dash.py'])
                else:
                    st.error("Invalid Username or Password")

        elif choice == "User Register":
            st.subheader("Create a User Account")
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.button("Register"):
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif register_user(new_username, new_password):
                    st.success("Account Created Successfully! You can now log in.")
                else:
                    st.error("Username already exists. Choose another one.")

        elif choice == "Admin Login":
            st.subheader("Login to Admin Account")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                admin = validate_admin_login(username, password)
                if admin:
                    st.success(f"Welcome {username}!")
                    st.session_state["authenticated"] = True
                    
                    # Display user login counts
                    conn = sqlite3.connect("users1.db")
                    c = conn.cursor()
                    c.execute("SELECT username, login_count FROM users")
                    users = c.fetchall()
                    conn.close()

                    st.subheader("User Login Counts")
                    for user in users:
                        st.text(f"Username: {user[0]}, Login Count: {user[1]}")
                else:
                    st.error("Invalid Username or Password")

        elif choice == "Admin Register":
            st.subheader("Create an Admin Account")
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.button("Register"):
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif register_admin(new_username, new_password):
                    st.success("Account Created Successfully! You can now log in.")
                else:
                    st.error("Username already exists. Choose another one.")

        if st.button('Go to Index'):
            st.session_state['page'] = 'index'

if __name__ == "__main__":
    main()
