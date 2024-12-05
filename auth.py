import streamlit as st
import json
import os

# Path to the user database (JSON file)
USER_DB_FILE = "user_db.json"

# Ensure the JSON file exists
if not os.path.exists(USER_DB_FILE):
    with open(USER_DB_FILE, "w") as f:
        json.dump({}, f)


# Load user database from JSON
def load_user_db():
    with open(USER_DB_FILE, "r") as f:
        return json.load(f)


# Save user database to JSON
def save_user_db(user_db):
    with open(USER_DB_FILE, "w") as f:
        json.dump(user_db, f, indent=4)


# Authenticate user
def authenticate_user(username, password):
    user_db = load_user_db()
    return user_db.get(username) == password


# Add a new user
def add_user(username, password):
    user_db = load_user_db()
    if username in user_db:
        return False  # User already exists
    user_db[username] = password
    save_user_db(user_db)
    return True


# Login or Signup Logic
def login_or_signup():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "current_user" not in st.session_state:
        st.session_state["current_user"] = ""

    if not st.session_state["logged_in"]:
        st.title("Login or Signup")

        # Allow user to select login or signup
        choice = st.radio("Select an option:", ["Login", "Signup"])

        if choice == "Login":
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                if authenticate_user(username, password):
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = username
                    st.success(f"Welcome back, {username}!")
                    st.stop()  # Stop execution, and allow app to continue after login
                else:
                    st.error("Invalid username or password. Please try again.")

        elif choice == "Signup":
            new_username = st.text_input("New Username", key="signup_username")
            new_password = st.text_input("New Password", type="password", key="signup_password")
            if st.button("Signup"):
                if add_user(new_username, new_password):
                    st.success(f"Account created successfully for {new_username}. You can now log in.")
                else:
                    st.error("Username already exists. Please choose a different one.")

        st.stop()  # Prevent further execution until user logs in
