import streamlit as st
from auth import login_or_signup

# Run the login/signup logic first
login_or_signup()

# If logged in, show the main app (PDF upload page)
if st.session_state["logged_in"]:
    st.sidebar.success(f"Logged in as: {st.session_state['current_user']}")
    if st.sidebar.button("Logout"):
        # Logout logic: Clear the session and redirect to login page
        st.session_state["logged_in"] = False
        st.session_state["current_user"] = ""
        st.stop()  # Stop execution to show the login/signup page after logout

    st.title("Welcome to ProXtract Insights!")
    st.write("You can now upload and process your PDFs.")

    # File upload and processing logic
    uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Process PDFs"):
        if not uploaded_files:
            st.error("Please upload at least one PDF file.")
        else:
            st.info("Processing your PDFs...")
            # Example: Add the code for processing PDF files here
            for uploaded_file in uploaded_files:
                file_path = f"uploads/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success("PDF processing complete!")
