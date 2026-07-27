import streamlit as st
import requests
import sqlite3
import hashlib

# 1. PAGE AND THEME CONFIGURATION
st.set_page_config(
    page_title="Financial RAG Assistant", 
    page_icon="📊", 
    layout="wide"
)

# Custom Styling for Background Logo Watermark / Custom Aesthetics
st.markdown("""
    <style>
    /* Subtle background styling */
    .stApp {
        background-color: #0e1117;
    }
    /* Logo accent for sidebar */
    .sidebar-logo {
        text-align: center;
        padding: 10px;
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
        border-bottom: 1px solid #262730;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. LOCAL SQLITE DATABASE INITIALIZATION
def init_db():
    conn = sqlite3.connect("data/rag_app.db")
    cursor = conn.cursor()
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    # Create history table linked to users
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            question TEXT,
            answer TEXT,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Helper utility for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_db_connection():
    return sqlite3.connect("data/rag_app.db")

# 3. SESSION STATE MANAGEMENT
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- MULTI-SCREEN NAVIGATION ---

# SCREEN A: AUTHENTICATION (Login / Register)
if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align: center;'>🔒 Secure Financial RAG Portal</h2>", unsafe_allow_html=True)
    
    # Simple tab layout for Login vs Sign Up
    auth_tab, register_tab = st.tabs(["Existing User Login", "Create New Account"])
    
    with auth_tab:
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In")
            
            if submitted:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT password FROM users WHERE username = ?", (user,))
                result = cursor.fetchone()
                conn.close()
                
                if result and result[0] == hash_password(pwd):
                    st.session_state.logged_in = True
                    st.session_state.username = user
                    # Load historical interactions into current session
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT question, answer FROM chat_history WHERE username = ? ORDER BY id ASC", (user,))
                    past_chats = cursor.fetchall()
                    conn.close()
                    
                    st.session_state.messages = []
                    for q, a in past_chats:
                        st.session_state.messages.append({"role": "user", "content": q})
                        st.session_state.messages.append({"role": "assistant", "content": a})
                        
                    st.rerun()
                else:
                    st.error("Invalid username or password configuration.")

    with register_tab:
        with st.form("register_form"):
            new_user = st.text_input("Choose Username")
            new_pwd = st.text_input("Choose Password", type="password")
            confirm_pwd = st.text_input("Confirm Password", type="password")
            registered = st.form_submit_button("Register Account")
            
            if registered:
                if not new_user or not new_pwd:
                    st.warning("Fields cannot be left blank.")
                elif new_pwd != confirm_pwd:
                    st.error("Passwords do not match.")
                else:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    try:
                        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_user, hash_password(new_pwd)))
                        conn.commit()
                        st.success("Account created successfully! Proceed to log in.")
                    except sqlite3.IntegrityError:
                        st.error("Username already taken.")
                    finally:
                        conn.close()

# SCREEN B: CORE APPLICATION DASHBOARD
else:
    # SIDEBAR: Control Deck & History
    with st.sidebar:
        st.markdown("<div class='sidebar-logo'>📊 FIN-RAG ENGINE</div>", unsafe_allow_html=True)
        st.write(f"Authenticated as: **{st.session_state.username}**")
        
        if st.button("Log Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.messages = []
            st.rerun()
            
        
        st.write("---")
        st.markdown("### 🎯 Document Focus")
        doc_filter = st.text_input("Target Document (Optional)", placeholder="e.g., WALMART_2023_10K")

        st.write("---")
        st.markdown("### 🕒 Recent Analytics Queries")
        
        # Pull live query history log from SQLite
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT question FROM chat_history WHERE username = ? ORDER BY id DESC LIMIT 8", (st.session_state.username,))
        recent_queries = cursor.fetchall()
        conn.close()
        
        if recent_queries:
            for q_tuple in recent_queries:
                # Truncate text for UI cleanliness
                short_q = q_tuple[0][:30] + "..." if len(q_tuple[0]) > 30 else q_tuple[0]
                st.caption(f"🔍 {short_q}")
        else:
            st.info("No prior questions logged.")

    # MAIN CONTENT PANEL
    st.title("📊 Corporate Filings AI Assistant")
    st.caption("Powered by a completely local Qwen-2.5-1.5B LLM & FAISS Vector Index")

    # Display active chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process live workspace queries
    if user_question := st.chat_input("Ask a question about your financial documents (e.g., Apple 2023 10-K)..."):
        
        # Display current question
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Query backend pipeline
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context chunks and generating answer..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/query", 
                        json={"question": user_question}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data["sources"]
                        
                        st.markdown(answer)
                        if sources:
                            with st.expander("📚 Viewed Reference Sources"):
                                for source in sources:
                                    st.write(f"- `{source}`")
                                    
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Save the exchange to historical records
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO chat_history (username, question, answer) VALUES (?, ?, ?)", 
                                       (st.session_state.username, user_question, answer))
                        conn.commit()
                        conn.close()
                        
                    else:
                        st.error(f"Backend API returned an error code: {response.status_code}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the backend server. Make sure your FastAPI terminal is still running!")