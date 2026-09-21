"""
auth_pages.py
Login/Signup, KYC submission form, and status screens (pending/rejected/
cancelled) shown before a user can reach the main cyberbullying-detection
project.
"""

import streamlit as st
import db


def apply_theme():
    """Simple dark/light theme toggle using custom CSS (independent of
    Streamlit's own theme setting, so it works the same on every machine)."""
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False

    if st.session_state.dark_mode:
        bg, fg, card = "#0e1117", "#fafafa", "#1c1f26"
    else:
        bg, fg, card = "#ffffff", "#1a1a1a", "#f7f7f9"

    st.markdown(f"""
        <style>
        .stApp {{ background-color: {bg}; color: {fg}; }}
        .auth-card {{
            background: {card}; padding: 30px; border-radius: 14px;
            max-width: 420px; margin: 40px auto; box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        </style>
    """, unsafe_allow_html=True)


def theme_toggle_button():
    col1, col2 = st.columns([5, 1])
    with col2:
        label = "☀️ Light" if st.session_state.get('dark_mode') else "🌙 Dark"
        if st.button(label, key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.get('dark_mode', False)
            st.rerun()


def render_login_signup():
    apply_theme()
    theme_toggle_button()

    st.markdown("<h1 style='text-align:center;'>🛡️ SafeSpace Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; opacity:0.7;'>KYC-verified access to the Cyberbullying Detection system</p>", unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

            if submitted:
                user = db.verify_login(username, password)
                if user:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    with tab_signup:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if submitted:
                if not new_username or not new_password:
                    st.error("Username and password are required.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(new_password) < 4:
                    st.error("Password should be at least 4 characters (demo minimum).")
                else:
                    ok, msg = db.create_user(new_username, new_password)
                    if ok:
                        st.success(f"{msg} Please log in, then complete KYC verification.")
                    else:
                        st.error(msg)

    st.divider()
    with st.expander("🔧 Admin access"):
        with st.form("admin_login_form"):
            admin_user = st.text_input("Admin username", key="admin_user")
            admin_pass = st.text_input("Admin password", type="password", key="admin_pass")
            admin_submit = st.form_submit_button("Admin Login")
            if admin_submit:
                user = db.verify_login(admin_user, admin_pass)
                if user and user['is_admin']:
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Invalid admin credentials.")
        


def render_kyc_form(user):
    apply_theme()
    st.title("📋 KYC Verification")
    st.info("This is a demo KYC form for academic project purposes only. "
            "**Do not enter real personal or government ID information** — "
            "use sample/fake data for the demo.")

    with st.form("kyc_form"):
        full_name = st.text_input("Full Name (as per ID)")
        id_number = st.text_input("ID Number (e.g. sample Aadhaar-style number)",
                                   placeholder="XXXX XXXX XXXX (use fake data)")
        dob = st.date_input("Date of Birth")
        address = st.text_area("Address")
        submitted = st.form_submit_button("Submit for Verification", type="primary", use_container_width=True)

        if submitted:
            if not full_name or not id_number or not address:
                st.error("Please fill in all fields.")
            else:
                db.submit_kyc(user['id'], full_name, id_number, dob, address)
                st.success("KYC submitted! An admin will review your details shortly.")
                st.rerun()

    if st.button("Log out"):
        del st.session_state.user
        st.rerun()


def render_status_screen(status: str, user):
    apply_theme()
    st.title("🛡️ SafeSpace Platform")

    if status == 'pending':
        st.warning("⏳ Your KYC verification is pending admin approval. "
                   "Please check back later.")
    elif status == 'rejected':
        st.error("❌ Your KYC verification was rejected. Please contact support "
                 "or try submitting again with correct details.")
        if st.button("Resubmit KYC"):
            st.session_state.force_kyc_form = True
            st.rerun()
    elif status == 'cancelled':
        st.error("🚫 Your access has been **cancelled** due to repeated "
                 "cyberbullying/policy violations detected on your account. "
                 "Contact an administrator if you believe this is a mistake.")

    if st.button("Log out"):
        del st.session_state.user
        st.rerun()
