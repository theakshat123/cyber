"""
admin_page.py
Hidden admin dashboard: approve/reject pending KYC submissions, view all
users, see flagged-comment counts, and manually reinstate cancelled users.
"""

import streamlit as st
import db


def render_admin_dashboard(user):
    st.title("🔧 Admin Dashboard")
    st.caption(f"Logged in as admin: **{user['username']}**")

    if st.button("Log out"):
        del st.session_state.user
        st.rerun()

    st.divider()

    tab_pending, tab_all = st.tabs(["📋 Pending KYC Requests", "👥 All Users"])

    with tab_pending:
        pending = db.get_pending_kyc_list()
        if not pending:
            st.info("No pending KYC requests.")
        for record in pending:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Username:** {record['username']}")
                    st.markdown(f"**Full Name:** {record['full_name']}")
                    st.markdown(f"**ID Number:** {record['id_number']}")
                    st.markdown(f"**DOB:** {record['dob']}")
                    st.markdown(f"**Address:** {record['address']}")
                    st.caption(f"Submitted: {record['submitted_at']}")
                with col2:
                    if st.button("✅ Approve", key=f"approve_{record['user_id']}", use_container_width=True):
                        db.decide_kyc(record['user_id'], approve=True)
                        st.rerun()
                    if st.button("❌ Reject", key=f"reject_{record['user_id']}", use_container_width=True):
                        db.decide_kyc(record['user_id'], approve=False)
                        st.rerun()

    with tab_all:
        all_users = db.get_all_users_with_kyc()
        if not all_users:
            st.info("No users yet.")
        else:
            status_colors = {
                'approved': '🟢', 'pending': '🟡', 'rejected': '🔴',
                'cancelled': '⛔', 'not_submitted': '⚪',
            }
            for record in all_users:
                icon = status_colors.get(record['status'], '⚪')
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2, 2, 1])
                    with c1:
                        st.markdown(f"{icon} **{record['username']}** — {record['status']}")
                        st.caption(f"Flagged comments: {record['flagged_count']} / {db.FLAG_THRESHOLD}")
                    with c2:
                        if record['full_name']:
                            st.caption(f"{record['full_name']} | {record['id_number']}")
                    with c3:
                        if record['status'] == 'cancelled':
                            if st.button("♻️ Reinstate", key=f"reinstate_{record['user_id']}", use_container_width=True):
                                db.reinstate_kyc(record['user_id'])
                                st.rerun()
