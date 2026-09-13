import streamlit as st
import time

if 'running' not in st.session_state:
    st.session_state.running = True

if st.session_state.running:
    if st.button("Cancel"):
        st.session_state.running = False
        st.rerun()
    with st.spinner("Running..."):
        time.sleep(1.5)
    st.rerun()
else:
    st.write("Done!")
