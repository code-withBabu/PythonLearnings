import streamlit as st
st.title("Hello, Streamlit!")
st.write("This is a simple Streamlit application.")
button_clicked = st.button("Click me!")
st.write("You clicked the button!" if button_clicked else "Button not clicked yet.")