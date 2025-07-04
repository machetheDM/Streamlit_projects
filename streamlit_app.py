import streamlit as st

#st.balloons()
st.header('st.button')

if st.button('Say Hello'):
    st.write('Button is Pressed')
else:
    st.write('Good Bye')
 