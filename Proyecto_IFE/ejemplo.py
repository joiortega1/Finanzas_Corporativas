import streamlit as st

if st.sidebar.button('Add all years'):
    year_filter = st.sidebar.multiselect('Select a year',[2000,2001,2002,2003,2004], default=[2000,2001,2002,2003,2004])
else:
    year_filter = st.sidebar.multiselect('Select a year',[2000,2001,2002,2003,2004], default=[2000])

type_filter = st.sidebar.multiselect('Choose a type',['Control','Experimental'], default=['Experimental'])

st.write(list(tuple(year_filter)), tuple(type_filter))
