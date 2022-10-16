import streamlit as st

st.set_page_config(page_title="Matching with transfers")

st.markdown("## Solving and Estimating the Choo and Siow Model")
description1 = (
    "### This  application  solves for the stable matching "
    + "and estimates the parameters of the joint surplus "
    + "in a [Choo and Siow 2006](https://www.jstor.org/stable/10.1086/498585?seq=1) model "
    + "(homoskedastic, with singles).\n"
)
st.markdown(description1)
description2 = (
    "It can be used in two ways: \n"
    + "  1. 'Solve for the stable matching:' for a given joint surplus matrix and given margins\n"
    + "  2. 'Simulate and estimate': specify a surplus function and margins and estimate the parameters."
)
st.markdown(description2)
