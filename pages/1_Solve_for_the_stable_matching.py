""" Interactive Streamlit application that solves for the stable matching
    and estimates the parameters of the joint surplus
    in a `Choo and Siow 2006 <https://www.jstor.org/stable/10.1086/498585?seq=1>`_ model
    (homoskedastic, with singles)
"""
from math import pow

import numpy as np
import streamlit as st
from st_aggrid import GridOptionsBuilder
import pandas as pd

from cupid_matching.model_classes import ChooSiowPrimitives

from utils.auxiliaries import _get_aggrid_data, _print_matching


st.set_page_config(page_title="Solving for the stable matching")

st.markdown(
    """
#### This solves for the stable matching in a [Choo and Siow 2006](https://www.jstor.org/stable/10.1086/498585?seq=1) matching model with transferable utilities.

#### It relies on the IPFP algorithm in [Galichon and Salanié 2021a](https://academic.oup.com/restud/advance-article-abstract/doi/10.1093/restud/rdab090/6478301) and on the estimation methods in Galichon and Salanié (2021b).

#### See also the [cupid_matching](https://pypi.org/project/cupid_matching/) package.
"""
)

expander_bar = st.expander("More information")
expander_bar.markdown(
    """
The app lets you choose the total numbers of men and women in a marriage market; the number of types of each;
the proportions of men and women in each type; and the values of the joint surplus for each pair of types.
"""
)

expander_bar.markdown("It plots your chosen joint surplus matrix;")

expander_bar.markdown(
    """
then it solves for the large market equilibrium in a simulated Choo and Siow market.
"""
)


list_n_types = [2, 3, 4, 5]
st.sidebar.subheader("Choose the numbers of types of each gender")
n_types_men = st.sidebar.radio("Number of categories of men", list_n_types)
n_types_women = st.sidebar.radio("Number of categories of women", list_n_types)

st.markdown(
    """
Now choose the numbers of men and women  of each type.
"""
)


nmen_str = "Numbers of men of each type"
df_nmen = pd.DataFrame(np.ones(n_types_men, dtype=int), columns=[nmen_str])

gb_nmen = GridOptionsBuilder.from_dataframe(df_nmen)
gb_nmen.configure_default_column(editable=True)

nwomen_str = "Numbers of women of each type"
df_nwomen = pd.DataFrame(np.ones(n_types_women, dtype=int), columns=[nwomen_str])

gb_nwomen = GridOptionsBuilder.from_dataframe(df_nwomen)
gb_nwomen.configure_default_column(editable=True)

col_men, col_women = st.columns(2)
with col_men:
    result_nmen = _get_aggrid_data(df_nmen, gb_nmen)
with col_women:
    result_nwomen = _get_aggrid_data(df_nwomen, gb_nwomen)

husband_types = [f"Husband {x}" for x in range(1, n_types_men + 1)]
wife_types = [f"Wife {y}" for y in range(1, n_types_women + 1)]

st.markdown(
    """
Finally, choose the joint surplus of couples.
"""
)

df_phi = pd.DataFrame(
    np.zeros((n_types_men, n_types_women + 1)),
    columns=["Partner types"] + wife_types,
)
df_phi["Partner types"] = husband_types

gb_phi = GridOptionsBuilder.from_dataframe(df_phi)
gb_phi.configure_default_column(editable=True)

result_phi = _get_aggrid_data(df_phi, gb_phi)

n_men = np.zeros(n_types_men, int)
for x in range(n_types_men):
    n_men[x] = result_nmen[nmen_str][x]

n_women = np.zeros(n_types_women, int)
for y in range(n_types_women):
    n_women[y] = result_nwomen[nwomen_str][y]

Phi = np.zeros((n_types_men, n_types_women))
for x in range(n_types_men):
    for y in range(n_types_women):
        Phi[x, y] = result_phi[f"Wife {y + 1}"][x]

# st.markdown("Here is your joint surplus by categories:")
# st.altair_chart(_plot_heatmap(Phi))

if st.button("Get the stable matching"):
    st.write(
        "With unobserved heterogeneity, the stable matching patterns are not integers."
    )
    choo_siow_instance = ChooSiowPrimitives(Phi, n_men, n_women)
    matching = choo_siow_instance.ipfp_solve()
    _print_matching(matching)
