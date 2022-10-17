""" Solves for the stable matching
    and estimates the parameters of a joint surplus function
    in a `Choo and Siow 2006 <https://www.jstor.org/stable/10.1086/498585?seq=1>`_ model
    (homoskedastic, with singles)
"""
from math import pow

import numpy as np
import pandas as pd
import streamlit as st

from cupid_matching.utils import nprepeat_col, nprepeat_row
from cupid_matching.choo_siow import entropy_choo_siow
from cupid_matching.min_distance import estimate_semilinear_mde
from cupid_matching.model_classes import ChooSiowPrimitives
from cupid_matching.poisson_glm import choo_siow_poisson_glm

from utils.auxiliaries import (
    _make_margins,
    _table_estimates,
    _print_surplus,
    _print_matching,
)


st.set_page_config(page_title="Simulating and estimating the Choo and Siow model")

description = (
    "#### This solves for the stable matching in, and estimates the parameters of, "
    + "a [Choo and Siow 2006](https://www.jstor.org/stable/10.1086/498585?seq=1) matching model with"
    + "with transferable utilities.\n"
    + "#### It relies on the IPFP algorithm in "
    + "[Galichon and Salanié 2021a](https://academic.oup.com/restud/advance-article-abstract/doi/10.1093/restud/rdab090/6478301)"
    + " and on the estimation methods in Galichon and Salanié (2022).\n"
    + "#### See also the [cupid_matching](https://pypi.org/project/cupid_matching/) package."
)

st.markdown(description)


expander_bar = st.expander("More information")
expander_bar.markdown(
    """
The app lets you choose the total numbers of men and women in a marriage market; the number of types of each;
the proportions of men and women in each type; and the parameters of a quadratic joint surplus function:
"""
)
expander_bar.latex(
    r"""
\Phi_{xy}=c_0+c_1 x + c_2 y + c_3 x^2 + c_4 x y + c_5 y^2
"""
)

expander_bar.markdown("It plots the resulting joint surplus matrix")

expander_bar.markdown(
    """
Then it solves for the large market equilibrium in a simulated Choo and Siow market,
and it fits the simulated data using the two estimators in Galichon-Salanié (2021b):

a minimum distance estimator and a Poisson GLM estimator.
"""
)

list_n_households = [1000, 10000, 100000]
st.sidebar.subheader("First, choose the total number of households")
n_households = st.sidebar.radio("Number of households", list_n_households)

list_n_types = [5, 10]
st.sidebar.subheader("Now, the numbers of types of each gender")
n_types_men = st.sidebar.radio("Number of categories of men", list_n_types)
n_types_women = st.sidebar.radio("Number of categories of women", list_n_types)

# nx = np.zeros(ncat_men)
# my = np.zeros(ncat_women)
# st.subheader("Second, choose the numbers of men and women in each category")
# for iman in range(ncat_men):
#     nx[iman] = st.slider(f"Number of men in category {iman+1}",
#                          min_value=1, max_value=10, step=1)
# for iwoman in range(ncat_women):
#     my[iwoman] = st.slider(f"Number of women in category {iwoman+1}",
#                            min_value=1, max_value=10, step=1)
#
st.sidebar.markdown(
    """
By default there are as many men as women.
You can also change the proportion.
"""
)
proportion_men = st.sidebar.slider(
    "Proportion of men", min_value=0.05, max_value=0.95, value=0.5
)


st.sidebar.markdown(
    """
By default each category within a gender has the same number of individuals.
You can also have the number increase by a factor two across categories, or decrease.
"""
)

list_scenarii = ["Constant", "Increasing", "Decreasing"]
scenario_men = st.sidebar.radio("Profile across categories for men", list_scenarii)
scenario_women = st.sidebar.radio("Profile across categories for women", list_scenarii)

nx = _make_margins(proportion_men, n_types_men, scenario_men)
my = _make_margins(1.0 - proportion_men, n_types_women, scenario_women)


st.sidebar.write("Finally, choose the coefficients of the 6 basis functions")
st.sidebar.latex(
    r"""
\begin{align*}
\Phi_{xy} &= c_0+c_1 x + c_2 y + c_3 x^2 \\
    &+ c_4 x y + c_5 y^2
\end{align*}
"""
)
min_c = np.array([-3.0] + [-2.0 / n_types_men] * 5)
max_c = np.array([3.0] + [2.0 / n_types_women] * 5)
true_coeffs = np.zeros(6)
coeff_names = [f"c[{i}]" for i in range(6)]


if "randoms" not in st.session_state:
    random_coeffs = np.round(min_c + (max_c - min_c) * np.random.rand(6), 2)
    st.session_state.randoms = random_coeffs

random_coeffs = st.session_state["randoms"]
for i in range(6):
    val_i = float(random_coeffs[i])
    true_coeffs[i] = st.sidebar.slider(
        coeff_names[i], min_value=min_c[i], max_value=max_c[i], value=val_i
    )

xvals = np.arange(n_types_men) + 1
yvals = np.arange(n_types_women) + 1

bases = np.zeros((n_types_men, n_types_women, 6))
bases[:, :, 0] = 1.0
xvals_mat = nprepeat_col(xvals, n_types_women)
yvals_mat = nprepeat_row(yvals, n_types_men)
bases[:, :, 1] = xvals_mat
bases[:, :, 2] = yvals_mat
bases[:, :, 3] = xvals_mat * xvals_mat
bases[:, :, 4] = np.outer(xvals, yvals)
bases[:, :, 5] = yvals_mat * yvals_mat

Phi = bases @ true_coeffs
st.markdown("Here is your joint surplus by types:")
_print_surplus(Phi)

cs_market = ChooSiowPrimitives(Phi, nx, my)

st.subheader(
    f"Here are the stable matching patterns in a sample of {n_households} households:"
)

mus_sim = cs_market.simulate(n_households)
muxy_sim, mux0_sim, mu0y_sim, n_sim, m_sim = mus_sim.unpack()
_print_matching(mus_sim)


st.subheader("Estimating the parameters.")

if st.button("Estimate"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "###### Below: the minimum distance estimator in Galichon and Salanié (2021b)."
        )
        mde_results = estimate_semilinear_mde(mus_sim, bases, entropy_choo_siow)
        mde_estimates = mde_results.estimated_coefficients
        mde_stderrs = mde_results.stderrs_coefficients

        _table_estimates(coeff_names, true_coeffs, mde_estimates, mde_stderrs)

        specif_test_stat = round(mde_results.test_statistic, 2)
        specif_test_pval = round(mde_results.test_pvalue, 2)
        st.markdown("###### It also gives us a specification test.")
        st.write(
            f"Test statistic: chi2({mde_results.ndf}) = {specif_test_stat} has p-value {specif_test_pval}"
        )

    with col2:
        st.markdown(
            "###### Below: the Poisson GLM estimator in Galichon and Salanié (2021b)."
        )
        pglm_results = choo_siow_poisson_glm(mus_sim, bases)

        pglm_estimates = pglm_results.estimated_beta
        pglm_stderrs = pglm_results.stderrs_beta

        _table_estimates(coeff_names, true_coeffs, pglm_estimates, pglm_stderrs)

        st.markdown(
            "###### The Poisson estimator also gives us the estimates of the expected utilities $u_x$ and $v_y$."
        )
        u = pglm_results.estimated_u
        v = pglm_results.estimated_v
        x_names = [f"Men {x}" for x in range(1, n_types_men + 1)]
        y_names = [f"Women {y}" for y in range(1, n_types_women + 1)]

        st.write("The expected utilities are:")
        df_u_estimates = pd.DataFrame(
            {"Estimated": u, "True": -np.log(mux0_sim / n_sim)}, index=x_names
        )
        st.table(df_u_estimates)
        df_v_estimates = pd.DataFrame(
            {"Estimated": v, "True": -np.log(mu0y_sim / m_sim)}, index=y_names
        )
        st.table(df_v_estimates)
