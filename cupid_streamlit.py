from typing import cast

import numpy as np
import pandas as pd
import streamlit as st
from bs_python_utils.bsnputils import nprepeat_col, nprepeat_row
from cupid_matching.choo_siow import entropy_choo_siow
from cupid_matching.min_distance import estimate_semilinear_mde
from cupid_matching.model_classes import ChooSiowPrimitives
from cupid_matching.poisson_glm import choo_siow_poisson_glm

from cupid_streamlit_utils import (
    _download_parameters_results,
    _make_margins,
    _plot_heatmap,
    _plot_matching,
    _table_estimates,
)

summary_file_name = "summary.txt"   # simulation will be domwloaded there on request

st.title("Separable matching with transfers")

st.markdown(
    """
> This solves for equilibrium in a [Choo and Siow 2006](https://www.jstor.org/stable/10.1086/498585?seq=1) matching model
with perfectly transferable utility.

> If there are enough cells, it fits a quadratic specification of the joint surplus.

> The app relies on the IPFP algorithm in [Galichon and Salanié 2022](https://academic.oup.com/restud/article-abstract/89/5/2600/6478301) and on the estimation methods in Galichon and Salanié (2023),

> which use the [cupid_matching](https://pypi.org/project/cupid_matching/) package.
"""
)

expander_bar = st.expander("More information")
expander_bar.markdown(
    """
The app lets you choose
* the total numbers  of men and women in a marriage market; 
* the number of types $X$ and $Y$ of each;
* the proportions of men and women in each type, $n_x$ and $m_y$;
* and a joint surplus function $\\mathbb{\\Phi}$.

If $(X-1) \\times (Y-1) > 6$, you will be asked to specify the six coefficients $c_k$ of a quadratic specification for $\\mathbb{\\Phi}$:
"""
)
expander_bar.latex(r"\Phi_{xy}=c_0+c_1 x + c_2 y + c_3 x^2 + c_4 x y + c_5 y^2")
expander_bar.markdown("      and the profile of $n_x$ and $m_y$.")
expander_bar.markdown(
    """
If $(X-1) \\times (Y-1) \\leq 6$, there are not enough cells to estimate a quadratic specification;
you will be asked to input values of your choice for the $n_x, m_y$, and $\\Phi_{xy}$.
"""
)


expander_bar.markdown(
    "In all cases, the app plots the resulting joint surplus matrix $\\mathbb{\\Phi}$;"
)

expander_bar.markdown(
    """
Then it solves for the large market equilibrium in a simulated Choo and Siow market.

If $(X-1)\\times (Y-1) > 6$, it also fits the quadratic specification on the simulated equilibrium patterns using the two estimators in Galichon-Salanié (2023):

a minimum distance estimator and a Poisson GLM estimator.
"""
)

list_nhh = [10_000, 100_000, 1_000_000]
st.sidebar.subheader("First, choose the total number of households")
n_households = cast(int, st.sidebar.radio("Number of households", list_nhh))

list_ncat = [2, 3, 5, 10]
st.sidebar.subheader("Now choose the numbers of types of each gender")
ncat_men = cast(int, st.sidebar.radio("Number of categories of men", list_ncat))
ncat_women = cast(int, st.sidebar.radio("Number of categories of women", list_ncat))

enough_cells = (ncat_men - 1) * (ncat_women - 1) > 6

xvals = np.arange(ncat_men) + 1
yvals = np.arange(ncat_women) + 1

if enough_cells:
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
    scenario_men = cast(
        str, st.sidebar.radio("Profile across categories for men", list_scenarii)
    )
    scenario_women = cast(
        str, st.sidebar.radio("Profile across categories for women", list_scenarii)
    )

    nx = _make_margins(proportion_men, ncat_men, scenario_men)
    my = _make_margins(1.0 - proportion_men, ncat_women, scenario_women)

    bases = np.zeros((ncat_men, ncat_women, 6))
    bases[:, :, 0] = 1.0
    xvals_mat = nprepeat_col(xvals, ncat_women)
    yvals_mat = nprepeat_row(yvals, ncat_men)
    bases[:, :, 1] = xvals_mat
    bases[:, :, 2] = yvals_mat
    bases[:, :, 3] = xvals_mat * xvals_mat
    bases[:, :, 4] = np.outer(xvals, yvals)
    bases[:, :, 5] = yvals_mat * yvals_mat

    st.sidebar.write("Finally, choose the coefficients of the 6 basis functions:")
    st.sidebar.latex(r"\Phi_{xy}=c_0+c_1 x + c_2 y + c_3 x^2 + c_4 x y + c_5 y^2")
    min_c = np.array([-3.0] + [-2.0 / ncat_men] * 5)
    max_c = np.array([3.0] + [2.0 / ncat_women] * 5)
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

    Phi = bases @ true_coeffs
else:
    st.markdown(
        """
        There are not enough cells to estimate a quadratic specification of the joint surplus;
        you need to choose the values of $n_x, m_y$, and $\\Phi_{xy}$.
        """
    )
    nx = np.zeros(ncat_men)
    my = np.zeros(ncat_women)
    st.subheader("Choose the numbers of men and women in each category")
    for iman in range(ncat_men):
        nx[iman] = st.slider(
            f"Number of men in category {iman+1}",
            min_value=1,
            max_value=10,
            step=1,
            value=5
        )
    for iwoman in range(ncat_women):
        my[iwoman] = st.slider(
            f"Number of women in category {iwoman+1}",
            min_value=1,
            max_value=10,
            step=1,
            value=5
        )
    Phi = np.zeros((ncat_men, ncat_women))
    for iman in range(ncat_men):
        for iwoman in range(ncat_women):
            Phi[iman, iwoman] = st.slider(
                f"Joint surplus in cell ({iman+1}, {iwoman+1})",
                min_value=-10.0,
                max_value=10.0,
                value=0.0
            )


# if st.sidebar.button("Download Marriages"):
#     _download_file("marriages.csv")

# if st.sidebar.button("Download Singles"):
#     _download_file("singles.csv")

# if st.button("Download Minimum Distance Estimates"):
#     _download_file("min_distance_estimates.csv")

# if st.button("Download Poisson-GLM Estimates"):
#     _download_file("poisson_glm_estimates.csv")


st.subheader("Here is your joint surplus by categories:")
st.altair_chart(_plot_heatmap(Phi, ".2f"))

cs_market = ChooSiowPrimitives(Phi, nx, my)

st.subheader(
    f"Here are the stable matching patterns in a sample of {n_households} households:"
)

mus_sim = cs_market.simulate(n_households)
muxy_sim, mux0_sim, mu0y_sim, n_sim, m_sim = mus_sim.unpack()

st.altair_chart(_plot_matching(mus_sim))

do_estimates = False

if enough_cells:
    st.subheader(
        "Estimating the parameters of the quadratic joint surplus $\\mathbb{\\Phi}$"
    )

    if st.button("Estimate"):
        do_estimates = True
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "#### Below: the minimum distance estimator in Galichon and Salanié"
                " (2023)."
            )
            st.write("It also gives us a specification test.")
            mde_results = estimate_semilinear_mde(mus_sim, bases, entropy_choo_siow)
            mde_estimates = mde_results.estimated_coefficients
            mde_stderrs = mde_results.stderrs_coefficients

            df_mde = _table_estimates(coeff_names, true_coeffs, mde_estimates, mde_stderrs)
            st.table(df_mde)

            specif_test_stat = round(mde_results.test_statistic, 2)
            specif_test_pval = round(mde_results.test_pvalue, 2)
            st.write(
                f"Test statistic: chi2({mde_results.ndf}) = {specif_test_stat} has"
                f" p-value {specif_test_pval}"
            )
            mde_test_results = (mde_results.ndf, specif_test_stat, specif_test_pval)

        with col2:
            st.markdown(
                "#### Here is the Poisson GLM estimator in Galichon and Salanié (2023);"
            )
            st.markdown(
                """ 
                as it requires numerical optimization, it is a bit slower.
                """
            )
            st.write(
                "It also gives us the estimates of the expected utilities $u_x$ and"
                " $v_y$."
            )

            pglm_results = choo_siow_poisson_glm(mus_sim, bases)

            u = pglm_results.estimated_u
            v = pglm_results.estimated_v
            pglm_estimates = pglm_results.estimated_beta
            pglm_stderrs = pglm_results.stderrs_beta

            df_poisson = _table_estimates(coeff_names, true_coeffs, pglm_estimates, pglm_stderrs)
            st.table(df_poisson)

            x_names = [str(x) for x in range(ncat_men)]
            y_names = [str(y) for y in range(ncat_women)]

            st.write("The expected utilities are:")
            df_u_estimates = pd.DataFrame(
                {"Estimated": u, "True": -np.log(mux0_sim / n_sim)},
                index=x_names,
            )
            st.table(df_u_estimates)
            df_v_estimates = pd.DataFrame(
                {"Estimated": v, "True": -np.log(mu0y_sim / m_sim)},
                index=y_names,
            )
            st.table(df_v_estimates)
        pars_res = (n_households, ncat_men, ncat_women, proportion_men,
                scenario_men, scenario_women, true_coeffs, coeff_names, 
                muxy_sim, mux0_sim, mu0y_sim, n_sim, m_sim, 
                df_mde, mde_test_results,
                df_poisson, df_u_estimates, df_v_estimates)

        _download_parameters_results(summary_file_name, True, pars_res)
    else:
        pars_res = (ncat_men, ncat_women, n_sim, m_sim, Phi, 
                muxy_sim, mux0_sim, mu0y_sim)
        _download_parameters_results(summary_file_name, False, pars_res)    
else:
    pars_res = (ncat_men, ncat_women, n_sim, m_sim, Phi, 
                muxy_sim, mux0_sim, mu0y_sim)
    _download_parameters_results(summary_file_name, False, pars_res)    
