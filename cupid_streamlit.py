from math import pow
from typing import Any, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from cupid_matching.choo_siow import entropy_choo_siow
from cupid_matching.matching_utils import Matching
from cupid_matching.min_distance import estimate_semilinear_mde
from cupid_matching.model_classes import ChooSiowPrimitives
from cupid_matching.poisson_glm import choo_siow_poisson_glm
from cupid_matching.utils import (
    bs_error_abort,
    nprepeat_col,
    nprepeat_row,
    test_matrix,
    test_vector,
)


def _make_profile(lambda_val: float, n: int, ncat: int) -> np.ndarray:
    """
    Creates a profile of numbers of individuals by type
    that changes exponentially

    Args:
        lambda_val: the exponent of the exponential
        n: the total number of individuals on that side of the market
        ncat: the number of types on that side of the market

    Returns:
        n_types: the number of individuals of each type
    """
    n1 = n * (lambda_val - 1.0) / (pow(lambda_val, ncat) - 1.0)
    n_types = n1 * np.logspace(
        base=lambda_val, start=0, stop=ncat - 1, num=ncat
    )
    return n_types


def _make_margins(n: int, ncat: int, scenario: str = "Constant") -> np.ndarray:
    """generates the numbers by type on one side of the market

    Args:
        n: the total number of individuals on that side of the market
        ncat: the number of types on that side of the market
        scenario: "Constant", "Increasing", or "Decreasing" numbers
        as a function of type

    Returns:
        the numbers by type on this side
    """
    n_constant = n / ncat
    if scenario == "Constant":
        n_types = np.full(ncat, n_constant)
        return n_types
    elif scenario == "Increasing":
        lambda_val = pow(2.0, 1.0 / (ncat - 1))
    elif scenario == "Decreasing":
        lambda_val = pow(2.0, 1.0 / (ncat - 1))
    else:
        bs_error_abort(f"Unknown scenario {scenario}")
    n_types = _make_profile(lambda_val, n, ncat)
    return n_types


def _table_estimates(
    coeff_names: list[str],
    true_coeffs: np.ndarray,
    estimates: np.ndarray,
    stderrs: np.ndarray,
) -> Any:
    """Creates a table of the estimates

    Args:
        coeff_names: the names of the coefficients
        true_coeffs: the true values of the coefficients
        estimates: the estimated values of the coefficients
        stderrs: the standard errors of the estimates

    Returns:
        the table
    """
    st.write("The coefficients are:")
    df_coeffs_estimates = pd.DataFrame(
        {
            "True": true_coeffs,
            "Estimated": estimates,
            "Standard errors": stderrs,
        },
        index=coeff_names,
    )
    return st.table(df_coeffs_estimates)


def _plot_heatmap(mat: np.ndarray, str_tit: Optional[str] = None) -> alt.Chart:
    """Plots a heatmap of the matrix

    Args:
        mat: the matrix to plot
        str_tit: a title, if any

    Returns:
        the heatmap
    """
    ncat_men, ncat_women = test_matrix(mat)
    mat_arr = np.empty((mat.size, 4))
    mat_min = np.min(mat)
    i = 0
    for ix in range(ncat_men):
        for iy in range(ncat_women):
            m = mat[ix, iy]
            s = m - mat_min + 1
            mat_arr[i, :] = np.array([ix, iy, m, s])
            i += 1

    mat_df = pd.DataFrame(mat_arr, columns=["Men", "Women", "Value", "Size"])
    mat_df = mat_df.astype(
        dtype={"Men": int, "Women": int, "Value": float, "Size": float}
    )
    base = alt.Chart(mat_df).encode(
        x="Men:O", y=alt.Y("Women:O", sort="descending")
    )
    mat_map = base.mark_circle(opacity=0.4).encode(
        size=alt.Size(
            "Size:Q", legend=None, scale=alt.Scale(range=[1000, 10000])
        ),
        # color=alt.Color("Value:Q"),
        # tooltip=alt.Tooltip('Value', format=".2f")
    )
    text = base.mark_text(baseline="middle", fontSize=16).encode(
        text=alt.Text("Value:Q", format=".2f"),
    )
    if str_tit is None:
        both = (mat_map + text).properties(width=500, height=500)
    else:
        both = (mat_map + text).properties(title=str_tit, width=400, height=400)
    return both


def _gender_singles(xvals: np.ndarray, str_gender: str) -> alt.Chart:
    """creates a histogram of singles for this side of the market

    Args:
        xvals: the numbers of singles by type
        str_gender: the side of the market, "men" or "women"

    Returns:
        the histogram
    """
    ncat = test_vector(xvals)
    if str_gender not in ["men", "women"]:
        bs_error_abort(f"{str_gender} is not a valid side")
    str_cat = "x" if str_gender == "men" else "y"
    str_val = f"Single {str_gender}"
    source = pd.DataFrame(
        {str_cat: np.arange(1, ncat + 1, dtype=int), str_val: xvals}
    )

    g_bars = (
        alt.Chart(source).mark_bar().encode(y=str_cat + ":O", x=str_val + ":Q")
    )
    return g_bars.properties(width=300, height=300)


def _plot_bars(mux0: np.ndarray, mu0y: np.ndarray) -> alt.Chart:
    """concatenates the two gender singles histograms

    Args:
        mux0: the numbers of single men by type
        mu0y: the numbers of single women by type

    Returns:
        the concatenated histogram
    """
    men_bars = _gender_singles(mux0, "men")
    women_bars = _gender_singles(mu0y, "women")
    return (men_bars & women_bars).properties(title="Singles")


def _plot_matching(mus: Matching) -> alt.Chart:
    """generates the complete plot of matching patterns

    Args:
        mus: the matching patterns

    Returns:
        the plot
    """
    muxy, mux0, mu0y, _, _ = mus.unpack()
    plotxy = _plot_heatmap(muxy, str_tit="Marriages")
    plotsingles = _plot_bars(mux0, mu0y)
    return plotxy | plotsingles


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
n_households = st.sidebar.radio("Number of households", list_nhh)

list_ncat = [2, 3, 5, 10]
st.sidebar.subheader("Now choose the numbers of types of each gender")
ncat_men = st.sidebar.radio("Number of categories of men", list_ncat)
ncat_women = st.sidebar.radio("Number of categories of women", list_ncat)

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
    scenario_men = st.sidebar.radio(
        "Profile across categories for men", list_scenarii
    )
    scenario_women = st.sidebar.radio(
        "Profile across categories for women", list_scenarii
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

    st.sidebar.write(
        "Finally, choose the coefficients of the 6 basis functions:"
    )
    st.sidebar.latex(
        r"\Phi_{xy}=c_0+c_1 x + c_2 y + c_3 x^2 + c_4 x y + c_5 y^2"
    )
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
        )
    for iwoman in range(ncat_women):
        my[iwoman] = st.slider(
            f"Number of women in category {iwoman+1}",
            min_value=1,
            max_value=10,
            step=1,
        )
    Phi = np.zeros((ncat_men, ncat_women))
    for iman in range(ncat_men):
        for iwoman in range(ncat_women):
            Phi[iman, iwoman] = st.slider(
                f"Joint surplus in cell ({iman+1}, {iwoman+1})",
                min_value=-10,
                max_value=10,
            )


st.subheader("Here is your joint surplus by categories:")
st.altair_chart(_plot_heatmap(Phi))

cs_market = ChooSiowPrimitives(Phi, nx, my)

st.subheader(
    f"Here are the stable matching patterns in a sample of {n_households} households:"
)

mus_sim = cs_market.simulate(n_households)
muxy_sim, mux0_sim, mu0y_sim, n_sim, m_sim = mus_sim.unpack()

st.altair_chart(_plot_matching(mus_sim))


if enough_cells:
    st.subheader(
        "Estimating the parameters of the quadratic joint surplus $\\mathbb{\\Phi}$"
    )

    if st.button("Estimate"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "#### Below: the minimum distance estimator in Galichon and Salanié (2023)."
            )
            st.write("It also gives us a specification test.")
            mde_results = estimate_semilinear_mde(
                mus_sim, bases, entropy_choo_siow
            )
            mde_estimates = mde_results.estimated_coefficients
            mde_stderrs = mde_results.stderrs_coefficients

            _table_estimates(
                coeff_names, true_coeffs, mde_estimates, mde_stderrs
            )

            specif_test_stat = round(mde_results.test_statistic, 2)
            specif_test_pval = round(mde_results.test_pvalue, 2)
            st.write(
                f"Test statistic: chi2({mde_results.ndf}) = {specif_test_stat} has p-value {specif_test_pval}"
            )

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
                "It also gives us the estimates of the expected utilities $u_x$ and $v_y$."
            )

            pglm_results = choo_siow_poisson_glm(mus_sim, bases)

            u = pglm_results.estimated_u
            v = pglm_results.estimated_v
            pglm_estimates = pglm_results.estimated_beta
            pglm_stderrs = pglm_results.stderrs_beta

            _table_estimates(
                coeff_names, true_coeffs, pglm_estimates, pglm_stderrs
            )

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
