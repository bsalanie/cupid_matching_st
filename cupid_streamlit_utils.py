from math import pow
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from bs_python_utils.bsnputils import check_matrix, check_vector
from bs_python_utils.bsutils import bs_error_abort
from cupid_matching.matching_utils import Matching


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
    n_types = n1 * np.logspace(base=lambda_val, start=0, stop=ncat - 1, num=ncat)
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
    return df_coeffs_estimates


def _plot_heatmap(mat: np.ndarray, str_format: str, str_tit: str | None = None) -> alt.Chart:
    """Plots a heatmap of the matrix

    Args:
        mat: the matrix to plot
        str_tit: a title, if any

    Returns:
        the heatmap
    """
    ncat_men, ncat_women = check_matrix(mat)
    mat_arr = np.empty((mat.size, 4), dtype=int)
    imat = np.round(mat)
    mat_min = np.min(imat)
    i = 0
    for ix in range(ncat_men):
        for iy in range(ncat_women):
            m = imat[ix, iy]
            s = m - mat_min + 1
            mat_arr[i, :] = np.array([ix, iy, m, s])
            i += 1

    mat_df = pd.DataFrame(mat_arr, columns=["Men", "Women", "Value", "Size"])
    mat_df = mat_df.astype(
        dtype={"Men": int, "Women": int, "Value": int, "Size": float}
    )
    base = alt.Chart(mat_df).encode(x="Men:O", y=alt.Y("Women:O", sort="descending"))
    mat_map = base.mark_circle(opacity=0.4).encode(
        size=alt.Size("Size:Q", legend=None, scale=alt.Scale(range=[1000, 10000])),
        # color=alt.Color("Value:Q"),
        # tooltip=alt.Tooltip('Value', format=".2f")
    )
    text = base.mark_text(baseline="middle", fontSize=16).encode(
        text=alt.Text("Value:Q", format=str_format),
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
    ncat = check_vector(xvals)
    if str_gender not in ["men", "women"]:
        bs_error_abort(f"{str_gender} is not a valid side")
    str_cat = "x" if str_gender == "men" else "y"
    str_val = f"Single {str_gender}"
    color_bar = "pink" if str_gender == "women" else "lightblue"
    source = pd.DataFrame({str_cat: np.arange(1, ncat + 1, dtype=int), str_val: xvals})

    g_bars = (
        alt.Chart(source)
        .mark_bar(color=color_bar)
        .encode(y=str_cat + ":O", x=str_val + ":Q")
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
    plotxy = _plot_heatmap(muxy, "d", str_tit="Marriages")
    plotsingles = _plot_bars(mux0, mu0y)
    return plotxy | plotsingles


# Functions to download  a NumPy array as a CSV file
def _convert_dataframe_to_csv(df: pd.DataFrame) -> str:
    return df.to_csv().encode("utf-8")


def _convert_arr_to_csv(arr: np.ndarray) -> str:
    return _convert_dataframe_to_csv(pd.DataFrame(arr))


def _download_numpy_as_csv(arr: np.ndarray, file_name: str):
    csv = _convert_arr_to_csv(arr)
    st.download_button(
        label=f"Download the {file_name} as a CSV file",
        data=csv,
        file_name=f"{file_name}.csv",
        mime="text/csv",
    )


def _download_dataframe_as_csv(df: pd.DataFrame, file_name: str):
    csv = _convert_dataframe_to_csv(df)
    st.download_button(
        label=f"Download the {file_name} as a CSV file",
        data=csv,
        file_name=f"{file_name}.csv",
        mime="text/csv",
    )

def _download_parameters_results(file_name: str, do_estimates: bool,
                                 pars_res: list):
    if do_estimates:
        n_households, ncat_men, ncat_women, proportion_men, \
            scenario_men, scenario_women, true_coeffs, coeff_names, \
            muxy_sim, mux0_sim, mu0y_sim, n_sim, m_sim, \
            df_mde, mde_test_results, \
            df_poisson, df_u_estimates, df_v_estimates = pars_res
        results_str = f"You chose a model with {ncat_men} types of men and {ncat_women} types of women,\n"
        results_str += f"   with the number of men {scenario_men} and a number of wwomen {scenario_women},\n"
        results_str += f"    and {n_households} households with a proportion of {100*proportion_men:.1f} percent of men.\n\n"
        results_str += f"The numbers of men in each category are:\n"
        for x in range(ncat_men):
            results_str += f"{x+1}: {n_sim[x]: d}\n"
        results_str += f"\n  the numbers of women in each category are:\n"
        for y in range(ncat_women):
            results_str += f"{y+1}: {m_sim[y]: d}\n"
        results_str += "\nYou chose the following coefficients:\n"
        for coeff, value in zip(coeff_names, true_coeffs, strict=True):
            results_str += f"{coeff}:  {value: >10.3f}\n"
        results_str += "\n\n"
        results_str += "This gives the following numbers of marriages:\n"
        for x in range(ncat_men):
            for y in range(ncat_women):
                results_str += f"{muxy_sim[x, y]: d}   "
            results_str += "\n"
        results_str += "\n\n"
        results_str += "The numbers of single men are:\n"
        for x in range(ncat_men):
            results_str += f"{x+1}: {mux0_sim[x]: d}\n"
        results_str += "\n\n"
        results_str += "The numbers of single women are:\n"
        for y in range(ncat_women):
            results_str += f"{y+1}: {mu0y_sim[y]: d}\n"
        results_str += f"\n\n Minimum distance estimation gives\n"
        results_str += df_mde.to_string()
        specif_test_ndf, specif_test_stat, specif_test_pval = mde_test_results
        results_str += f"\n\nSpecification test statistic: chi2({specif_test_ndf}) = {specif_test_stat}\n"
        results_str += f"     with  p-value {specif_test_pval}\n\n"
        results_str += f"\n\n Poisson-GLM estimation gives\n"
        results_str += df_poisson.to_string()
        results_str += f"\n\n  the expected utilities of men are:\n"
        results_str += df_u_estimates.to_string()
        results_str += f"\n\n  the expected utilities of women are:\n"
        results_str += df_v_estimates.to_string()
    else:
        ncat_men, ncat_women, n_sim, m_sim, Phi, muxy_sim, mux0_sim, mu0y_sim \
              = pars_res
        results_str=f"You chose a model with {ncat_men} types of men and {ncat_women} types of women;\n"
        results_str += f"\n  the numbers of men in each category are:\n"
        for x in range(ncat_men):
            results_str += f"{x+1}: {n_sim[x]: d}\n"
        results_str += f"\n  the numbers of women in each category are:\n"
        for y in range(ncat_women):
            results_str += f"{y+1}: {m_sim[y]: d}\n"
        results_str += "\nYou chose the following joint surplus matrix:\n"
        for x in range(ncat_men):
            for y in range(ncat_women):
                results_str += f"{Phi[x, y]: 10.2f} "
            results_str += "\n"
        results_str += "\n\n"
        results_str += "This gives the following numbers of marriages:\n"
        for x in range(ncat_men):
            for y in range(ncat_women):
                results_str += f"{muxy_sim[x, y]: d}   "
            results_str += "\n"
        results_str += "\n\n"
        results_str += "The numbers of single men are:\n"
        for x in range(ncat_men):
            results_str += f"{x+1}: {mux0_sim[x]: d}\n"
        results_str += "\n\n"
        results_str += "The numbers of single women are:\n"
        for y in range(ncat_women):
            results_str += f"{y+1}: {mu0y_sim[y]: d}\n"
        
    st.download_button(
        label=f"Download summary to {file_name}",
        data=results_str,
        file_name=f"{file_name}",
        mime="text/csv",
    )

