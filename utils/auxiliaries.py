import sys
from math import pow
from typing import Optional, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder


def _get_aggrid_data(df: pd.DataFrame, gb: GridOptionsBuilder) -> pd.DataFrame:
    response = AgGrid(
        df,
        gridOptions=gb.build(),
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
    )
    return response["data"]


def _make_margins(n: int, n_types: int, scenario: str = "Constant") -> np.ndarray:
    """create constant or decreasing or increasing margins

    Args:
        n: number of individuals
        ncat: number of types
        scenario: profile of numbers per type, defaults to "Constant"

    Returns:
        an `n_types`-vector of numbers per type
    """
    nx_constant = n / n_types
    if scenario == "Constant":
        nx = np.full(n_types, nx_constant)
    elif scenario == "Increasing":
        lambda_vals = pow(2.0, 1.0 / (n_types - 1))
        n1 = n * (lambda_vals - 1.0) / (pow(lambda_vals, n_types) - 1.0)
        nx = n1 * np.logspace(base=lambda_vals, start=0, stop=n_types - 1, num=n_types)
    elif scenario == "Decreasing":
        lambda_vals = pow(2.0, -1.0 / (n_types - 1))
        n1 = n * (lambda_vals - 1.0) / (pow(lambda_vals, n_types) - 1.0)
        nx = n1 * np.logspace(base=lambda_vals, start=0, stop=n_types - 1, num=n_types)
    return nx


def _table_estimates(
    coeff_names: List,
    true_coeffs: np.ndarray,
    estimates: np.ndarray,
    stderrs: np.ndarray,
):
    """Creates a table with estimates and standard errors

    Args:
        coeff_names: list of coefficient names
        true_coeffs: true values of coefficients
        estimates: estimated values of coefficients
        stderrs: standard errors of coefficients
    """
    st.write("The coefficients are:")
    df_coeffs_estimates = pd.DataFrame(
        {"True": true_coeffs, "Estimated": estimates, "Standard errors": stderrs},
        index=coeff_names,
    )
    return st.table(df_coeffs_estimates)


def _plot_heatmap(mat: np.ndarray, str_tit: Optional[str] = None):
    """Plots a heatmap of a matrix

    Args:
        mat: a matrix
        str_title: title of the plot, defaults to None
    """
    ncat_men, ncat_women = mat.shape
    mat_arr = np.empty((mat.size, 4))
    mat_min = np.min(mat)
    i = 0
    for ix in range(ncat_men):
        for iy in range(ncat_women):
            m = mat[ix, iy]
            s = m - mat_min + 1
            mat_arr[i, :] = np.array([ix + 1, iy + 1, m, s])
            i += 1

    mat_df = pd.DataFrame(mat_arr, columns=["Men", "Women", "Value", "Size"])
    mat_df = mat_df.astype(
        dtype={"Men": int, "Women": int, "Value": float, "Size": float}
    )
    base = alt.Chart(mat_df).encode(x="Men:O", y=alt.Y("Women:O", sort="descending"))
    mat_map = base.mark_circle(opacity=0.4).encode(
        size=alt.Size("Size:Q", legend=None, scale=alt.Scale(range=[1000, 10000])),
        color=alt.Color("Value:Q"),
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


def _gender_bars(xvals: np.ndarray, str_gender: str):
    """Plots a bar chart of values by types for a given gender

    Args:
        xvals: values by types
        str_gender: "men: or "women""
    """
    ncat = xvals.size
    match str_gender:
        case "men":
            str_cat = "x"
        case "women":
            str_cat = "y"
        case _:
            print(f"str_cat cannot be {str_cat}")
            sys.exit(1)

    str_cat = "x" if str_gender == "men" else "y"
    str_val = f"Single {str_gender}"
    source = pd.DataFrame({str_cat: np.arange(1, ncat + 1), str_val: xvals})

    g_bars = (
        alt.Chart(source)
        .mark_bar()
        .encode(x=alt.X(str_cat, axis=alt.Axis(tickCount=ncat, grid=False)), y=str_val)
    )
    return g_bars.properties(width=300, height=300)


def _plot_bars(mux0: np.ndarray, mu0y: np.ndarray):
    men_bars = _gender_bars(mux0, "men")
    women_bars = _gender_bars(mu0y, "women")
    return (men_bars & women_bars).properties(title="Singles")


def _plot_matching(mus: Tuple[np.ndarray, np.ndarray, np.ndarray]):
    """Plots the stable matching patterns

    Args:
        mus: a tuple of three arrays (marriages, single men, single women)
    """
    muxy, mux0, mu0y, _, _ = mus.unpack()
    plotxy = _plot_heatmap(muxy, str_tit="Marriages")
    plotsingles = _plot_bars(mux0, mu0y)
    return plotxy & plotsingles
