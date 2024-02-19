import numpy as np
from scipy import optimize
from numpy.typing import NDArray
from typing import TypeAlias, Tuple
from dataclasses import dataclass, astuple

FloatArray:TypeAlias = NDArray[np.float64]

@dataclass(frozen=True)
class SolveWHAMResult:
    E_M: FloatArray
    Ag_M: FloatArray
    log_AZ_K: FloatArray


def lAy_to_Ag(
        lAy_K: FloatArray,
        beta_K: FloatArray,
        N_K: FloatArray,
        H_M: FloatArray,
        E_M: FloatArray,
        E_min: float) -> FloatArray:
    return (H_M / np.array([N_K[k] / np.exp(lAy_K[k]) * np.exp(-beta * (E_M - E_min)) for k, beta in enumerate(beta_K)]).sum(axis=0))

def Ag_to_lAy(
        Ag_M: FloatArray,
        beta_K: FloatArray,
        E_M: FloatArray,
        E_min: float) -> FloatArray:
    return np.array([(Ag_M * np.exp(-beta * (E_M - E_min))).sum() for k, beta in enumerate(beta_K)])

def Ag_to_lAZ(
        Ag_M: FloatArray,
        beta_K: FloatArray,
        E_M: FloatArray,
        E_min: float) -> FloatArray:
    return np.array([
        -beta * E_min + np.log((Ag_M * np.exp(-beta * (E_M - E_min))).sum())
        for beta in beta_K])

def solve_wham(
        beta_K: FloatArray,
        E_NK: FloatArray,
        n_bins: int = 1000) -> SolveWHAMResult:
    """caluculate density of state by WHAM

    Parameters
    ----------
    beta_k : list[np.float64]
        list of inv. temperature
    E_n_k : list[list[np.float64]]
        list of energy for each beta
    bin_size : int
        size of bin to calc WHAM

    Returns
    -------
    E_m : list[np.float64]
        ヒストグラムの bin の中央値
    g_m : list[np.float64]
        状態密度
    """
    # binの切り方を決定
    M = n_bins
    K = beta_K.size
    #E_all = np.concatenate(E_n_k, axis=0)
    #_, E_bin = np.histogram(E_all, bins=bin_size)
    d: float = (E_NK.max() - E_NK.min()) / M
    E_M: FloatArray = np.linspace(E_NK.min(), E_NK.max(), M)
    E_edge_M: FloatArray = np.linspace(E_NK.min() - d/2, E_NK.max() + d/2, M+1)

    ## binの中央値
    #E_m = smooth(E_bin, 2)
    E_min: float = E_M.min()
    
    # 各 beta の E についてそれぞれのヒストグラム
    h_KM: FloatArray = np.array([np.histogram(E_NK[k], bins=E_edge_M)[0] for k, _ in enumerate(beta_K)])

    H_M: FloatArray = h_KM.sum(axis=0)
    N_K: FloatArray = h_KM.sum(axis=1)

    def lAY_diff(lAY_2K: FloatArray) -> FloatArray:
        # z_1=1に規格化して計算
        lAY_1 = 0
        lAY_K: FloatArray = np.array([lAY_1, *lAY_2K])
        Ag_M = lAy_to_Ag(lAY_K, beta_K, N_K, H_M, E_M, E_min)
        new_lAY_K = Ag_to_lAy(Ag_M, beta_K, E_M, E_min)
        return new_lAY_K[1:] - lAY_K[1:]

    result, err = optimize.leastsq(lAY_diff, np.zeros(K-1))
    lAy_K = np.array([1, *result])
    Ag_M = lAy_to_Ag(lAy_K, beta_K, N_K, H_M, E_M, E_min)
    log_AZ_K = Ag_to_lAZ(Ag_M, beta_K, E_M, E_min)
    return SolveWHAMResult(E_M, Ag_M, log_AZ_K) # type: ignore


import matplotlib.pyplot as plt

def solve_split_evidence(
        beta_K: FloatArray,
        E_NK: FloatArray,
        W: int,
        n_bins_wham: int = 1000
        ):
    ranges = [(s,t) for s in range(0, beta_K.size, W-1) if s+1!=(t:=min(s+W, beta_K.size))]
    beta_groups = [beta_K[s:t] for s,t in ranges]

    log_Z_groups: list[FloatArray] = []
    last_log_Z: float = 0
    log_z_1 = None
    for (s,t), beta_S in zip(ranges, beta_groups):
        #beta_S = beta_K[s:t]
        E_NS = E_NK[s:t]
        E_Ms, Ag_M, log_AZ_S = astuple(solve_wham(beta_S, E_NS, n_bins_wham))
        log_Z_S: FloatArray = log_AZ_S - log_AZ_S[0] + last_log_Z
        last_log_Z = log_Z_S[-1]
        log_Z_groups.append(log_Z_S)
    
    log_Z_K = np.array(sum([log_Z_S[:-1].tolist() for log_Z_S in log_Z_groups], []) + [log_Z_groups[-1][-1]])
    seam_beta = beta_K[[t for _,t in ranges[:-1]]]
    seam_log_Z = log_Z_K[[t for _,t in ranges[:-1]]]

    plt.plot(beta_K, log_Z_K, "o-")
    plt.scatter(seam_beta, seam_log_Z, s=100, c="k", marker="x") # type: ignore
    if False:
        for b,z in zip(beta_groups, log_Z_groups):
            plt.plot(b, z, "o-")
    plt.yscale("asinh") # type: ignore
    plt.xscale("log")
    plt.xlim(beta_K.min(), 1)
    plt.ylim(None, 0)

