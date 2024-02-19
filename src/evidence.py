import numpy as np
from scipy import optimize
from numpy.typing import NDArray
#from typing import TypeAlias
from dataclasses import dataclass

#FloatArray: TypeAlias = NDArray[np.float64]
FloatArray = NDArray[np.float64] # TypeAlias

from functools import cached_property as c_property

@dataclass
class SolveWHAM:
    beta_K: FloatArray
    E_NK: FloatArray
    n_bins: int = 1000

    @c_property
    def M(self) -> int:
        return self.n_bins
    
    @c_property
    def K(self) -> int:
        return self.beta_K.size

    @c_property
    def d(self) -> float:
        return (self.E_NK.max() - self.E_NK.min()) / self.M
    
    @c_property
    def E_M(self) -> FloatArray:
        return np.linspace(self.E_NK.min(), self.E_NK.max(), self.M)
    
    @c_property
    def E_edge_M(self) -> FloatArray:
        return np.linspace(self.E_NK.min() - self.d/2, self.E_NK.max() + self.d/2, self.M+1)
    
    @c_property
    def E_min(self) -> float:
        return self.E_M.min()

    @c_property
    def h_KM(self):
        return np.array([np.histogram(self.E_NK[k], bins=self.E_edge_M)[0] for k, _ in enumerate(self.beta_K)])
    
    @c_property
    def H_M(self) -> FloatArray:
        return self.h_KM.sum(axis=0)
    
    @c_property
    def N_K(self) -> FloatArray:
        return self.h_KM.sum(axis=1)
    
    def lAY_to_Ag(
            self,
            lAY_K: FloatArray) -> FloatArray:
        return (self.H_M / np.array([
            self.N_K[k] / np.exp(lAY_K[k]) * np.exp(-beta * (self.E_M - self.E_min))
            for k, beta in enumerate(self.beta_K)
            ]).sum(axis=0))

    def Ag_to_lAY(
            self,
            Ag_M: FloatArray) -> FloatArray:
        return np.array([
            (Ag_M * np.exp(-beta * (self.E_M - self.E_min))).sum()
            for k, beta in enumerate(self.beta_K)
            ])

    def Ag_to_lAZ(
            self,
            Ag_M: FloatArray) -> FloatArray:
        return np.array([
            -beta * self.E_min + np.log((Ag_M * np.exp(-beta * (self.E_M - self.E_min))).sum())
            for beta in self.beta_K])
    
    def lAY_diff(self, lAY_2K: FloatArray) -> FloatArray:
        lAY_1 = 0
        lAY_K: FloatArray = np.array([lAY_1, *lAY_2K])
        Ag_M = self.lAY_to_Ag(lAY_K)
        new_lAY_K = self.Ag_to_lAY(Ag_M)
        return new_lAY_K[1:] - lAY_K[1:]
    
    def __post_init__(self):
        result, self.res_cov = optimize.leastsq(self.lAY_diff, np.zeros(self.K - 1))
        self.lAY_K = np.array([0, *result])
        self.Ag_M = self.lAY_to_Ag(self.lAY_K)

        self.log_AZ_K = self.Ag_to_lAZ(self.Ag_M)


@dataclass
class SolveEvidence:
    beta_K: FloatArray
    E_NK: FloatArray
    W: int
    n_bins_wham: int = 1000

    @c_property
    def K(self) -> int:
        return self.beta_K.size

    @c_property
    def range_groups(self) -> list[tuple[int, int]]:
        return [(s,t) for s in range(0, self.K, self.W-1) if s+1!=(t:=min(s+self.W, self.K))]

    @c_property
    def beta_groups(self) -> list[FloatArray]:
        return [self.beta_K[s:t] for s,t in self.range_groups]
    
    def __post_init__(self):
        self.solvers: list[SolveWHAM] = []
        self.log_Z_groups: list[FloatArray] = []
        last_log_Z: float = 0
        for (s,t), beta_S in zip(self.range_groups, self.beta_groups):
            E_NS = self.E_NK[s:t]
            solver = SolveWHAM(beta_S, E_NS)
            log_AZ_S = solver.log_AZ_K
            log_Z_S: FloatArray = log_AZ_S - log_AZ_S[0] + last_log_Z
            last_log_Z = log_Z_S[-1]
            self.solvers.append(solver)
            self.log_Z_groups.append(log_Z_S)
        
        self.log_Z_K = np.array(sum([log_Z_S[:-1].tolist() for log_Z_S in self.log_Z_groups], []) + [self.log_Z_groups[-1][-1]])
        self.seams_beta = self.beta_K[[t for _,t in self.range_groups[:-1]]]
        self.seams_log_Z = self.log_Z_K[[t for _,t in self.range_groups[:-1]]]

def calc_evidence_bootstrap(
        beta_K: FloatArray,
        ll_NK: FloatArray,
        W: int,
        n_bins_wham: int=1000,
        n_bootstrap: int=10,
        random_state=None,
        verbose=False):
    if verbose:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x: x
    else:
        tqdm = lambda x: x

    rnd = np.random.RandomState(random_state)
    log_Z_KB: list[FloatArray] = []
    solvers: list[SolveEvidence] = []
    for r in tqdm(range(n_bootstrap)):
        E_boot_NK = np.array([rnd.choice(e_n, e_n.size, True) for e_n in -ll_NK])
        solver = SolveEvidence(beta_K, E_boot_NK, W, n_bins_wham)
        log_Z_KB.append(solver.log_Z_K)
        solvers.append(solver)
    
    return np.array(log_Z_KB), solvers

if __name__=="__main__":
    import matplotlib as mpl
    mpl.use("TkAgg")
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    #with open(r"C:\Git\film-theckness-estimation\estimator\sl_paper_PA_S36_3\loglikelihood_indices.txt", encoding="UTF-8") as file:
    #    beta_K: NDArray[np.float64] = np.array([float(line.split(", ")[0])for line in file.readlines()[3:]])
    #    np.save(r"C:\Git\film-theckness-estimation\estimator\sl_paper_PA_S36_3\beta_k.npy", beta_K)
    beta_K: NDArray[np.float64] = np.load(r"C:\Git\film-theckness-estimation\estimator\sl_paper_PA_S36_3\beta_k.npy")
    lls: NDArray[np.float64] = np.load(r"C:\Git\film-theckness-estimation\estimator\sl_paper_PA_S36_3\loglikelihood_n_k_r.npy")
    ll_n_k = lls[..., lls.shape[-1]//2:].transpose(1,0,2).reshape(beta_K.size, -1)

    log_Z_KB, solvers = calc_evidence_bootstrap(beta_K, ll_n_k, 6, verbose=True)
    means = log_Z_KB.mean(axis=0)
    errs = np.abs(np.percentile(log_Z_KB, [0, 100], axis=0) - log_Z_KB.mean(axis=0))

    plt.errorbar(
        beta_K, means, yerr=errs,
        fmt="o-", capsize=3, markersize=5)
    #plt.scatter(solver.seams_beta, solver.seams_log_Z, s=100, c="k", marker="x") # type: ignore
    plt.yscale("asinh") # type: ignore
    plt.xscale("log")
    plt.xlim(beta_K.min(), 1)
    plt.ylim(means[beta_K==1]*1.1, 0)
    plt.show()