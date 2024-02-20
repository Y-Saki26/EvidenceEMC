import numpy as np
from scipy import optimize
from numpy.typing import NDArray, ArrayLike
from typing import List, Tuple
from dataclasses import dataclass
import warnings
from functools import cached_property as c_property

#FloatArray: TypeAlias = NDArray[np.float64]
FloatArray = NDArray[np.float64] # TypeAlias

def poly_inverse(y_target: float, x: FloatArray, y: FloatArray) -> float:
    for i, xi in enumerate(x):
        if i+1==len(x): break
        xj, yi, yj = x[i+1], y[i], y[i+1]
        if yi==y_target: return xi
        if yi<y_target<=yj:
            s = (y_target - yi) / (yj - yi)
            return (1-s) * xi + s * xj
    raise ValueError("target value is out of the value range")


def pow_floor(x: ArrayLike, base: float=10) -> ArrayLike:
    return base ** np.floor(np.log(x) / np.log(base))


def interp_lin(x: FloatArray, k: int=2) -> FloatArray:
    new_x = []
    for i, xi in enumerate(x):
        if i+1==len(x):
            new_x.append(xi)
        else:
            new_x.extend(np.linspace(xi, x[i+1], k+1)[:-1])
    return np.array(new_x)


def splice_lin(x: FloatArray, k: int) -> FloatArray:
    d = x[1] - x[0]
    return np.array([x[0] + d * i for i in range(-k, 0)] + list(x))


def expand_betas(betas: FloatArray, k_interp: int=2, k_splice: int=4) -> FloatArray:
    return np.exp(splice_lin(interp_lin(np.log(betas), k_interp), k_splice))


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
            self.N_K[k] * np.exp(-lAY_K[k] -beta * (self.E_M - self.E_min))
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
        return new_lAY_K - lAY_K
    
    def __post_init__(self):
        result, self.res_cov = optimize.leastsq(self.lAY_diff, np.zeros(self.K - 1))
        self.lAY_K = np.array([0, *result])
        self.Ag_M = self.lAY_to_Ag(self.lAY_K)

        self.log_AZ_K = self.Ag_to_lAZ(self.Ag_M)

    def dist(self, beta: float) -> FloatArray:
        with np.errstate(divide="ignore"):
            logL_M = np.log(self.Ag_M) - beta * self.E_M
            midrange = (logL_M[np.isfinite(logL_M)].min() + logL_M[np.isfinite(logL_M)].max()) / 2
            L_M = np.exp(logL_M - midrange)
        return L_M / L_M.sum()


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
    def range_groups(self) -> List[Tuple[int, int]]:
        return [(s,t) for s in range(0, self.K, self.W-1) if s+1!=(t:=min(s+self.W, self.K))]

    @c_property
    def beta_groups(self) -> List[FloatArray]:
        return [self.beta_K[s:t] for s,t in self.range_groups]
    
    def __post_init__(self):
        self.solvers: List[SolveWHAM] = []
        self.log_Z_groups: List[FloatArray] = []
        last_log_Z: float = 0
        for (s,t), beta_S in zip(self.range_groups, self.beta_groups):
            E_NS = self.E_NK[s:t]
            solver = SolveWHAM(beta_S, E_NS, self.n_bins_wham)
            log_AZ_S = solver.log_AZ_K
            log_Z_S: FloatArray = log_AZ_S - log_AZ_S[0] + last_log_Z
            last_log_Z = log_Z_S[-1]
            self.solvers.append(solver)
            self.log_Z_groups.append(log_Z_S)
        
        self.log_Z_K = np.array(sum([log_Z_S[:-1].tolist() for log_Z_S in self.log_Z_groups], []) + [self.log_Z_groups[-1][-1]])
        self.seams_index = np.array([t for _,t in self.range_groups[:-1]])
        self.seams_beta = self.beta_K[self.seams_index]
        self.seams_log_Z = self.log_Z_K[self.seams_index]
    
    def pred_dist(self, new_betas) -> Tuple[List[FloatArray], List[FloatArray]]:
        new_beta_groups = []
        stat_groups = []
        for s, sub_solver in enumerate(self.solvers):
            if s==0:
                betas = new_betas[new_betas <= sub_solver.beta_K.max()]
            else:
                betas = new_betas[(new_betas >= sub_solver.beta_K.min()) & (new_betas <= sub_solver.beta_K.max())]
            stats: List = []
            for beta in betas:
                P_M = sub_solver.dist(beta)
                d = sub_solver.E_M[1] - sub_solver.E_M[0]
                wide_E_M = [sub_solver.E_M[0] - d, *sub_solver.E_M, sub_solver.E_M[-1] + d]
                wide_CP_M = [0, *np.cumsum(P_M), 1]
                stats.append([
                    *[poly_inverse(p, wide_E_M, wide_CP_M) for p in [0.25, 0.5, 0.75]], # type: ignore
                    (P_M * sub_solver.E_M).sum(),
                    ])
            new_beta_groups.append(betas)
            stat_groups.append(np.array(stats).T)
        return new_beta_groups, stat_groups


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
        except ModuleNotFoundError:
            warnings.warn("`tqdm` could not be imported. `verbose` is ignored.")
            tqdm = lambda x: x
    else:
        tqdm = lambda x: x

    rnd = np.random.RandomState(random_state)
    log_Z_KB: List[FloatArray] = []
    solvers: List[SolveEvidence] = []
    for r in tqdm(range(n_bootstrap)):
        E_boot_NK = np.array([rnd.choice(e_n, e_n.size, True) for e_n in -ll_NK])
        solver = SolveEvidence(beta_K, E_boot_NK, W, n_bins_wham)
        log_Z_KB.append(solver.log_Z_K)
        solvers.append(solver)
    
    return np.array(log_Z_KB), solvers

from multiprocessing import Pool


def _calc_evidence_sub(params: Tuple) -> FloatArray:    
    beta_K: FloatArray
    ll_NK: FloatArray
    W: int
    n_bins_wham: int
    beta_K, ll_NK, W, n_bins_wham, random_state = params
    rnd = np.random.RandomState(random_state)
    E_boot_NK = np.array([rnd.choice(e_n, e_n.size, True) for e_n in -ll_NK])
    solver = SolveEvidence(beta_K, E_boot_NK, W, n_bins_wham)
    return solver.log_Z_K


def calc_evidence_bootstrap_parallel(
        beta_K: FloatArray,
        ll_NK: FloatArray,
        W: int,
        n_bins_wham: int=1000,
        n_bootstrap: int=10,
        random_state=None,
        processes=None):

    rnd = np.random.RandomState(random_state)
    with Pool(processes) as p:
        log_Z_KB = p.map(_calc_evidence_sub, [
            (beta_K, ll_NK, W, n_bins_wham, None if random_state is None else hash((random_state, r)) % np.iinfo(int).max)
            for r in range(n_bootstrap)
        ])
    return np.array(log_Z_KB)

if __name__=="__main__":
    import matplotlib as mpl
    mpl.use("TkAgg")
    import matplotlib.pyplot as plt

    beta_K: NDArray[np.float64] = np.load(r"C:\Git\film-theckness-estimation\estimator\sl_paper_PA_S36_3\beta_k.npy")
    lls: NDArray[np.float64] = np.load(r"C:\Git\film-theckness-estimation\estimator\sl_paper_PA_S36_3\loglikelihood_n_k_r.npy")
    ll_n_k = lls[..., lls.shape[-1]//2:].transpose(1,0,2).reshape(beta_K.size, -1)

    log_Z_KB, solvers = calc_evidence_bootstrap(beta_K, ll_n_k, 6, verbose=True) # n_bootstrap=100: 35.26 sec
    # parallel processing
    #log_Z_KB = calc_evidence_bootstrap_parallel(beta_K, ll_n_k, 6, n_bootstrap=100) # 7.16 sec
    means = log_Z_KB.mean(axis=0)
    errs = np.abs(np.percentile(log_Z_KB, [25, 75], axis=0) - log_Z_KB.mean(axis=0))
    plt.errorbar(
        beta_K, means, yerr=errs,
        fmt="o-", capsize=3, markersize=5)
    plt.scatter(beta_K[solvers[0].seams_index], means[solvers[0].seams_index], s=50, marker="x") # type: ignore
    plt.yscale("asinh") # type: ignore
    plt.xscale("log")
    plt.xlim(beta_K.min(), 1)
    plt.ylim(means[beta_K==1]*1.1, 0)
    plt.show()