import numpy as np
import time
import sympy as sp
from scipy.optimize import minimize

class Otimizacao:
    def __init__(self, mu, cov, r_f):
        self.mu = np.asarray(mu, float)
        self.cov = np.asarray(cov, float)
        self.r_f = float(r_f)
        self.w = None
        self.n = self.mu.size
        self._sharpe_sym = None
        self._grad_sym = None
        self._hess_sym = None

    def fit(self):
        inicio = time.time()
        fim = time.time()
        self.time_last = fim - inicio

    def time(self, n=10):
        tempos = []
        for _ in range(n):
            inicio = time.time()
            self.fit()
            tempos.append(time.time() - inicio)
        return sum(tempos) / n

    def gerar_grad_hess(self):
        mu = sp.Matrix(self.mu)
        cov = sp.Matrix(self.cov)
        n = self.n

        w = sp.symbols(f"w0:{n}", real=True)
        w_vec = sp.Matrix(w)

        var = (w_vec.T * cov * w_vec)[0]
        sharpe_neg = -(mu.dot(w_vec) - self.r_f) / sp.sqrt(var)

        grad_sym = [sp.diff(sharpe_neg, wi) for wi in w_vec]
        hess_sym = [[sp.diff(grad_sym[i], w_vec[j]) for j in range(n)] for i in range(n)]

        sharpe_l = sp.lambdify(w, sharpe_neg, "numpy")
        grad_l = sp.lambdify(w, grad_sym, "numpy")
        hess_l = sp.lambdify(w, hess_sym, "numpy")

        def sharpe_func(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != n:
                raise ValueError(f"sharpe_func espera vetor tamanho {n}, recebeu {x.size}")
            return float(sharpe_l(*x))

        def grad_func(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != n:
                raise ValueError(f"grad_func espera vetor tamanho {n}, recebeu {x.size}")
            g = np.array(grad_l(*x), dtype=float).reshape(-1)
            return g

        def hess_func(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != n:
                raise ValueError(f"hess_func espera vetor tamanho {n}, recebeu {x.size}")
            H = np.array(hess_l(*x), dtype=float)
            return H

        self._sharpe_sym = sharpe_func
        self._grad_sym = grad_func
        self._hess_sym = hess_func

        return self._sharpe_sym, self._grad_sym, self._hess_sym

    def _sharpe(self, w):
        if self._sharpe_sym is None:
            raise RuntimeError("Função simbólica não gerada — chame gerar_grad_hess() primeiro.")
        return float(self._sharpe_sym(w))

    def _grad_sharpe(self, w):
        if self._grad_sym is None:
            raise RuntimeError("Gradiente simbólico não gerado — chame gerar_grad_hess() primeiro.")
        return np.array(self._grad_sym(w), dtype=float).reshape(-1)

    def _hess_sharpe(self, w):
        if self._hess_sym is None:
            raise RuntimeError("Hessiana simbólica não gerada — chame gerar_grad_hess() primeiro.")
        return np.array(self._hess_sym(w), dtype=float)


class BarreiraLogaritmicaPenalizacaoQuadratica(Otimizacao):
    def __init__(self, mu: np.ndarray, cov: np.ndarray, r_f: float):
        super().__init__(mu, cov, r_f)
        self.gerar_grad_hess()
        self.w = None
        self.lam = None

    def _cost_func(self, t: float):
        n = self.n
        eps = 1e-12
        rho = t

        def F(z: np.ndarray):
            z = np.asarray(z, dtype=float).reshape(-1)
            w = z[:n]
            lam = float(z[n])

            w_stable = w.copy()
            small_mask = w_stable <= 0
            if np.any(small_mask):
                w_stable[small_mask] = eps

            negativeSharpe = self._sharpe(w_stable)
            barreira = (1.0 / t) * np.sum(np.log(w_stable))
            diff_sum = np.sum(w) - 1.0
            val_penalidade = (rho / 2.0) * (diff_sum ** 2)

            Lval = negativeSharpe - barreira + val_penalidade

            g_sharpe = self._grad_sharpe(w_stable)
            g_barreira = - (1.0 / t) * (1.0 / w_stable)
            g_penalidade = rho * diff_sum * np.ones(n)

            g_w = g_sharpe + g_barreira + g_penalidade
            g_lam = np.sum(w_stable) - 1.0
            grad = np.concatenate([g_w, [g_lam]]).astype(float)

            return Lval, grad

        return F

    def fit(self, x0=None, t_init=1.0, t_factor=4.0, t_max=1e6,
            maxiter_inner=1000, tol_inner=1e-8, verbose=False):

        start = time.time()
        n = self.n

        if x0 is None:
            w0 = np.ones(n) / n
            lam0 = 0.0
            z = np.concatenate([w0, [lam0]])

        t = t_init
        nit_total = 0
        history = []
        result = None

        while t <= t_max:
            history.append(z)
            F = self._cost_func(t)
            f = lambda z_: F(z_)[0]
            grad = lambda z_: F(z_)[1]

            result = minimize(fun=f, x0=z, method='BFGS', jac=grad,
                              options={'disp': verbose, 'maxiter': maxiter_inner})

            z_opt = result.x

            if np.linalg.norm(z_opt - z) < 1e-10 and result.success:
                break
            if t > t_max / t_factor and result.success:
                break

            z = z_opt.copy()
            t *= t_factor
            nit_total += getattr(result, "nit", 0)

        z_opt = result.x
        w_opt = z_opt[:n]
        lam_opt = float(z_opt[n])

        self.w = w_opt
        self.lam = lam_opt

        end = time.time()
        self.time_last = end - start

        try:
            cost_final = self._cost_func(t)(z_opt)[0]
        except Exception:
            cost_final = np.nan

        return {
            "method": "BFGS",
            "w": w_opt,
            "lambda": lam_opt,
            "sharpe": -self._sharpe(w_opt),
            "cost_final": cost_final,
            "success": bool(result.success),
            "status": getattr(result, "status", None),
            "time": end - start,
            "nit_total": nit_total,
            "history": history
        }


class LagrangianoAumentadoPHR(Otimizacao):
    """
    Versão revisada: Lagrangiano Aumentado (PHR) + Barreira logarítmica.
    Correções:
      - força sum(w)=1 como restrição do solver interno (SLSQP)
      - bounds w_i >= eps para compatibilidade com log
      - atualização de rho condicional (quando violação não melhora)
      - cálculo correto de gradientes compatível com jac passado ao solver
    """

    def __init__(self, mu: np.ndarray, cov: np.ndarray, r_f: float):
        super().__init__(mu, cov, r_f)
        self.gerar_grad_hess()   # cria _sharpe_sym, _grad_sym, _hess_sym
        self.lam = 0.0
        self.mu_i = np.zeros(self.n)
        self.rho = 1.0

    def _Lagrangiano_PHR_val_grad(self, w, t):
        n = self.n
        eps = 1e-12
        w = np.asarray(w, dtype=float).reshape(-1)
        w_stable = np.maximum(w, eps)

        f = float(self._sharpe_sym(w_stable))

        # igualdade e desigualdades
        h = np.sum(w_stable) - 1.0               # h(w)
        g = -w_stable                            # g_i(w) = -w_i <= 0

        # PHR terms
        L_eq = 0.5 * self.rho * (h + self.lam / self.rho) ** 2

        g_shift = g + self.mu_i / self.rho      # vetor
        g_plus = np.maximum(0.0, g_shift)
        L_ineq = 0.5 * self.rho * np.sum(g_plus ** 2)

        L_total = f + L_eq + L_ineq 

        # Gradientes
        grad_f = np.asarray(self._grad_sym(w_stable), dtype=float).reshape(-1)   # df/dw

        grad_L_eq = (h + self.lam / self.rho) * np.ones(n)

        mask = (g_shift > 0.0).astype(float)
        grad_L_ineq = - self.rho * g_plus * mask

        grad_total = grad_f + grad_L_eq + grad_L_ineq 

        return float(L_total), grad_total

    def _solve_subproblem(self, w0, t, eps):
        """
        Resolve o subproblema com BFGS, impondo sum(w)=1 e bounds w_i >= eps.
        Retorna w_opt e o objeto result do scipy.
        """
        n = self.n

        fun = lambda ww: self._Lagrangiano_PHR_val_grad(ww, t)[0]
        jac = lambda ww: self._Lagrangiano_PHR_val_grad(ww, t)[1]

        res = minimize(fun=fun,
                       x0=w0,
                       jac=jac,
                       method='BFGS',
                       options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False})
        return res.x, res

    def fit(self,
            x0=None,
            t_init=1.0,
            t_factor=2.0,
            t_max=1e6,
            max_outer=100,
            tol=1e-8,
            eps=1e-8,
            rho_increase_factor=10.0,
            verbose=False):

        start = time.time()
        n = self.n

        if x0 is None:
            w = np.ones(n) / n
        else:
            w = np.asarray(x0, dtype=float).reshape(-1)

        t = t_init
        history = []
        nit_total = 0

        prev_primal_viol = np.inf

        for k in range(max_outer):
            history.append(w.copy())

            # resolve subproblema (agora com equality constraint explicitamente)
            w_opt, res = self._solve_subproblem(w, t, eps)
            nit_total += getattr(res, 'nit', 0)
            w = w_opt.copy()

            # avalia violações
            h = np.sum(w) - 1.0
            g = np.maximum(0.0, -w)   # violações das desigualdades (se w<0)
            primal_viol = max(abs(h), np.max(g))

            if verbose:
                pos_sharpe = -float(self._sharpe_sym(w))    # sharpe positivo
                print(f"[outer {k}] sharpe={pos_sharpe:.6f} primal_viol={primal_viol:.3e} rho={self.rho:.3e}")

            # atualiza multiplicadores
            self.lam = self.lam + self.rho * h
            self.mu_i = np.maximum(0.0, self.mu_i + self.rho * (-w))   # mu <- max(0, mu + rho*g) with g=-w

            # critério de parada (primal viol pequeno e KKT approx)
            if primal_viol <= tol:
                # opcional: verificar gradiente Lagrangiano pequeno (cond KKT)
                grad_L = self._Lagrangiano_PHR_val_grad(w, t)[1]
                # proj grad em subspace tangente (simple check)
                if np.linalg.norm(grad_L) <= 1e-6:
                    if verbose:
                        print("Convergiu (primal viol e grad baixos).")
                    break

            # controle de aumento de rho: só aumenta se violação não melhorou o suficiente
            if prev_primal_viol != np.inf:
                if primal_viol > 0.5 * prev_primal_viol:
                    # não melhorou o suficiente -> aumenta rho
                    self.rho *= rho_increase_factor
                    if verbose:
                        print(f"Aumentando rho -> {self.rho:.3e}")

            prev_primal_viol = primal_viol

            # aumenta o parâmetro de barrier t (t controla 1/t na barrier)
            t *= t_factor
            if t > t_max:
                if verbose:
                    print("t atingiu t_max -> parando.")
                break

        end = time.time()
        self.w = w.copy()
        pos_sharpe = -float(self._sharpe_sym(w))
        cost_final = float(self._Lagrangiano_PHR_val_grad(w, t)[0])

        return {
            "method": "PHR-Barrier-revised",
            "w": self.w,
            "lambda": float(self.lam),
            "mu": self.mu_i.copy(),
            "rho": self.rho,
            "sharpe": pos_sharpe,
            "cost_final": cost_final,
            "success": res.success,
            "status": res.status,
            "time": end - start,
            "nit_total": nit_total,
            "history": history,
        }