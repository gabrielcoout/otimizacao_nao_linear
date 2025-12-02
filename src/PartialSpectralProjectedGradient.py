import numpy as np
import time
from src.Otimizacao import Otimizacao
from src.Methods import busca_linear_backtracking


def proj_simplex(v):
    # Projeta w no simplex: w>=0, sum w = 1
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w


class ProjectedSpectralProjectedGradient(Otimizacao):
    """
    Implementação reduzida do PSPG para o caso do simplex.
    """

    def __init__(self, mu, cov, r_f, delta_min=1e-4):
        super().__init__(mu, cov, r_f)
        self.delta_min = delta_min
        self.gerar_grad_hess()

    # -------------------------------------------------------------------------------------
    # PROJEÇÃO PARCIAL do artigo: Ω(x, δ)
    # Para simplex, isso significa elevar artificialmente um piso mínimo nos w_i
    # -------------------------------------------------------------------------------------
    def proj_parcial(self, x, delta):
        # Para o simplex, "Ω(x, δ)" = { w>=0 , sum(w)=1, w_i >= -δ }
        # Mas como w>=0 no simplex, escolhemos piso = -δ => efeito: nenhum.
        # Então implementamos a versão realmente útil:
        # w_i >= -δ  → torna a projeção menos restritiva.
        # Projeto via shift + simplex projection.
        return proj_simplex(np.maximum(x, -delta))

    # -------------------------------------------
    # Gp(x, δ, σ) = P_{Ω(x,δ)} ( x - σ∇f(x) ) - x
    # -------------------------------------------
    def gP(self, x, delta, sigma):
        y = x - sigma * self._grad_sharpe(x)
        p = self.proj_parcial(y, delta)
        return p - x

    # -------------------------------------------------------------------------------------
    # Internal step (algoritmo 3.3) reduzido para o simplex
    # -------------------------------------------------------------------------------------
    def internal_step(self, x, alpha=1e-2):
        # Step "unconstrained": x - α∇f
        d = -self._grad_sharpe(x)
        x_trial = x + alpha * d

        # Se caiu fora, projetar
        x_proj = proj_simplex(x_trial)

        # Monotonicidade (obrigatório no artigo)
        if self._sharpe(x_proj) <= self._sharpe(x):
            return x_proj
        else:
            # backtracking interno
            t = alpha
            for _ in range(20):
                t *= 0.5
                x_trial = x + t * d
                x_proj = proj_simplex(x_trial)
                if self._sharpe(x_proj) <= self._sharpe(x):
                    return x_proj
            return x  # fallback

    # -------------------------------------------------------------------------------------
    # FIT: PSPG
    # -------------------------------------------------------------------------------------
    def fit(self, x0=None, max_iter=5000, tol=1e-6, 
            alpha_armijo=1e-4, sigma_min=1e-10, sigma_max=1e10, 
            m_history=10, verbose=False):
        """
        m_history: int, número de iterações passadas para busca não-monótona (GLL).
                   Aumentar isso permite passos maiores.
        """
        start = time.time()
        n = self.n
        
        # Inicialização
        if x0 is None:
            x = np.ones(n) / n
        else:
            x = proj_simplex(x0)
            
        f = self._sharpe
        g = self._grad_sharpe
        
        delta = np.ones(n) * self.delta_min
        
        # Para busca não-monótona
        last_m_values = [f(x)]
        
        # Variáveis para o passo espectral
        x_prev = None
        g_prev = None
        
        history = [x.copy()]
        
        for k in range(max_iter):
            fx = f(x)
            gradx = g(x)
            
            # Critério de parada básico na norma do gradiente projetado (opcional aqui, 
            # pois verificamos d também)
            
            # ----- CÁLCULO DO PASSO ESPECTRAL (Barzilai-Borwein) -----
            if x_prev is None:
                # Inicialização do sigma pode ser mais inteligente baseada no gradiente
                # Mas 1.0 ou uma pequena fração costuma funcionar
                sigma = 1.0 
                # Opcional: sigma = min(1.0, 1.0 / np.linalg.norm(gradx))
            else:
                s = x - x_prev
                y = gradx - g_prev
                sy = np.dot(s, y)
                ss = np.dot(s, s)
                
                # Proteção numérica e escolha do BB step
                if sy <= 0:
                    # Se a curvatura for negativa (não-convexa), fallback para passo seguro
                    sigma = sigma_max # Ou mantenha o anterior
                else:
                    # Passo espectral BB1
                    sigma = np.clip(ss / sy, sigma_min, sigma_max)

            # ----- GERAÇÃO DA DIREÇÃO (Projeção Espectral) -----
            # d = P(x - sigma*g) - x
            # Aqui usamos sua função gP que lida com o delta adaptativo
            d = self.gP(x, delta, sigma)
            
            # Critério de parada: se a direção projetada é nula, convergiu
            if np.linalg.norm(d) < tol:
                if verbose: print(f"Convergiu na iteração {k} por norma(d)")
                break

            # ----- BUSCA LINEAR NÃO-MONÓTONA (GLL) -----
            # Ao invés de comparar com f(x), comparamos com o max dos últimos m valores
            f_ref = max(last_m_values)
            
            t = 1.0
            x_next = x
            
            # Loop de Backtracking
            while True:
                x_trial = x + t * d
                # Nota: gP já projeta parcialmente, mas para garantir viabilidade final
                # no Sharpe Ratio, garantimos que está no simplex aqui
                x_proj = proj_simplex(x_trial)
                
                if f(x_proj) <= f_ref + alpha_armijo * t * np.dot(gradx, d):
                    x_next = x_proj
                    break
                
                t *= 0.5
                if t < 1e-12:
                    # Se o passo for muito pequeno, aceitamos x_next como x (sem movimento)
                    # ou tentamos o internal_step como fallback de emergência
                    x_next = x
                    break
            
            # ----- ATUALIZAÇÃO -----
            # Se o passo t foi extremamente pequeno, aí sim tentamos o internal_step
            # ou aumentamos o delta (como no seu código original)
            if t < 1e-10:
                if verbose: print("Passo espectral falhou, tentando gradiente simples.")
                # Fallback simples: gradiente projetado sem sigma espectral
                x_fallback = self.internal_step(x, alpha=1e-2)
                if self._sharpe(x_fallback) < fx:
                    x_next = x_fallback
                else:
                    # Se nem o fallback ajudar, convergiu ou travou
                    break

            # Lógica adaptativa do Delta (do seu código original)
            if t < 0.1:
                delta *= 10
            else:
                # Opcional: relaxar o delta se o passo for bom, para não ficar enorme
                delta = np.maximum(delta * 0.5, self.delta_min)

            # Atualiza históricos
            x_prev = x.copy()
            g_prev = gradx.copy()
            x = x_next
            
            current_f = f(x)
            history.append(x.copy())
            
            # Atualiza memória não-monótona
            last_m_values.append(current_f)
            if len(last_m_values) > m_history:
                last_m_values.pop(0)

        end = time.time()

        sharpe_final = -self._sharpe(x)
        return {
            "method": "PSPG",
            "w": x,
            "sharpe": sharpe_final,
            "success": True,
            "time": end - start,
            "nit_total": k,
            "history": history
        }