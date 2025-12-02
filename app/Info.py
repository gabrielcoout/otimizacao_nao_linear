import streamlit as st
import pandas as pd

CSV_PATH = 'data/ibxl.csv'
df_ibov = pd.read_csv(CSV_PATH, index_col=0)
ibv_50 = list(df_ibov.index)
nome_ibv_50 = list(df_ibov['acao'])

def Info():
    st.set_page_config(
        page_title="Informações",
        page_icon="❓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title(" ❓ Informações")
    st.markdown("---")
    
    st.markdown(" #### Empresas listadas")
    tabela_info = pd.DataFrame({'Tickers': ibv_50, "Ações": nome_ibv_50})
    st.dataframe(tabela_info, hide_index = True)
    st.markdown("""
## Resumo do Relatório – Otimização Não Linear de Portfólios

---

### Visão Geral
Este painel interativo aplica métodos de otimização não linear para construir carteiras eficientes de ações da B3 com base na Teoria Moderna do Portfólio.  
O objetivo é encontrar alocações que maximizem retorno ajustado ao risco, respeitando:

- soma dos pesos igual a 100%;
- proibição de venda a descoberto (wi ≥ 0);
- uso de dados históricos reais.

Os ativos disponíveis são obtidos automaticamente a partir do índice IBXL, e os preços são coletados via *yfinance*.

---

### Formulação do Problema
Para n ativos escolhidos:

- **Retornos médios anuais:** µ  
- **Covariância anualizada:** Σ  
- **Taxa livre de risco:** rf  
- **Pesos:** w

O retorno da carteira é  
**R(w) = µᵀ w**

e o risco (volatilidade):

**σ(w) = sqrt(wᵀ Σ w)**.

A métrica de desempenho utilizada é o **Índice de Sharpe**:

**S(w) = (µᵀ w – rf) / sqrt(wᵀ Σ w)**.

O problema é formulado como:

- minimizar –S(w)  
- sujeito a 1ᵀ w = 1 e w ≥ 0  

A região viável é um **simplexo**, onde os pesos são não-negativos e somam 1.

---

### Métodos de Otimização Implementados

#### 1. Maximização do Índice de Sharpe (Método Híbrido)
O método é dividido em duas etapas:

**(a) Solução Analítica (irrestrita)**  
Calcula-se o portfólio tangente fechado e verifica-se se os pesos pertencem ao intervalo [0,1].  
Se a solução for viável, ela é aceita.

**(b) Solução Numérica (Gradiente Ascendente)**  
Caso contrário, aplica-se um algoritmo iterativo com:
- atualização w ← w + η ∇S  
- projeção dos pesos no simplexo (clipping + normalização)  
- critério de parada pelo módulo da variação do Sharpe

Esse método se mostrou eficiente e estável.

---

#### 2. Método de Barreiras Logarítmicas com Penalidade Quadrática
A restrição wi ≥ 0 é tratada com a barreira:

**ϕ(w) = −∑ log(wi)**

e a soma 1ᵀ w = 1 é incorporada como penalidade:

**(t/2)(1ᵀ w − 1)²**

O parâmetro t cresce iterativamente seguindo uma estratégia de caminho central, e cada subproblema é resolvido por BFGS.

Apesar de correto teoricamente, o método apresentou **instabilidade numérica** quando alguns pesos se aproximaram de zero.

---

#### 3. Método de Restrições Ativas
Método alternativo no qual identificam-se pesos que ficam nulos na solução (restrições ativas).  
Resolve-se então o problema reduzido somente nas variáveis livres.  
Adequado quando a solução ótima apresenta muitos pesos zerados.

---

### Resultados Numéricos
- O método de barreiras tornou-se numericamente frágil devido à explosão do gradiente e da Hessiana próximos de **wᵢ → 0**.  
- O método híbrido (solução analítica + gradiente projetado) apresentou o melhor comportamento geral.  
- As carteiras obtidas são consistentes com a teoria de Markowitz e respeitam integralmente as restrições práticas do investidor.

---

### Organização do Código
O projeto é modular e dividido em:

- coleta e pré-processamento de dados  
- cálculo de métricas financeiras  
- métodos de otimização  
- interface Streamlit (página principal, informações e visualizações)

Essa estrutura facilita manutenção, testes e expansão do painel.

---
""")