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
    with st.container():
        st.markdown("""
        ## Otimizadores:
        Este painel utiliza dois métodos de otimização de portfólio baseado na Teoria Moderna do Portfólio (Markowitz). 
        O objetivo principal do primeiro método é construir a carteira de Mínima Variância (menor risco) para um nível de retorno-alvo que você define.
        Enquanto que para o segundo método, o objetivo é ...

        ### Método 1: Uma Abordagem Híbrida
        O algoritmo opera em duas etapas para encontrar a melhor alocação de ativos:

        **1. A Solução Analítica (Lagrange)**
        
        Primeiro, o otimizador tenta encontrar a solução "matematicamente perfeita" usando um método clássico (Multiplicadores de Lagrange). Esta solução calcula a alocação de ativos que minimiza o risco para o seu retorno-alvo em um mundo ideal.

        **2. A Verificação de Realidade e Otimização Numérica**

        * **O Problema:** A solução "perfeita" muitas vezes inclui alocações irreais para um investidor comum (venda a descoberto com pesos negativos ou alavancagem com pesos > 100%).
        * **A Solução:** O código verifica se isso aconteceu. Se a solução matemática for irreal, ela é descartada.
        * **O Plano B (Gradient Descent):** O algoritmo ativa um otimizador numérico. Este método:
            * Começa com uma carteira simples (ex: pesos iguais).
            * Itera milhares de vezes, fazendo pequenos ajustes para minimizar o risco e, ao mesmo tempo, atingir o retorno-alvo.
            * Crucialmente, ele força que todos os pesos se mantenham entre 0% e 100% e que a soma total seja 100%.
        
        ### Método 2: 

        ---
        **O Resultado Final:** Você vê a carteira com o menor risco (volatilidade) possível para o retorno desejado, respeitando as restrições do mundo real (sem vender a descoberto).
        """)