import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from src.Dados import get_dados, get_metricas, obter_ibxl, obter_selic_atual
from src.Visualizacoes import graficoAcoes, graficoRetorno, graficoVolatilidade, graficoRAcumulado, plot_correlacao
from src.Otimizacao import BarreiraLogaritmicaPenalizacaoQuadratica



CSV_PATH = 'data/ibxl.csv'
META_PATH = 'data/ibxl_meta.json'

def pipeline_dados(ibv_path, meta_path):
    obter_ibxl()
    df_ibov = pd.read_csv(ibv_path, index_col=0)
    ibv_50 = list(df_ibov.index)
    return ibv_50

ibv_50 = pipeline_dados(CSV_PATH, META_PATH)

# metricas = get_metricas(dados)
# covariancia = metricas['covariancia']
# retorno_medio = metricas['retorno_medio']
# n_ativos = len(retorno_medio)


metodos = {
    "BarreiraLogaritmicaPenalizacaoQuadratica": BarreiraLogaritmicaPenalizacaoQuadratica,
    "GradienteAscendente": "GradienteAscendente"}

def Home():
    st.set_page_config(
        page_title="Dashboard de montagem de portfólio",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📊 Portfólio de Investimento Otimizado")
    st.markdown("---")

    with st.container(horizontal=True, horizontal_alignment="center"):
        selecionadas = st.pills(
            "Selecione os ativos:",
            options=ibv_50,
            selection_mode="multi",
            default=["WEGE3", "MULT3", "SUZB3"] 
        )

    if not selecionadas:
        st.info("Por favor, selecione um ou mais ativos acima para carregar os dados.")
    else:
        dados = get_dados(selecionadas)
        info_ativos = get_metricas(dados)
        with st.form(key='meu_formulario'):
            col1,col2, col3 = st.columns(3)
            with col1:
                r_f = st.number_input("Taxa livre de Risco", value=0.05, step=0.05, format="%.3f")
            with col2:
                capital = st.number_input("Qual valor deseja investir? (Ex: 1000.00)", min_value = 0.0)
            with col3:
                metodo_escolhido = st.selectbox(
                    "Método de otimização:",
                    options=list(metodos.keys())
                )
            submit_button = st.form_submit_button(label='Rodar')
            
        st.markdown("---")

        if submit_button:
            pass
            # col1, col2= st.columns(2)
            # with col1:
            #     ax = graficoAcoes(dados)
            #     st.pyplot(ax.get_figure())
            # with col2:
            #     ax2 = graficoVolatilidade(info_ativos)
            #     st.pyplot(ax2.get_figure())
                
                
            # col3,col4 = st.columns(2)
            # with col3:
            #     ax3 = graficoRAcumulado(info_ativos)
            #     st.pyplot(ax3.get_figure())
            # with col4:
            #     ax4 = graficoRetorno(info_ativos)
                # st.pyplot(ax4.get_figure())
            
            with st.container(border = True):
                st.write("Método de minimização do risco:")
                metricas = get_metricas(dados)
                # r_f = obter_selic_atual()
                cov = metricas['covariancia'].values
                mu = metricas['retorno_medio'].values

                mu = mu.flatten()
                cov = cov

                otimizador = metodos[metodo_escolhido](mu, cov, r_f)

                solucao = otimizador.fit()
                pesos = solucao['w']
                retorno = mu @ pesos 
                risco = np.sqrt(pesos.T @ cov @ pesos)
                sharpe = solucao['sharpe']

                investimento = pesos*capital
                dict_pesos = {'Ativos': selecionadas, 'Pesos': pesos, 'Investimento (R$)': investimento}
                col1,col2 = st.columns(2)
                col2.metric(label = "Ganho",
                            value = f'{retorno*capital:.2f} R$',
                            border = True)
                col1.metric(label = 'Montante final',
                            value = f'{capital*(1+retorno):.2f} R$',
                            border = True)
            

                col1,col2,col3 = st.columns(3)
                col1.metric(label = "Retorno Anualizado",
                            value = f'{retorno:.3f}',
                            border = True)
                col2.metric(label = 'Risco',
                            value = f'{risco:.3f}',
                            border = True)
                col3.metric(label = "Índice Sharpe Anualizado",
                            value = f'{sharpe:.3f}',
                            border = True)
                st.dataframe(dict_pesos)
                st.write("Distribuição de pesos:")
                st.bar_chart(data=dict_pesos, x='Ativos', y='Pesos')