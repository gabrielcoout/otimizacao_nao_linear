# Dashboard de Montagem de Portfólio

Este projeto implementa um **dashboard interativo em Streamlit** para otimização de portfólio de investimentos.  
Ele coleta dados de ações do índice IBrX-50, calcula métricas financeiras (retorno médio, risco, covariância, volatilidade, correlação) e aplica métodos de otimização para sugerir a melhor alocação de ativos.  
O usuário pode selecionar ativos, definir capital e taxa livre de risco, e visualizar métricas, gráficos e pesos ótimos.

## Instalação

Clone o repositório e instale as dependências:

```bash
pip install -r requirements.txt
```

## Execução
Para rodar o dashboard, execute:
```bash
streamlit run main.py
```
O Streamlit abrirá automaticamente no navegador (por padrão em http://localhost:8501).

# Estrutura

main.py → ponto de entrada do dashboard

Dados/ → funções para coleta e processamento de dados

Otimizacao/ → métodos de otimização de portfólio

Visualizacoes/ → gráficos e plots

src/Methods/ -> backtracking, gradientedescendente, metodo de newton

data/ → cache de dados e metadados

analises/ -> programas python que executam as analises contidas no relatório



# Metodos em src/

Classe Abstrata:   src/Otimizacao

Classes Concretas: src/GradienteAscendenteProjetado

                   src/GradienteEspectralProjetado

                   src/BarreiraLogaritmicaPenalizacaoQuadratica

                   src/Analitico
                   


# Objetivo
O projeto facilita a montagem de portfólios otimizados com base em métricas de risco e retorno, permitindo comparar diferentes métodos de otimização não linear e visualizar resultados de forma clara e interativa.
