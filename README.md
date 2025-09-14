# Projeto de Previsão de Vendas - Hackathon 2025

Este repositório contém a solução completa para o desafio de previsão de demanda de produtos, desenvolvido para o Hackathon 2025. O objetivo principal é prever a quantidade de vendas semanais para cada combinação de produto e ponto de venda (PDV), visando otimizar a gestão de estoque e a reposição de produtos.

O notebook anexo (`Desafio.ipynb`) implementa um pipeline de Machine Learning de ponta a ponta, desde o processamento de dados em larga escala até o treinamento, otimização e interpretabilidade de um modelo preditivo.

## Sumário

1.  [Objetivo do Projeto](#1-objetivo-do-projeto)
2.  [Estrutura do Pipeline](#2-estrutura-do-pipeline)
3.  [Detalhamento das Decisões Técnicas](#3-detalhamento-das-decisões-técnicas)
    -   [Tecnologias e Bibliotecas](#tecnologias-e-bibliotecas)
    -   [Processamento de Dados Distribuído com Dask](#processamento-de-dados-distribuído-com-dask)
    -   [Engenharia de Atributos (Feature Engineering)](#engenharia-de-atributos-feature-engineering)
    -   [Escolha do Modelo: LightGBM](#escolha-do-modelo-lightgbm)
    -   [Estratégia de Validação e Otimização](#estratégia-de-validação-e-otimização)
    -   [Interpretabilidade do Modelo com SHAP](#interpretabilidade-do-modelo-com-shap)
    -   [Geração de Previsões Iterativas](#geração-de-previsões-iterativas)
4.  [Análise de Resultados](#4-análise-de-resultados)
5.  [Como Executar o Projeto](#5-como-executar-o-projeto)
6.  [Estrutura do Arquivo Final](#6-estrutura-do-arquivo-final)

---

## 1. Objetivo do Projeto

O desafio consiste em prever a demanda (quantidade vendida) de diversos produtos em múltiplos pontos de venda para as cinco primeiras semanas de 2023. Uma previsão acurada é crucial para evitar rupturas de estoque (perda de vendas) e excesso de inventário (custos de armazenamento), otimizando assim a cadeia logística.

## 2. Estrutura do Pipeline

O notebook segue uma estrutura lógica e sequencial para garantir a reprodutibilidade e a clareza do processo:

1.  **Setup do Ambiente:** Instalação e importação das bibliotecas necessárias.
2.  **Carga e Consolidação dos Dados:** Montagem do Google Drive, descompressão dos arquivos `.zip` e leitura dos dados em formato Parquet de forma eficiente.
3.  **Merge dos Datasets:** Utilização do Dask para unir os diferentes datasets (vendas, informações de produtos e de PDVs) em um único DataFrame consolidado, sem sobrecarregar a memória RAM.
4.  **Limpeza e Pré-processamento:** Tratamento de valores nulos, remoção de dados inconsistentes (ex: vendas com quantidade negativa) e otimização dos tipos de dados para reduzir o consumo de memória.
5.  **Engenharia de Atributos:** Criação de variáveis preditivas (features) a partir dos dados brutos, como features de data, sazonais, de lag e médias móveis.
6.  **Divisão para Treino e Validação:** Separação dos dados em conjuntos de treino e validação utilizando uma abordagem cronológica, fundamental para problemas de séries temporais.
7.  **Otimização de Hiperparâmetros:** Uso da biblioteca **Optuna** para encontrar a melhor combinação de parâmetros para o modelo, maximizando sua performance.
8.  **Treinamento do Modelo Final:** Treinamento de um modelo LightGBM com os melhores parâmetros encontrados, utilizando o conjunto de dados completo.
9.  **Análise e Interpretabilidade:** Utilização da biblioteca **SHAP** para entender quais variáveis mais influenciam as previsões do modelo.
10. **Geração das Previsões:** Implementação de uma lógica de previsão iterativa para estimar as vendas das cinco semanas de janeiro de 2023.
11. **Formatação da Saída:** Geração do arquivo `previsao_final_hackathon.csv` no formato exigido para submissão.

## 3. Detalhamento das Decisões Técnicas

Cada etapa do projeto foi guiada por decisões técnicas que visam eficiência, performance e robustez.

### Tecnologias e Bibliotecas

-   **Dask:** Escolhido para manipulação de dados que potencialmente não caberiam na memória RAM. Sua capacidade de computação paralela e "lazy evaluation" é ideal para o merge e processamento inicial dos arquivos Parquet.
-   **Pandas:** Utilizado para a manipulação final dos dados após a consolidação e redução do volume.
-   **LightGBM:** Selecionado como o algoritmo de modelagem. É um modelo de Gradient Boosting extremamente rápido e eficiente, que lida nativamente com variáveis categóricas e entrega alta performance em dados tabulares.
-   **Optuna:** Empregado para a otimização de hiperparâmetros. É uma ferramenta moderna e mais eficiente que métodos tradicionais como Grid Search, permitindo encontrar melhores modelos em menos tempo.
-   **SHAP (SHapley Additive exPlanations):** Utilizado para a interpretabilidade do modelo. Essencial para entender o comportamento da "caixa-preta" e extrair insights de negócio, justificando as previsões.

### Processamento de Dados Distribuído com Dask

A primeira decisão crítica foi como lidar com múltiplos arquivos Parquet de grande volume. Em vez de carregar tudo em memória com Pandas — o que poderia causar um crash no ambiente —, optou-se por usar Dask. Isso permitiu que as operações de `read_parquet` e `merge` fossem planejadas de forma "preguiçosa" (lazy) e executadas em chunks, garantindo que o processo fosse concluído mesmo em ambientes com memória limitada.

### Engenharia de Atributos (Feature Engineering)

A qualidade das features é o fator mais importante para o sucesso de um modelo preditivo. As principais decisões foram:

-   **Agregação Semanal:** O problema pede previsões semanais. Agregar os dados transacionais em uma base semanal (`df_weekly`) reduz o ruído, estabiliza o sinal e alinha os dados diretamente ao objetivo de negócio.
-   **Features de Lag e Médias Móveis:** A decisão mais impactante. A venda de uma semana (`vendas_semana_anterior`) é, quase sempre, o melhor preditor da venda da semana seguinte. Médias móveis (`media_movel_4s`) capturam tendências recentes e suavizam flutuações.
-   **Features Sazonais e de Eventos:** Vendas são influenciadas por sazonalidade. Foram criadas variáveis para identificar a `semana_do_ano`, `mes`, e eventos importantes como feriados (`is_holiday_week`) e semanas de pagamento (`is_payday_week`), que comprovadamente impactam o consumo.
-   **Features de Preço:** A variável `variacao_preco_pct` foi criada para que o modelo pudesse aprender sobre a elasticidade do preço, ou seja, como as vendas reagem a mudanças no valor do produto.

### Escolha do Modelo: LightGBM

O LightGBM foi escolhido por três motivos principais:
1.  **Performance:** É significativamente mais rápido que outras implementações de Gradient Boosting.
2.  **Acurácia:** Consistentemente apresenta resultados de ponta em competições e problemas com dados tabulares.
3.  **Flexibilidade:** Lida bem com um grande número de features e trata variáveis categóricas de forma otimizada, sem a necessidade de one-hot encoding extensivo.

### Estratégia de Validação e Otimização

-   **Validação Cronológica:** Para um problema de previsão, é fundamental validar o modelo em dados do "futuro" (período mais recente). Uma divisão aleatória causaria *data leakage* (vazamento de dados), resultando em uma métrica de performance irrealista. Por isso, as últimas 8 semanas de 2022 foram separadas como conjunto de validação.
-   **Métrica de Otimização (WMAPE):** O **Weighted Mean Absolute Percentage Error** foi escolhido como a métrica para o Optuna. Diferente do erro percentual simples, o WMAPE pondera os erros pelo volume de vendas, dando mais importância aos produtos de maior giro, que são os mais críticos para o negócio.
-   **Otimização com Optuna:** Em vez de um "chute" manual de parâmetros ou um Grid Search custoso, o Optuna busca inteligentemente o espaço de hiperparâmetros, convergindo para uma solução de alta qualidade de forma muito mais eficiente.

### Interpretabilidade do Modelo com SHAP

Para garantir que o modelo não era apenas acurado, mas também fazia sentido do ponto de vista de negócio, a análise SHAP foi realizada. Ela permitiu visualizar a importância de cada feature e entender sua direção de impacto, confirmando que o modelo aprendeu padrões lógicos e relevantes.

### Geração de Previsões Iterativas

Para prever as 5 semanas de janeiro de 2023, não se pode simplesmente criar as features de lag de uma vez, pois os dados reais dessas semanas são desconhecidos. A solução implementada foi um **loop iterativo**:
1.  Prevê a semana 1.
2.  Adiciona a previsão da semana 1 ao histórico de dados.
3.  Usa esse histórico atualizado para criar as features de lag para a semana 2.
4.  Prevê a semana 2, e assim por diante.

Essa abordagem simula um cenário real e é a forma correta de gerar previsões multi-step para modelos que dependem de features de lag.

## 4. Análise de Resultados

-   **Performance do Modelo:** O modelo final, treinado com os parâmetros otimizados pelo Optuna, atingiu um **WMAPE de Validação de 38.32%**. Este valor, que representa uma estimativa realista da performance em dados novos, é considerado competitivo para um problema complexo de previsão de demanda com milhares de séries temporais distintas.
-   **Importância das Features (SHAP):** A análise de interpretabilidade revelou que as variáveis mais importantes para o modelo foram:
    1.  `pdv` (identificador do ponto de venda)
    2.  `produto` (identificador do produto)
    3.  `vendas_semana_anterior` (vendas da semana anterior - lag 1)
    4.  `media_movel_4s` (tendência recente de vendas)
    5.  `preco_medio_anterior` (preço praticado)

Isso confirma que o modelo aprendeu padrões lógicos: a demanda depende fortemente do local e do item, além de ser influenciada pelo histórico recente de vendas e pelo preço.

## 5. Como Executar o Projeto

1.  **Ambiente:** O notebook foi projetado para ser executado no Google Colab com um ambiente de GPU.
2.  **Dados:** Certifique-se de que o arquivo `hackathon_2025_templates.zip` esteja na pasta `/content/drive/MyDrive/desafio/` do seu Google Drive.
3.  **Execução:** Execute as células do notebook em ordem sequencial. O processo completo, incluindo a otimização com Optuna, pode levar várias horas.
4.  **Saída:** Ao final da execução, o arquivo `previsao_final_hackathon.csv` será gerado e salvo na pasta `/content/drive/MyDrive/Desafio/`.

## 6. Estrutura do Arquivo Final

O arquivo de submissão `previsao_final_hackathon.csv` contém as seguintes colunas:
-   `semana`: Semana do ano (1 a 5 para janeiro de 2023).
-   `pdv`: Identificador do ponto de venda.
-   `produto`: Identificador do produto.
-   `quantidade`: Previsão da quantidade de vendas (arredondada para o inteiro mais próximo).

