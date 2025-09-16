# 🧠 Detector de Câncer Cerebral com CNN e PyTorch

## 📖 Descrição do Projeto

Este projeto é um aplicativo web desenvolvido com Streamlit que utiliza uma **Rede Neural Convolucional (CNN)**, construída com PyTorch, para classificar imagens de ressonância magnética (MRI) do cérebro. O modelo é capaz de identificar se uma imagem apresenta sinais de tumor cerebral ou se é saudável.

O objetivo principal é demonstrar um fluxo completo de um projeto de Deep Learning, desde o treinamento do modelo até sua implantação em uma interface interativa e informativa.

## ✨ Funcionalidades

-   **Upload de Imagens:** Permite que o usuário carregue uma imagem de MRI em formato JPG, JPEG ou PNG.
-   **Análise em Tempo Real:** O modelo processa a imagem e retorna o diagnóstico em poucos segundos.
-   **Interface Informativa:** Apresenta abas com a ferramenta de análise, métricas de performance detalhadas (Matriz de Confusão, Relatório de Classificação) e explicações sobre o comportamento do modelo.
-   **Design Responsivo:** A interface se adapta a diferentes tamanhos de tela, de desktops a celulares.

## 🛠️ Tecnologias Utilizadas

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red?style=for-the-badge&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## 🚀 Como Executar o Projeto

Disponível para uso em: https://cancerdetectionproject.streamlit.app/?embed_options=disable_scrolling,dark_theme

Caso queira executar o aplicativo localmente:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/jturini/brain-cancer-detector-cnn.git](https://github.com/SEUUSUARIO/brain-cancer-detector-cnn.git)
    cd brain-cancer-detector-cnn
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # No Windows
    .\venv\Scripts\activate
    # No macOS/Linux
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute o aplicativo Streamlit:**
    ```bash
    streamlit run app.py
    ```
    O aplicativo abrirá automaticamente no seu navegador.

## 📈 Performance do Modelo

O modelo foi avaliado em um conjunto de teste separado, alcançando uma **acurácia de 95%**. As métricas detalhadas abaixo mostram um modelo com altíssima sensibilidade (Recall) para detectar tumores, priorizando a segurança do diagnóstico. Mais informações disponíveis no site do projeto

##  dataset

A base de dados utilizada para o treinamento e teste do modelo foi o "Brain Cancer Detection MRI Images", disponível no Kaggle.
-   **Fonte:** [Link para o Dataset no Kaggle](https://www.kaggle.com/datasets/hamzahabib47/brain-cancer-detection-mri-images)

## ⚠️ Aviso Legal

Este projeto foi desenvolvido para fins educacionais e de demonstração. **Não deve ser utilizado como uma ferramenta de diagnóstico médico real.** Sempre consulte um profissional de saúde qualificado para obter diagnósticos e tratamentos.




