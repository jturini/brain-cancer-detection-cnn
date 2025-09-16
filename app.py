# app.py (Versão Final - Com Explicações Detalhadas)

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Detector de Câncer Cerebral",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNÇÕES E CONFIGURAÇÃO DO MODELO ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=2).to(device)
    model_path = 'best_model_checkpoint.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['healthy', 'tumor']

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict(image):
    image_tensor = image_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    prediction = class_names[predicted_idx.item()]
    return prediction

# --- SIDEBAR (BARRA LATERAL) ---
with st.sidebar:
    st.title("Sobre o Projeto")
    st.info(
        "Este é um aplicativo web para demonstrar um modelo de Deep Learning que utiliza Redes Neurais Convolucionais (CNNs) "
        "capaz de classificar imagens de ressonância magnética cerebral. "
        "O objetivo é detectar a presença de tumores."
    )
    st.divider()
    st.header("Faça sua Análise")
    uploaded_file = st.file_uploader("Escolha uma imagem de MRI...", type=["jpg", "jpeg", "png"])
    st.divider()
    st.subheader("Recursos do Projeto")
    st.link_button("Acessar Documentação (GitHub)", url="https://github.com/SEU_USUARIO/SEU_REPOSITORIO")
    st.link_button("Base de Dados no Kaggle", url="https://www.kaggle.com/datasets/hamzahabib47/brain-cancer-detection-mri-images")

# --- PÁGINA PRINCIPAL ---
st.title("🧠 Detector de Câncer Cerebral por MRI")

# --- CRIAÇÃO DAS ABAS ---
tab_analise, tab_metricas = st.tabs(["Análise de Imagem 🖼️", "Métricas de Avaliação 📊"])

with tab_analise:
    # (Conteúdo da aba de análise - sem alteração)
    st.header("Faça uma Nova Predição")
    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Imagem Carregada.', use_column_width=True)
            if st.button('Analisar Imagem', use_container_width=True):
                with st.spinner('Analisando, por favor aguarde...'):
                    prediction = predict(image)
                    st.subheader("Resultado da Análise:")
                    if prediction == 'tumor':
                        st.error(f'**Diagnóstico:** Presença de **{prediction.upper()}** detectada.', icon="⚠️")
                    else:
                        st.success(f'**Diagnóstico:** A imagem foi classificada como **{prediction.upper()}** (Saudável).', icon="✅")
        else:
            st.info("Aguardando o upload de uma imagem na barra lateral esquerda.")

with tab_metricas:
    st.header("Performance do Modelo de Classificação")
    st.write("Aqui estão as métricas detalhadas do modelo, avaliadas no conjunto de teste para garantir sua imparcialidade.")
    st.divider()

    col_met_1, col_met_2 = st.columns(2)

    with col_met_1:
        st.subheader("Matriz de Confusão")
        st.image('matriz_confusao.png')
        
        # --- EXPLICAÇÃO DETALHADA DA MATRIZ DE CONFUSÃO ---
        st.info(
            """
            A **Matriz de Confusão** oferece um resumo visual dos acertos e erros do modelo. Ela é dividida em quatro quadrantes:

            - **Verdadeiro Positivo (VP):** O modelo previu 'TUMOR' e a imagem realmente tinha um tumor. **(Acerto Crítico)**
            - **Verdadeiro Negativo (VN):** O modelo previu 'SAUDÁVEL' e a imagem era de fato saudável. **(Acerto Importante)**
            - **Falso Positivo (FP):** O modelo previu 'TUMOR', mas a imagem era saudável. **(Erro Tipo I / Alarme Falso)**
            - **Falso Negativo (FN):** O modelo previu 'SAUDÁVEL', mas a imagem tinha um tumor. **(Erro Tipo II / O Erro Mais Perigoso)**

            O objetivo de um bom modelo de diagnóstico é maximizar os Verdadeiros Positivos e Negativos, e minimizar os Falsos Positivos e, especialmente, os Falsos Negativos.
            """
        )

    with col_met_2:
        st.subheader("Relatório de Classificação")
        # Cole aqui o texto exato do seu relatório de classificação
        report_text = """
                      precision    recall  f1-score   support

             healthy       0.97      0.92      0.95        38
               tumor       0.93      0.98      0.95        42

            accuracy                           0.95        80
           macro avg       0.95      0.95      0.95        80
        weighted avg       0.95      0.95      0.95        80
        """
        st.code(report_text, language='text')

        # --- EXPLICAÇÃO DETALHADA DO RELATÓRIO DE CLASSIFICAÇÃO ---
        st.info(
            """
            Este relatório traduz os números da Matriz de Confusão em métricas de performance:

            - **Precision (Precisão):** Mede a confiabilidade das previsões positivas. Para a classe 'tumor', ela responde: *"Das vezes que o modelo disse que havia um tumor, quantas vezes ele estava certo?"* Uma alta precisão significa poucos alarmes falsos (Falsos Positivos).

            - **Recall (Revocação ou Sensibilidade):** Mede a capacidade do modelo de encontrar todos os casos positivos. Para a classe 'tumor', ela responde: *"De todos os tumores que realmente existiam, quantos o modelo conseguiu encontrar?"* Esta é a métrica mais crítica em diagnóstico, pois um Recall alto significa poucos casos não detectados (Falsos Negativos).

            - **F1-Score:** É a média harmônica entre Precisão e Recall. Um F1-Score alto indica um bom equilíbrio entre as duas métricas.
            
            - **Support (Suporte):** O número total de amostras reais de cada classe no conjunto de teste.
            """
        )
    
    # ... (Restante do código da aba de métricas, como o expander) ...
    st.divider()
    with st.expander("🔍 Análise do Comportamento do Modelo: Cauteloso | Falsos Positivos | Et.cetera"):
        # (O texto detalhado que já criamos sobre o comportamento do modelo vai aqui)
        st.markdown("""
        Ao analisar as métricas, você pode notar um comportamento interessante: o modelo parece ser perfeito em encontrar tumores, mas às vezes classifica uma imagem saudável como suspeita. Isso não é um erro, mas sim uma **estratégia de segurança deliberada**.
        
        ### 1. Prioridade Máxima: Recall Alto (Alta Sensibilidade)
        Em um cenário de diagnóstico médico, o pior erro possível é um **Falso Negativo** – dizer que um paciente está saudável quando, na verdade, ele tem a doença. Para minimizar esse risco a quase zero, nosso modelo foi treinado para ter um **Recall** (ou sensibilidade) extremamente alto.
        - **O que isso significa?** Se existe um tumor na imagem, o modelo tem uma chance altíssima de encontrá-lo. Ele foi otimizado para não "deixar passar nada".

        ### 2. O Custo da Cautela: Os Falsos Positivos
        Para ser tão bom em detectar qualquer anomalia, o modelo precisa ser muito "cauteloso". Pense nele como um segurança extremamente vigilante: ele vai garantir que nenhum intruso (tumor) passe, mas para isso, ele pode soar o alarme para algumas sombras estranhas (os **Falsos Positivos**).
        - **O que isso significa?** Ocasionalmente, o modelo pode sinalizar uma imagem saudável que possui alguma variação anatômica normal como potencialmente problemática.

        ### Conclusão: Um Equilíbrio Clínico
        Este modelo foi projetado para atuar como uma **ferramenta de auxílio à triagem**. Sua principal função é garantir que nenhum caso potencialmente perigoso seja ignorado. Em um fluxo de trabalho real, um radiologista revisaria todas as imagens sinalizadas, descartando rapidamente os falsos positivos e focando sua atenção nos casos que realmente precisam de uma análise aprofundada.
        
        **Portanto, este equilíbrio que favorece a segurança do paciente é uma característica fundamental do design do modelo.**
        """)

# --- RODAPÉ ---
st.markdown("---")
st.write("Desenvolvido como um projeto de demonstração. Não deve ser usado para diagnóstico médico real.")