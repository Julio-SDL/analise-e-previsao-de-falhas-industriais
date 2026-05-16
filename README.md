# 🏭 Análise e Previsão de Falhas Industriais

Repositório desenvolvido durante o curso **Programação em Inteligência Artificial Generativa** no SENAI.

Documenta a jornada de aprendizado em IA e Machine Learning, desde os primeiros conceitos até um sistema completo de monitoramento industrial com previsão de falhas.

---

## 📚 Estudos

Projetos desenvolvidos ao longo do curso, na ordem em que foram aprendidos:

### 1. 🤖 Chatbot Gemini
Primeiro contato com IA generativa — chatbot interativo usando a API do Google Gemini.
- Integração com API externa via variável de ambiente
- Loop de conversa em tempo real com o modelo `gemini-2.5-flash-lite`

**Tecnologias:** Python · Google Generative AI · python-dotenv

---

### 2. 🌸 Classificação Iris
Primeiro modelo de Machine Learning do curso, usando o dataset clássico de flores Iris.
- Árvore de Decisão para classificar 3 espécies de flores
- Avaliação com acurácia, matriz de confusão e relatório de classificação

**Tecnologias:** Python · Scikit-learn

---

### 3. 🧹 Tratamento de Dados
Aprendizado de limpeza e preparação de dados reais com ruídos e inconsistências.
- Preenchimento de valores nulos com média e mediana
- Detecção e tratamento de outliers pelo método IQR
- Aplicado em dados de sensores industriais (temperatura, vibração, pressão)

**Tecnologias:** Python · Pandas

---

### 4. 🧠 Rede Neural com Scikit-learn
Construção de uma rede neural (MLP) para prever falhas industriais.
- Normalização dos dados com `StandardScaler`
- Classificador `MLPClassifier` com camadas ocultas
- Avaliação com acurácia e matriz de confusão

**Tecnologias:** Python · Scikit-learn · Pandas

---

### 5. ⚡ Rede Neural com TensorFlow
Evolução para deep learning com TensorFlow/Keras.
- Rede neural com camadas `Dense` e ativações `relu` e `sigmoid`
- Treinamento com validação e monitoramento por épocas
- Mesmo problema do módulo anterior, agora com framework profissional

**Tecnologias:** Python · TensorFlow · Keras · Scikit-learn

---

## 🚀 Projeto Final

Sistema completo de monitoramento e previsão de falhas em equipamentos industriais, unindo tudo que foi aprendido no curso.

### 🔍 Análise e Previsão de Falhas
Sistema interativo com menu que permite:
- Inserir dados dos sensores manualmente e receber previsão de falha em tempo real
- Consultar uma IA (Gemini) com perguntas sobre os dados da máquina
- Gerar relatório técnico completo via IA, salvo em arquivo `.txt`
- Relatório automático ao iniciar com ranking de variáveis críticas, alertas de risco combinado e recomendações de manutenção

O modelo Random Forest foi treinado com dados de sensores industriais (temperatura, pressão, vibração, tempo de operação) e features derivadas de engenharia de dados.

**Tecnologias:** Python · Scikit-learn · Pandas · Google Generative AI · python-dotenv

---

### 📊 Dashboard de Sensores Industriais
Dashboard visual de monitoramento industrial com 5 gráficos:
- Histórico de temperatura com falhas destacadas e limiar de atenção
- Importância dos sensores na previsão de falhas (Random Forest)
- Dispersão temperatura × pressão com zona de risco destacada
- Boxplot de vibração — Normal vs Falha
- Histograma de desgaste acumulado com percentil de alerta

Rodapé com estatísticas rápidas: total de registros, taxa de falhas, mediana dos sensores e variável mais crítica.

**Tecnologias:** Python · Matplotlib · Scikit-learn · Pandas · NumPy

---

## 🎁 Extra

### 👁️ Visão Computacional
Classificação de imagens com Rede Neural Convolucional (CNN) — desenvolvido após o projeto final como conteúdo adicional.
- Leitura e pré-processamento de imagens com OpenCV
- CNN com camadas convolucionais, pooling, flatten e densas
- Classificação binária: gato ou cão
- Acurácia de 100% no dataset de treino

**Tecnologias:** Python · TensorFlow · Keras · OpenCV · Scikit-learn

---

## 🛠️ Tecnologias utilizadas

Python · TensorFlow · Keras · Scikit-learn · Pandas · NumPy · Matplotlib · OpenCV · Google Generative AI · python-dotenv

---

## 🔑 Variáveis de ambiente

Os projetos que usam a API do Google Gemini precisam de um arquivo `.env` com:

```
GOOGLE_API_KEY=sua_chave_aqui
```

Veja o arquivo `.env.exemplo` para referência. A chave pode ser obtida gratuitamente no [Google AI Studio](https://aistudio.google.com).
