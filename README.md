# 📉 **Customer Churn Prediction**

Modelo de ML para prever a probabilidade de um cliente cancelar os serviços de uma empresa de telecomunicações, servido através de uma **API**.

Dataset utilizado: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Sobre o projeto

Churn é um forte indicador de saúde de negócios baseados em assinatura. Antecipar um potencial cliente propenso a cancelar o serviço permite que a empresa atue de forma estratégica (oferecendo descontos ou ajustes de plano) antes da perda do cliente.

---

## Dataset

- **7.043 clientes** e **21 atributos** (dados demográficos, serviços contratados, tipo de contrato e cobrança).
- **Sem valores nulos e sem registros duplicados.**
- Variável-alvo `Churn` **desbalanceada**: ~ **73% "Não" vs 27% "Sim"**.

##  Principais insights da EDA

- Forte correlação entre `TotalCharges`, `tenure` e `MonthlyCharges` (esperado, já que `TotalCharges ≈ tenure × MonthlyCharges`), indicando multicolinearidade entre essas variáveis numéricas.
- Sem outliers relevantes nas variáveis numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`).
- Distribuição de classes desbalanceada reforçou a necessidade de técnicas de balanceamento antes do treinamento.

---

## Metodologia

| Etapa | Técnica utilizada |
|---|---|
| Encoding de variáveis categóricas | `LabelEncoder` |
| Split treino/teste | 80% / 20% |
| Balanceamento de classes | Comparação entre **SMOTE (oversampling)** e **RandomUnderSampler (undersampling)** |
| Seleção de modelo | Validação cruzada (5-fold) comparando Decision Tree, Random Forest e XGBoost |
| Avaliação final | Classification report + matriz de confusão no conjunto de teste |

---

## 📈 Resultados

### Validação cruzada (acurácia média, 5-fold)

| Modelo | Oversampling (SMOTE) | Undersampling |
|---|---|---|
| Decision Tree | 0.78 ± 0.07 | 0.67 ± 0.01 |
| **Random Forest** | **0.84 ± 0.07** | **0.74 ± 0.01** |
| XGBoost | 0.83 ± 0.08 | 0.72 ± 0.02 |

O **Random Forest** teve o melhor desempenho em ambos os cenários e foi o modelo escolhido.

### Desempenho no conjunto de teste (Random Forest + SMOTE)

| Classe | Precision | Recall | F1-score | Suporte |
|---|---|---|---|---|
| Não Churn (0) | 0.85 | 0.85 | 0.85 | 1.036 |
| Churn (1) | 0.58 | 0.57 | 0.58 | 373 |
| **Acurácia geral** | | | **0.78** | 1.409 |

> **Observação:** o experimento com *undersampling* apresentou recall bem superior para a classe de churn (**0.77 vs. 0.57**), ou seja, identifica mais clientes que realmente cancelariam, mas às custas de mais falsos positivos. Como em problemas de churn o custo de "não detectar" um cliente propenso a cancelar costuma ser maior do que o de uma abordagem preventiva desnecessária, essa é uma troca relevante a ser avaliada e um candidato natural para ser mais explorado.

---

## API

O modelo é servido via **FastAPI**, expondo um endpoint que recebe os dados cadastrais/de consumo de um cliente e retorna a probabilidade de churn.

### `POST /processar`

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  ...,
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

**Resposta:**

```json
{
  "Churn": 0.31,
  "No Churn": 0.69
}
```

---

## Tecnologias

- **Pandas / NumPy**
- **Matplotlib / Seaborn** — visualização
- **Scikit-learn** — encoding, split, modelos e métricas
- **imbalanced-learn** — SMOTE e RandomUnderSampler
- **XGBoost**
- **FastAPI** — API

---

## ▶ Como rodar o projeto

### 1. Clonar o repositório e instalar as dependências

```bash
git clone https://github.com/ricardo-ervilha/Customer-Churn-Prediction.git
cd customer-churn-prediction

uv sync
```

### 2. Gerar os artefatos do modelo

Execute o `notebook.ipynb` (dentro de `src/`) do início ao fim. Ele fará a EDA, o treinamento e salvará `customer_churn_model.pkl` e `encoders.pkl`, necessários para a API.

### 3. Subir a API

```bash
uvicorn main:app --reload
```

A API ficará disponível em `http://127.0.0.1:8000`.

### 4. Testar o endpoint

```bash
python teste_api.py
```

---

## 💡 Possíveis melhorias

- Tuning de hiperparâmetros;
- Avaliar métricas de negócio para decidir entre Oversampling e Undersampling;
- Testar `OneHotEncoding`.