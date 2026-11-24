# desafio_bootcamp_dio_deteccao_fraudes_transacoes_bancarias
Desafio: Hora de praticar, aprender fazendo e treinar modelos que detectam fraude em transações reais de cartão de crédito, um problema em que a fraude é rara e a acurácia engana.


## 1. Visão Geral e O Problema do Desbalanceamento
- Resumo do objetivo: identificar transações fraudulentas (`Class = 1`) em relação às normais (`Class = 0`) a partir de `Time`, `Amount` e componentes de PCA (`V1` a `V28`).
- Por que a acurácia engana: com cerca de `0,17%` de fraudes e `99,83%` de operações normais, um modelo ingênuo que prevê sempre `0` alcança acurácia altíssima e falha completamente no objetivo de negócio.
- Como o dataset é carregado: leitura direta via URL no notebook (sem subir o CSV ao repositório devido ao limite de tamanho do GitHub).

## 2. Métricas de Avaliação na Classe de Fraude
- **Recall (Revocação):** proporção de fraudes reais corretamente detectadas (crítico para reduzir fraudes que passam despercebidas).
- **Precisão:** proporção das transações marcadas como fraude que realmente são fraude (ajuda a controlar alertas falsos e bloqueios indevidos).
- **F1-Score:** equilíbrio entre precisão e recall na classe `1`.
- Curvas utilizadas na análise: **Curva de Precisão-Recall** (principal para cenários desbalanceados) e **Curva ROC**, além do `classification_report`.



## 3. Preparação dos Dados e Pipeline
- Transformação logarítmica (`np.log1p`) em `Amount` para lidar com assimetria/cauda longa.
- Padronização com `StandardScaler` e divisão treino/teste estratificada (`stratify`) para preservar a proporção de fraudes.
- Encapsulamento em `Pipeline` (e busca de hiperparâmetros com `GridSearchCV` guiada por recall) para manter o fluxo organizado e evitar vazamento de dados.

## 4. Comparação entre Modelos
- Tabela ou resumo comparando **Regressão Logística** (baseline), **Random Forest** (com `class_weight='balanced'`) e **XGBoost** — destacando **recall**, **precisão** e **F1-score da classe 1** (não apenas a acurácia).
- Breve leitura do trade-off entre encontrar mais fraudes e gerar mais falsos positivos.

## 5. Limiar de Decisão e Explicabilidade com SHAP
- **Ajuste de limiar  de `0,5` para `0,3` por que reduzir o corte de probabilidade aumenta a captura de fraudes e qual o impacto esperado na precisão.
- **Principais achados do SHAP:** quais variáveis tiveram maior peso global e como features específicas empurraram previsões para fraude ou transação normal (incluindo exemplos de casos observados).


## 6. O Que Eu Mudei em Relação à Aula

Ao estruturar o projeto de detecção de fraude no `creditcard.csv`, organizei o fluxo com foco em evitar vazamento de dados, lidar com o forte desbalanceamento da base (~0,17% de fraudes) e tratar o limiar de probabilidade como uma decisão explícita de negócio. A partir das features originais, incorporei transformações para `Amount` (`np.log1p`) e para `Time` (extração temporal, chegando a `31` colunas em `X`), fiz a divisão estratificada em `70%` treino e `30%` teste (`199.364` e `85.443` registros, respectivamente, mantendo `148` fraudes no teste) e apliquei o `StandardScaler` apenas no conjunto de treino antes de transformar o teste.

Na etapa de balanceamento e modelagem, apliquei o balanceamento exclusivamente nos dados de treino (`X_train_scaled` e `y_train`), preservando o conjunto de teste na proporção real e evitando combinar balanceamento de dados com peso duplicado no estimador para não inflar os falsos positivos. Em vez de depender apenas do corte padrão de `0.5`, utilizei `predict_proba` para avaliar o impacto de baixar o limiar para `0.3` e comparei Regressão Logística, Random Forest e XGBoost na mesma base de teste. Com isso, o **XGBoost treinado nos dados balanceados com limiar `>= 0.3`** se destacou como escolha principal por manter alto recall (`0,8311`, com `123` fraudes detectadas e `25` perdidas) e entregar precisão de `0,4377` e `F1-score` de `0,5734` com apenas `158` falsos positivos — superando o volume de alertas falsos do Random Forest (`341` `FP`) e da Regressão Logística (`4.212` `FP` no mesmo limiar). Por fim, mapeei os índices `f0..fn` do XGBoost para os nomes reais das colunas na análise de importância (`gain`) e salvei o `scaler`, o modelo e o limiar com `joblib` para reutilização em novas transações.


## 7. Conclusão

Resumo Executivo — Detecção de Fraude (creditcard.csv)1. Principais achados e preparo dos dadosDesbalanceamento severo: a base tem cerca de 0,17% de fraudes (Class = 1), o que torna a acurácia global enganosa e exige avaliar recall, precisão, F1-score, matriz de confusão e curva de Precisão-Recall (PR-AUC).Engenharia e prevenção de vazamento: o conjunto foi dividido de forma estratificada (70% treino / 30% teste, com 148 fraudes no teste), incorporando transformações temporais e de valor (Amount / Time), aplicando StandardScaler apenas no treino (fit_transform) e restringindo o balanceamento (como SMOTE) exclusivamente aos dados de treino.Sinais não lineares dominam: relações não lineares entre as componentes e variáveis transformadas separam muito melhor fraudes de transações normais do que um limite de decisão estritamente linear.2. Comparação de modelos (conjunto de teste, limiar >= 0.3)ModeloPrecisão (fraude)Recall (fraude)F1-scoreVP (TP)FN (FN)FP (FP)Leitura práticaXGBoost (balanceado)0,4377 (43,77%)0,8311 (83,11%)0,573412325158Melhor equilíbrio entre captura de fraudes e controle de alertas falsosRandom Forest (balanceado)0,2682 (26,82%)0,8446 (84,46%)0,407212523341Pega 2 fraudes a mais que o XGBoost, mas gera mais que o dobro de FPRegressão Logística (referência)0,0304 (3,04%)0,8919 (89,19%)0,0588132164.212Recall alto, porém com excesso de falsos positivos (4.212)3. Escolha do modelo e do limiarModelo principal: XGBoost treinado em dados balanceados, por alcançar o maior F1-score (0,5734) e reduzir drasticamente os falsos positivos (158 contra 341 do Random Forest e 4.212 da Regressão Logística no corte 0.3).Escolha do limiar (0.3): mantém recall acima de 83% (123 de 148 fraudes detectadas) com cerca de 43,8% de precisão nos alertas gerados. Na Regressão Logística, reduzir o limiar de 0.5 para 0.3 trouxe apenas +4 fraudes (128 → 132) ao custo de +2.319 falsos positivos (1.893 → 4.212); já no XGBoost balanceado, o corte 0.3 entrega alto recall com volume de FP muito mais manejável.Critério de ajuste final: caso a operação exija ainda menos FN ou ainda menos FP, basta alterar o corte na etapa de decisão (y_prob >= limiar) sem precisar retreinar o modelo.




Em um cenário com forte desbalanceamento (~0,17% de fraudes), a acurácia global não reflete a capacidade real de detecção — por isso a avaliação priorizou **recall**, **precisão**, **F1-score**, **matriz de confusão** e **curva de Precisão-Recall**.

* **Modelo escolhido:** **XGBoost treinado em dados balanceados** (com `StandardScaler` e balanceamento aplicados apenas no treino, e avaliação no teste estratificado com `148` fraudes), alcançando **F1-score de `0,5734`**, **recall de `0,8311`** (`123` fraudes detectadas e `25` perdidas) e **precisão de `0,4377`** com **`158` falsos positivos** no limiar **`>= 0.3`**.
* **Comparação com os demais modelos:** o Random Forest balanceado no limiar `0.3` obteve recall levemente maior (`0,8446`, `125` fraudes), porém com mais que o dobro de falsos positivos (`341`, precisão de `0,2682` e F1 de `0,4072`), enquanto a Regressão Logística gerou excesso de alertas falsos (`4.212` `FP` no limiar `0.3`).
* **Ajuste operacional:** o limiar de `0.3` oferece um bom equilíbrio entre capturar a maior parte das fraudes e manter o volume de falsos alertas sob controle, podendo ser ajustado na regra `(y_prob >= limiar)` conforme o custo operacional de falsos negativos e falsos positivos.

* **Gráfico Shap**
Resultado da transação: $f(x) = -11.179$ (bem abaixo do valor base $E[f(X)] = -0.061$), indicando uma previsão fortemente inclinada para transação legítima (classe 0), muito distante de qualquer limiar de fraude como 0.3 ou 0.5.Maior impacto negativo (empurra para não fraude, em azul):V4 (-1.103): maior contribuição individual, com -2.76.V14 (0.854): segunda maior redução, com -2.59.V11 (-1.613) e V12 (0.636): reduzem o score em -1.31 e -1.3, respectivamente.Valor e tempo da transação:Amount_log (3.648) contribui com -0.9 e Amount (37.4) com -0.5, ambos puxando para baixo.Time (127294) também empurra para baixo com -0.64, junto com V2 (1.891, -0.53) e V22 (-2.167, -0.69).Demais variáveis: o grupo 22 other features soma apenas +0.1 (único bloco em vermelho/rosa), impacto mínimo frente ao conjunto de fatores negativos.Leitura práticaPara este caso específico, a combinação de V4, V14, V11 e V12 — reforçada por Amount_log, Time, V22, V2 e Amount — leva o score de -0.061 para -11.179, tornando a classificação como não fraude bastante clara. Vale observar também que Amount e Amount_log aparecem juntos na explicação local, refletindo a presença de ambas as representações no conjunto de features


## 8. Como Executar e Estrutura do Repositório
- Link ou referência ao notebook principal (com saídas, tabelas e gráficos já salvos):
 https://colab.research.google.com/drive/1w47EpnkDUKM0o_dNzYK3MougBKvoR8wA?usp=sharing
