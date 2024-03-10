import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

tabela = pd.read_csv("clientes.csv")
print(tabela.info())

# Label Encoder
codificador =  LabelEncoder()
tabela["profissao"] = codificador.fit_transform(tabela["profissao"])
tabela["mix_credito"] = codificador.fit_transform(tabela["mix_credito"])
tabela["comportamento_pagamento"] = codificador.fit_transform(tabela["comportamento_pagamento"])


y = tabela["score_credito"]
x = tabela.drop(columns={"score_credito", "id_cliente"})

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

modelo_arvoredecisao.fit(x_treino,  y_treino)
modelo_knn.fit(x_treino, y_treino)

previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

previsoes = modelo_arvoredecisao.predict(tabela)
print(previsoes)

# print(accuracy_score(" ArvoreDecisao Y Teste: ", y_teste, previsao_arvoredecisao))
# print(accuracy_score("PrevisaoKNN Y Teste: ", y_teste, previsao_knn))

