import pandas as pd 
from sklearn.tree import DecisionTreeClassifier


#Armazenando com leitura os dados dos conjuntos test e train
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Visualizando as primeiras linhas do conjunto Train
train.head()

#Removendo colunas desnecessárias para predição como nome, ingresso e cabina 

train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#Criando novo DataFrame a partir de One-hot enconding
new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)

#Visualizando as primeiras linhas do Dataset test
new_data_test.head()

#Visualizando as primeiras linhas do Dataset Train
new_data_train.head()


#Quantificando os valores nulos do conjunto train
new_data_train.isnull().sum().sort_values(ascending=False).head(10)

#Quantificando os valores nulos do conjunto test
new_data_test.isnull().sum().sort_values(ascending=False).head(10)

#Preenchendo os valores nulos dos conjuntos com a média das idades dos passageiros
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace=True)

#Preenchendo o valor nulo 'Fare' encontrado no new_data_test, vou imputar a média de 'Fare'
new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)

#Separando as features e target para criação do modelo
x = new_data_train.drop('Survived', axis=1)
y = new_data_train['Survived']

#Criando o modelo
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(x,y)

#Criando DataFrame incluindo as colunas PassengerId e Survived para exportar em .csv
submission = pd.DataFrame()
submission['PassengerId'] = new_data_test['PassengerId']
submission['Survived'] = tree.predict(new_data_test)

#Criando o arquivo Csv 
submission.to_csv('submission.csv', index=False)



