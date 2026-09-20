import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = LogisticRegression()
Tmodel = SentenceTransformer('all-MiniLm-L6-v2')

data = pd.read_csv('spam_dataset_1000.csv')
X_array = np.array(data["text"])
y = np.array(data["label"])
X_embedding = Tmodel.encode(X_array)
x_train,x_test,y_train,y_test=train_test_split(X_embedding,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
predict = model.predict(x_test)
accuracy=accuracy_score(y_test,predict)
print(f"the predict value is : {predict}")
print(f"the true value is : {y_test}")
print(f"the accuracy scorre is :{accuracy}")