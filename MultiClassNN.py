import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
data=load_iris()
X=data.data
Y=data.target
# print(X.shape)
# print(Y.shape)
Y=to_categorical(Y)
# print(Y[:5])
X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,test_size=0.2,random_state=42
)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model=Sequential()
model.add(Dense(
    16,activation="relu",input_shape=(4,)
))
model.add(Dense(8, activation="relu"))
model.add(Dense(
    3,activation="softmax"
))
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
history=model.fit(
    X_train,Y_train,validation_split=0.2,epochs=50
)
# print(model.predict(X_test))
import numpy as np
prediction=model.predict(X_test)
predicted_classes=np.argmax(prediction,axis=1)
actual_classes=np.argmax(Y_test,axis=1)
print(predicted_classes[:10])
print(actual_classes[:10])