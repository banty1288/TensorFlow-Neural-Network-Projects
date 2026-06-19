from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data=load_breast_cancer()
X=data.data
Y=data.target
# print(X.shape)
# print(Y.shape)
# Train-Test-Split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
print(X_train[:2])
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
model=Sequential()
# Adding first layer
model.add(
    Dense(16,activation="relu",
          input_shape=(30,))
)
model.add(Dropout(0.3))
# model.summary()
# Adding second layer
model.add(
    Dense(4,activation="relu")
)
# model.summary()
# Adding output layer
model.add(
    Dense(1,activation="sigmoid")
)
# model.summary()
# Compilation of the network
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# history=model.fit(
#     X_train,Y_train,epochs=20,batch_size=32,validation_split=0.2
# )
from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
history=model.fit(
    X_train,
    Y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stop]
)
loss,accuracy=model.evaluate(X_test,Y_test)
print("Test Accuracy:", accuracy)
print("Test Loss:", loss)
print(history.history.keys())

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['Train','Validation'])
plt.show()