from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data=fetch_california_housing()
X=data.data
Y=data.target
# print(X.shape)
# print(Y.shape)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(
    Dense(16,activation="relu",input_shape=(8,))
)
model.add(
    Dense(8,activation="relu")
)
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=['mae']
)
history=model.fit(
    X_train,Y_train,validation_split=0.2,epochs=50,batch_size=32)
prediction=model.predict(X_test) 
# print(prediction[:5])  
# for i in range(10):
#     print("Actual:",Y_test[i])
#     print("Predicted:",prediction[i][0])
#     print("  ")
loss,mae=model.evaluate(X_test,Y_test)
print("Test MAE",mae)