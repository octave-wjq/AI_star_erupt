from sklearn import preprocessing
import numpy as np
def mnuberstd(X):
    X_mean=X.mean(axis=0)
    X_std=pow(X.std(axis=0),0.5)
    print(1)
    X=(X-X_mean)/X_std
    return X
y=np.full((3,2),3)
print(mnuberstd(y))