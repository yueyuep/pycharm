from sklearn.neural_network import MLPClassifier
x=[[0.,0.],[1.,1.]];
y=[0,1]
model=MLPClassifier(solver='lbfgs')
model.fit(x,y)
label=model.predict([[0.5,0.]])
print(label);