net = NeuralNetwork([784 30 10]);
net.SGDClf(trainX,trainY,30,10,3,testX,testY+1);