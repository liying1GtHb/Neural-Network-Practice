
% [~,~,y] = SimpleNeuralNetworkYL([784,30,10],trainX, trainY, 30, 10, 3, 2, testX, testY+1);

% [~,~,y] = NeuralNetworkYL([784,30,10],trainX, trainY, 30, 10, 3, 2, trainX(:,1:100), trainYdigits(1:100)+1);

[~,~,y] = NeuralNetworkMatYL([784,30,10],trainX, trainY, 30, 10, 3, 2, testX, testY+1);