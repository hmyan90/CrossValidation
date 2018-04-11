% set global variance
folderNum = 100;  % set 10 for 10-fold, set 100 for leave one out cross validation
repeatTime = 1; % set 500 for 10-fold, set 1 for leave one out cross validation
isBias = 0; % set 1 for with bias term, set 0 for without bias term 
lambdaRange = [logspace(-2, 2, 50)]';

% load data from file
trainMat = load('train.txt');
testMat = load('test.txt');

% use cross validation to find best Lambda
numPerFolder = size(trainMat, 1)/folderNum;
RMSES = zeros(size(lambdaRange, 1), folderNum, repeatTime);
identMat = eye(10);
if (isBias == 0)
  identMat(1,1) = 0;
endif

for l = 1: size(lambdaRange, 1)
  lambda = lambdaRange(l);
  for r = 1: repeatTime
    trainMatShuffled = trainMat(randperm(size(trainMat, 1)), :); 
    for f = 1: folderNum
      trainSet = trainMatShuffled([1:(f-1)*numPerFolder, f*numPerFolder+1:folderNum*numPerFolder], :);
      validSet = trainMatShuffled((f-1)*numPerFolder+1: f*numPerFolder, :);
      train_x = (trainSet(:, 1))';
      train_X = transX(train_x);
      train_t = trainSet(:, 2);
      %disp(det(train_X*train_X'+lambda*eye(10)));
      w = inv(train_X*train_X'+lambda*identMat)*train_X*train_t;
      valid_x = (validSet(:, 1))';
      valid_X = transX(valid_x); 
      valid_t = validSet(:, 2);
      RMSES(l, f, r) = sqrt(mean((valid_X'*w - valid_t).^2));
    endfor
  endfor
endfor 

RMSES = mean(RMSES, 3); % two dimension, row: lambda, col: folder
[_, idx] = min(mean(RMSES, 2));
bestLambda = lambdaRange(idx);
disp('Best lambda is: '), disp(bestLambda);

% plot errorBar on lambda
errorbar(log(lambdaRange), mean(RMSES, 2), std(RMSES, 1, 2));
xlabel('log(\lambda)', 'fontsize', 14); 
ylabel('RMSE',  'fontsize', 14);
title('Leave one out cross validation without bias');
print('LOOWithoutBias.jpg');

% calculate w using bestLambda 
x = (trainMat(:, 1))';
X = transX(x);
t = trainMat(:, 2);
w = inv(X*X'+bestLambda*identMat)*X*t;
disp('Cofficients using best lambda is: '), disp(w);

% calculate test error
test_x = (testMat(:, 1))';
test_X = transX(test_x);
test_t = testMat(:, 2);
RMSE = sqrt(mean((test_X'*w - test_t).^2));
disp('RMSE of test set is: '), disp(RMSE);
