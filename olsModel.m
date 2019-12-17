% destination path for the file
rootFilePath = '/Users/donvanderkrogt/matlab/fintech/';

filename = strcat(rootFilePath, 'cleanDatacsv.csv');
delimiterIn = ',';
headerlinesIn = 1;
T = readtable(filename);
monthlyYield = T.m_yield;

% remove variables from table for Binominal Logistic Regression
T = removevars(T, {'amount', 'default', 'm_yield', 'm_reportasofeod', 'interestandpenaltypaymentsmade', 'principalpaymentsmade'});

% convert table to matrix
X = table2array(T);

% create cross-validation partition for data
c = cvpartition(monthlyYield, 'KFold', 10);

idxTrain = training(c, 1);
idxTest = ~idxTrain;
XTrain = X(idxTrain,:);
yTrain = monthlyYield(idxTrain);
XTest = X(idxTest,:);
yTest = monthlyYield(idxTest);

% run Lasso Binominal Logistic Regression
[B, FitInfo] = lassoglm(XTrain, yTrain, 'normal', 'CV', 3);
idxLamdaMinDeviance = FitInfo.IndexMinDeviance;
B0 = FitInfo.Intercept(idxLamdaMinDeviance);
coef = [B0; B(:, idxLamdaMinDeviance)];

test = [ones(length(XTest), 1) XTest];
yhat = times(transpose(test),coef);