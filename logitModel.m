% destination path for the file
rootFilePath = '/Users/donvanderkrogt/matlab/fintech/';

filename = strcat(rootFilePath, 'cleanDatacsv.csv');
delimiterIn = ',';
headerlinesIn = 1;
T = readtable(filename);

% create observed default value
obsDefault = T.default;
% remove variables from table for Binominal Logistic Regression
T = removevars(T, {'amount', 'default', 'm_reportasofeod' 'm_yield', 'interestandpenaltypaymentsmade', 'principalpaymentsmade'});

% convert table to matrix
X = table2array(T);

% create cross-validation partition for data
c = cvpartition(obsDefault, 'KFold', 10);

idxTrain = training(c, 1);
idxTest = ~idxTrain;
XTrain = X(idxTrain,:);
yTrain = obsDefault(idxTrain);
XTest = X(idxTest,:);
yTest = obsDefault(idxTest);

% run Lasso Binominal Logistic Regression
[B, FitInfo] = lassoglm(XTrain, yTrain, 'binomial', 'CV', 3);
idxLamdaMinDeviance = FitInfo.IndexMinDeviance;
B0 = FitInfo.Intercept(idxLamdaMinDeviance);
coef = [B0; B(:, idxLamdaMinDeviance)];

% lassoPlot(A, FitInfo, 'plottype', 'CV');
yhat = glmval(coef, XTest, 'logit');
yhatBinom = (yhat>=0.5);
yTest = (yTest==1);

c = confusionchart(yTest,yhatBinom);

accuracy = (c.NormalizedValues(1, 1) + c.NormalizedValues(2, 2)) / (c.NormalizedValues(1, 1) + c.NormalizedValues(1, 2) + c.NormalizedValues(2, 1) + c.NormalizedValues(2, 2))
sensitivity = c.NormalizedValues(2, 2) / (c.NormalizedValues(2, 1) + c.NormalizedValues(2, 2))
specifity = c.NormalizedValues(1, 1) / (c.NormalizedValues(1, 1) + c.NormalizedValues(2, 1))

% show used variabels
for i = 1:length(coef)
    if (coef(i) ~= 0)
        predictiveVars(i) = T(
    end
        
     
end

% Compute Area under the curve
[X, Y, T, AUC] = perfcurve(yTest, yhat, 1);
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Logistic Regression')


