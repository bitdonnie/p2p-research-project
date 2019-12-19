% destination path for the file
rootFilePath = '/Users/donvanderkrogt/matlab/fintech/';

filename = strcat(rootFilePath, 'pc_data.csv');
delimiterIn = ',';
headerlinesIn = 1;
T = readtable(filename);

% create observed default value
y = T.default;
% remove variables from table for Binominal Logistic Regression
T = removevars(T, {'m_yield', 'default'});

% convert table to matrix
X = table2array(T);

% create cross-validation partition for data
c = cvpartition(y, 'KFold', 10);

idxTrain = training(c, 1);
idxTest = ~idxTrain;
XTrain = X(idxTrain,:);
yTrain = y(idxTrain);
XTest = X(idxTest,:);
yTest = y(idxTest);

[B, dev, stats] = glmfit(XTrain, yTrain, 'binomial');

% lassoPlot(A, FitInfo, 'plottype', 'CV');
yhat = glmval(B, XTest, 'logit');
yhatBinom = (yhat>=0.5);
yTest = (yTest==1);

c = confusionchart(yTest,yhatBinom);

accuracy = (c.NormalizedValues(1, 1) + c.NormalizedValues(2, 2)) / (c.NormalizedValues(1, 1) + c.NormalizedValues(1, 2) + c.NormalizedValues(2, 1) + c.NormalizedValues(2, 2));
sensitivity = c.NormalizedValues(2, 2) / (c.NormalizedValues(2, 1) + c.NormalizedValues(2, 2));
specifity = c.NormalizedValues(1, 1) / (c.NormalizedValues(1, 1) + c.NormalizedValues(2, 1));

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


