import export_fig.*

% destination path for the file
rootFilePath = '/Users/donvanderkrogt/matlab/fintech/';

filename = strcat(rootFilePath, 'pc_data.csv');
delimiterIn = ',';
headerlinesIn = 1;
T = readtable(filename);

% create observed default value
y = T.m_yield;
% remove variables from table for Binominal Logistic Regression
T = removevars(T, {'m_yield', 'default'});

% convert table to matrix
X = table2array(T);

% create cross-validation partition for data
c = cvpartition(length(y), 'KFold', 10);

idxTrain = training(c, 1);
idxTest = ~idxTrain;
XTrain = X(idxTrain,:);
yTrain = y(idxTrain);
XTest = X(idxTest,:);
yTest = y(idxTest);

mdl = fitlm(XTrain, yTrain);

yhat = predict(mdl, XTest);

hold on
scatter(yTest,yhat)
plot(yTest,yTest)
xlabel('Yearly Yield')
ylabel('Predicted Yearly Yield')
hold off
export_fig(strcat(rootFilePath, 'Figures/', 'OLSPredictions_PCA.png'))

% write logit model to xlxs 
mdl = fitlm(X, y, 'linear', ...
	 'VarNames', cat(1, T.Properties.VariableNames{:}, {'default'}));
writetable(mdl.Coefficients, strcat(rootFilePath, 'Tables/PCAOLSTable.xlsx'));
