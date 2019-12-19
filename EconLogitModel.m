% destination path for the file
rootFilePath = '/Users/donvanderkrogt/matlab/fintech/';

filename = strcat(rootFilePath, 'latestData.csv');
delimiterIn = ',';
headerlinesIn = 1;
T = readtable(filename);

% create observed default value
y = T.default;

% remove irrelevant variables from table
relVariables = ["age", "interest", "loanduration", "existingliabilities", "m_newcreditcustomer", "verificationtype3", "verificationtype4", "gender2", "gender3", "country2", "country3", "country4", "useofloan2", "useofloan3", "useofloan4", "useofloan5", "useofloan6", "useofloan7", "useofloan8", "useofloan9", "education2", "education3", "education4", "education5", "employmentstatus1", "homeownershiptype1", "log_incometotal", "log_appliedamount", "log_liabilitiestotal"];
tableProps = T.Properties.VariableNames;
for i = 1:length(tableProps)
    if ~any(contains(relVariables, string(tableProps(i))))
        T = removevars(T, tableProps(i));
    end
end

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

% train a binomial generalized model with train set
mdl = fitglm(XTrain, yTrain, 'linear', ...
      'distr', 'binomial', 'VarNames', cat(1, T.Properties.VariableNames{:}, {'default'}));

% test the binomial generalized model with test set
yhat = glmval(mdl.Coefficients.Estimate, XTest, 'logit');
yhatBinom = (yhat>=0.5);
yTest = (yTest==1);

c = confusionchart(yTest, yhatBinom);
export_fig(strcat(rootFilePath, 'Figures/', 'ConfusionMatrix_Eco.png'))

accuracy = (c.NormalizedValues(1, 1) + c.NormalizedValues(2, 2)) / (c.NormalizedValues(1, 1) + c.NormalizedValues(1, 2) + c.NormalizedValues(2, 1) + c.NormalizedValues(2, 2))
sensitivity = c.NormalizedValues(2, 2) / (c.NormalizedValues(2, 1) + c.NormalizedValues(2, 2))
specifity = c.NormalizedValues(1, 1) / (c.NormalizedValues(1, 1) + c.NormalizedValues(2, 1))

% Compute Area under the curve
[X, Y, T, AUC] = perfcurve(yTest, yhat, 1);
hold on
plot(X,Y)
plot([0, 1], [0, 1])
xlabel('False positive rate') 
ylabel('True positive rate')
title('Economic motivation for Classification by Binomial Logistic Regression')
hold off
export_fig(strcat(rootFilePath, 'Figures/', 'AUC_Eco.png'))

% write logit model to xlxs 
mdl = fitglm(X, y, 'linear', ...
      'distr', 'binomial', 'VarNames', cat(1, T.Properties.VariableNames{:}, {'default'}));
writetable(mdl.Coefficients, strcat(rootFilePath, 'Tables/EcoLogitTable.xlsx'));
