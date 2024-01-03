clear all

exfile = readtable('BaselTest1.csv');
data = table2array(exfile(:,1:5));

%data2 = data(1:7:end,:);

YearCol = nan(length(data),1);
MonthCol = nan(length(data),1);
DayCol = nan(length(data),1);
HourCol = nan(length(data),1);

data = [data YearCol MonthCol DayCol HourCol];

for i=1:length(data)
    date = string(data(i,1));
    year = extractBetween(date,1,4);
    month = extractBetween(date,5,6);
    day = extractBetween(date,7,8);
    hour = extractBetween(date,9,10);

    data(i,6) = year;
    data(i,7) = month;
    data(i,8) = day;
    data(i,9) = hour;
end

requiredData = data(data(:,7) == 3 & data(:,8) >=7 & data(:,8) <= 21 & ((data(:,9) >= 0 & data(:,9) <= 3) | data(:,9) == 6),2:5);

%padaryti su 16 inputu
input1 = requiredData(1:5:end,1);
input2 = requiredData(1:5:end,2);
input3 = requiredData(1:5:end,3);
input4 = requiredData(1:5:end,4);

input5 = requiredData(2:5:end,1);
input6 = requiredData(2:5:end,2);
input7 = requiredData(2:5:end,3);
input8 = requiredData(2:5:end,4);

input9 = requiredData(3:5:end,1);
input10 = requiredData(3:5:end,2);
input11 = requiredData(3:5:end,3);
input12 = requiredData(3:5:end,4);

input13 = requiredData(4:5:end,1);
input14 = requiredData(4:5:end,2);
input15 = requiredData(4:5:end,3);
input16 = requiredData(4:5:end,4);

output = requiredData(5:5:end,1);

preparedData = [input1 input2 input3 input4 input5 input6 input7 input8 input9 input10 input11 input12 input13 input14 input15 input16 output];

% Normalize the input data to the range [0, 1]
%[preparedData, settings] = mapminmax(preparedData',0,1);

%preparedData = preparedData';

inputs = preparedData(:,1:16);
targets = preparedData(:,17);

%TEST
%min_x = min(targets(:));
%max_x = max(targets(:));
%X_norm = (targets - min_x) ./ (max_x - min_x) .* 2 - 1;
%TEST

[inputs, settings1] = mapminmax(inputs',-1,1);
[targets, settings2] = mapminmax(targets',-1,1);

inputs = inputs';
targets = targets';

train_ratio = 0.784; % 80% of the data for training, 20% for validation
train_size = round(train_ratio * length(preparedData));

train_inputs = inputs(1:train_size,:);
train_targets = targets(1:train_size,:);
test_inputs = inputs(train_size+1:end,:);
test_targets = targets(train_size+1:end,:);


mdl = fitlm(train_inputs, train_targets);

predicted_targets = predict(mdl, test_inputs);

predicted_targets = mapminmax('reverse', predicted_targets', settings2)';
test_targets = mapminmax('reverse', test_targets', settings2)';

for i=1:length(predicted_targets)
    error2 = abs(predicted_targets(i,1) - test_targets(i,1));
    allErrors2(i,1) = error2;
end

figure
plot(mdl);

disp('TRAINING DATA:');

trainOutput = predict(mdl, train_inputs);

train_inputs = mapminmax('reverse', train_inputs', settings1)';
train_targets = mapminmax('reverse', train_targets', settings2)';
trainOutput = mapminmax('reverse', trainOutput', settings2)';

tmin = min(abs(trainOutput - train_targets));
tmax = max(abs(trainOutput - train_targets));
tmae = mean(abs(trainOutput - train_targets));
tmse = mean((trainOutput - train_targets).^2);
tstd = std(abs(trainOutput - train_targets));

disp(['Minimum: ', num2str(tmin)]);
disp(['Maximum: ', num2str(tmax)]);
disp(['Mean Absolute Error: ', num2str(tmae)]);
disp(['Mean Squared Error: ', num2str(tmse)]);
disp(['Standart Variation: ', num2str(tstd)]);

disp('PREDICTED DATA:');

pmin = min(allErrors2);
pmax = max(allErrors2);
pmae = mean(allErrors2);
pmse = mean((predicted_targets - test_targets).^2);
pstd = std(allErrors2);

disp(['Minimum: ', num2str(pmin)]);
disp(['Maximum: ', num2str(pmax)]);
disp(['Mean Absolute Error: ', num2str(pmae)]);
disp(['Mean Squared Error: ', num2str(pmse)]);
disp(['Standart Variation: ', num2str(pstd)]);
