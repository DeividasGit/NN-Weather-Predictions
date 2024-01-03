clear all

exfile = readtable('BaselTest1.csv');
data = table2array(exfile(:,1:2));

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

    data(i,3) = year;
    data(i,4) = month;
    data(i,5) = day;
    data(i,6) = hour;
end

requiredData = data(data(:,4) == 3 & data(:,5) >=7 & data(:,5) <= 21 & ((data(:,6) >= 0 & data(:,6) <= 3) | data(:,6) == 6),2);

input1 = requiredData(1:5:end,:);
input2 = requiredData(2:5:end,:);
input3 = requiredData(3:5:end,:);
input4 = requiredData(4:5:end,:);
output = requiredData(5:5:end,:);

preparedData = [input1 input2 input3 input4 output];

% Normalize the input data to the range [0, 1]
%[preparedData, settings] = mapminmax(preparedData',0,1);

%preparedData = preparedData';

inputs = preparedData(:,1:4);
targets = preparedData(:,5);

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

net = feedforwardnet(5);

net.trainFcn = 'trainbr';
net.trainParam.epochs = 100; % Set the maximum number of epochs
net.trainParam.goal = 1e-3; % Set the performance goal
net.trainParam.showWindow = true; % Show training progress window
net.trainParam.min_grad = 1e-2;
net.trainParam.max_fail = 6;
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio   = 0.2;
net.divideParam.testRatio  = 0;
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';

% Train the neural network
[net, tr] = train(net, train_inputs', train_targets');

predicted_targets = net(test_inputs');
predicted_targets = predicted_targets';

predicted_targets = mapminmax('reverse', predicted_targets', settings2)';
test_targets = mapminmax('reverse', test_targets', settings2)';

for i=1:length(predicted_targets)
    error2 = abs(predicted_targets(i,1) - test_targets(i,1));
    allErrors2(i,1) = error2;
end

figure
plotperform(tr)

disp('TRAINING DATA:');

trainOutput = sim(net, train_inputs')';

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

