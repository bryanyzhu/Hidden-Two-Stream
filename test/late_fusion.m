% Test the accuracy of very deep networks for UCF101 
clear all;clc;

split = 1;   % change split to corresponding test split
splitStr = sprintf('%02d', split);

%% get the groud-truth labels from ucf101 documents 
labelFile = strcat('./testlist', splitStr, '.txt');
groudTruth = textread(labelFile,'%s','delimiter','\n'); 
 
label = zeros(length(groudTruth),1);
tmp = {};
index = 1;

for i = 1:length(groudTruth)
    tmp{i} = strsplit(groudTruth{i}, '/');
    if i == 1
      label(i) = index;
    else
      if strcmp(tmp{i}{1}, tmp{i-1}{1})
        label(i) = index;
      else
        index = index + 1;
        label(i) = index;
      end
    end
    
end

%% load the prediction scores from UCF101 very deep model 
predict1 = load(strcat('./spatial_quality100_split', splitStr, '.mat'));
predict2 = load(strcat('./temporal_hidden_split', splitStr, '.mat'));

data1 = predict1.spatial_prediction;
data2 = predict2.hidden_prediction;    

data1(data1 < 0) = 0;
data2(data2 < 0) = 0;

for i = 1:size(data1, 1)
    data1(i, :) = data1(i, :) / sum(data1(i, :));
    data2(i, :) = data2(i, :) / sum(data2(i, :));
end

%% weighted average
fusion_weight = 1.5;
data = (data1 + data2 * fusion_weight);

[~, ind_combine] = max(data,[],2);
[~, ind_spatial] = max(data1,[],2);
[~, ind_temporal] = max(data2,[],2);

accuracy_spatial = sum(label == ind_spatial)/length(groudTruth)
accuracy_temporal = sum(label == ind_temporal)/length(groudTruth)
accuracy_average = sum(label == ind_combine)/length(groudTruth)

