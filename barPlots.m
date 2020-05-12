% IMK Kernels
set(0, 'DefaultAxesFontWeight', 'normal', ...
      'DefaultAxesFontSize', 18, ...
      'DefaultAxesFontAngle', 'normal', ... 
      'DefaultAxesFontWeight', 'normal', ... 
      'DefaultAxesTitleFontWeight', 'bold', ...
      'DefaultAxesTitleFontSizeMultiplier', 1.2) ;
set(groot,'defaultLineLineWidth',1)

% 32 64 128 256
y = [.886 .859 .725; .874 .86 .719; .875 .861 .709; .88 .868 .713];
%x = [32, 64, 128, 256]
x = categorical({'32','64','128','256'});
x = reordercats(x,{'32','64','128','256'});
figure
b = bar(x, y)
legend('IMK', 'RBF', 'POLY')
title('IMK Algorithm Performance on Musk1 With Various Kernels')
xlabel('Clusters')
ylabel('Accuracy')
ax = b.Parent;
ax.YMinorTick = 'on'
%xticks([0 .1 .2 .3 .4 .5 .6 .7 .9 1.0])
set(ax, 'YTick', 0:.1:1)
ylim([0 1])
% 
% ci = 0.95 ;
% alpha = 1 — ci;
% n = size(dat,1);
% T_multiplier = tinv(1-alpha/2, n-1);
% ci95 = T_multiplier*std(dat)/sqrt(n);


%% SMI
% avg, minmax, mma
y = [.869, .877, .859; .849, .829, .769; .605, .621, .616];
e = [.0231, .0237, .025; .0242, .024, .0249; .0295, .0319, .026]

%x = [32, 64, 128, 256]
x = categorical({'Musk1', 'Elephant', 'Fox'});
x = reordercats(x,{'Musk1', 'Elephant', 'Fox'});
c = 3;
figure
hold on
bar(x, y)
for j = 1:3
    for i = 1:3
        errorbar(i+.4,y(i,j),e(i,j))
    end
end


legend('Mean', 'Min-Max', 'Combined')
title('Accuracy of Various Mapping Functions for Simple MI')
xlabel('Dataset')
ylabel('Accuracy')

%% SMI Kernel- Musk
%% SMI
% avg, minmax, mma
y = [.869, .877, .859; .77,  .812, .796; .881, .906, .856];
e = [.0231, .0237, .025; .0242, .024, .0249; .0295, .0319, .026]

%x = [32, 64, 128, 256]
x = categorical({'RBF', 'Polynomial', 'Linear'});
x = reordercats(x,{'RBF', 'Polynomial', 'Linear'});
c = 3;
figure
hold on
b = bar(x, y)
title('Accuracy on Musk1 with Various Kernels for Simple MI')
xlabel('Kernel')
ylabel('Accuracy')
legend('Mean', 'Min-Max', 'Combined')
ax = b.Parent;
ax.YMinorTick = 'on'
%xticks([0 .1 .2 .3 .4 .5 .6 .7 .9 1.0])
set(ax, 'YTick', 0:.1:1)

%% HBOW clusters

%y = [0.704 0.734 0.73 0.734 0.74 .76; 0.713 0.78 0.785 0.79 0.757 0.708; 0.802 0.812 0.795 .826 0.838 0.846; 0.576 0.785 0.824 0.843 0.895 0.876; 0.76 0.785  0.819 0.843 0.882 0.874 ];
y = [0.576 0.785 0.824 0.843 0.895 0.876; 0.76 0.785  0.819 0.843 0.882 0.874 ];

%x = [32, 64, 128, 256]
% x = categorical({'Uncertainty', 'Plausibility', 'Codebook', 'Euclidean', 'Similarity'});
% x = reordercats(x,{'Uncertainty', 'Plausibility', 'Codebook', 'Euclidean', 'Similarity'});


x = categorical({'Euclidean', 'Similarity'});
x = reordercats(x,{'Euclidean', 'Similarity'});

figure
hold on
b = bar(x, y)
title('HBOW Vocabulary Size Experiement')
xlabel('Mapping Function')
ylabel('Accuracy')
legend('8', '16', '32', '64','128', '256')
ax = b.Parent;
ax.YMinorTick = 'on'
%xticks([0 .1 .2 .3 .4 .5 .6 .7 .9 1.0])
set(ax, 'YTick', 0:.1:1)

%% Yards
y = [0.853 0.795 ;0.78 0.802; 0.572 0.569 ]
%x = [32, 64, 128, 256]
x = categorical({'Musk1', 'Elephant', 'Fox'});
x = reordercats(x,{'Musk1', 'Elephant', 'Fox'});

figure
hold on
b = bar(x, y)
title('YARDS Vocabulary Experiment')
xlabel('Dataset')
ylabel('Accuracy')
legend('All Instances', 'Positive Only')
ax = b.Parent;
ax.YMinorTick = 'on'
%xticks([0 .1 .2 .3 .4 .5 .6 .7 .9 1.0])
set(ax, 'YTick', 0:.1:1)

%% DBOW
y = [.932 .935 .922 .921, .905]
%x = [32, 64, 128, 256]
c = 1:5

figure
hold on
for i = 1:5
    bar(c(i),y(i),.6)

end
legend('SVM, Sim, K=128','SVM, Sim, K=all','SVM, Sim, K=all*3', 'SVM, Euclid, K=256' , 'Adaboost, Sim, K=256')
ylabel('Accuracy on Musk1')
title('Summary of DBOW Tuning Experiments')
ylim([.8 1])

%%
y = [0.853 0.795 ]
x = [1 2]
%x = [32, 64, 128, 256]
%x = categorical({'Musk1'});
%x = reordercats(x,{'Musk1'});
figure
hold on
b = bar(x, y)
title('YARDS Vocabulary Experiment')
xlabel('Dataset')
ylabel('Accuracy')
legend('All Instances', 'Positive Only')
ax = b.Parent;
ax.YMinorTick = 'on'
%xticks([0 .1 .2 .3 .4 .5 .6 .7 .9 1.0])
set(ax, 'YTick', 0:.1:1)

%% HBOW Sim

% plaus, uncert euc, yards sim, codebook, 
y = [.79 .76 .882 .853 .895 .846 ; .803 .8 .78 .78 .775  .735 ; .607 .598 .5832  .583 .57 .57  ] 
%x = [32, 64, 128, 256]
x = categorical({'Musk1', 'Elephant', 'Fox'});
x = reordercats(x,{'Musk1', 'Elephant', 'Fox'});

figure
hold on
b = bar(x, y)
title('HBOW Mapping Function Experiment')
xlabel('Dataset')
ylabel('Accuracy')
legend('Plausibility', 'Uncertainty', 'Similarity', 'Yards', 'Euclidean', 'Codebook' )
ax = b.Parent;
ax.YMinorTick = 'on'
%xticks([0 .1 .2 .3 .4 .5 .6 .7 .9 1.0])
set(ax, 'YTick', 0:.1:1)
ylim([0 1])


%% Final Plot
% plaus, uncert euc, yards sim, codebook, 
y = [.932 .907 .895 .88 .853  ; .8 .84 .803 .849  .78  ; .693 .685 .607 .605 .583    ] 
%x = [32, 64, 128, 256]
x = categorical({'Musk1', 'Elephant', 'Fox'});
x = reordercats(x,{'Musk1', 'Elephant', 'Fox'});

figure
hold on
b = bar(x, y)
title('Embedded Space Algorithm Accuracies ')
xlabel('Dataset')
ylabel('Accuracy')
legend('DBOW',  'IMK', 'HBOW',  'SMI', 'YARDS')
ax = b.Parent;
ax.YMinorTick = 'on'
%xticks([0 .1 .2 .3 .4 .5 .6 .7 .9 1.0])
set(ax, 'YTick', 0:.1:1)
ylim([0 1])



