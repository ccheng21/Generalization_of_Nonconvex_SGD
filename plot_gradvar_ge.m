function plot_gradvar_ge()
% Get the data
titles = 'Neural Network: MNIST, Batch-size: 128, Augmented Training Data';
%titles = 'ResNet: CIFAR-10, Batch-size: 128, Augmented Training Data';
fileID_left = [fopen('path1\var_gradient.txt','r');
          fopen('path2\var_gradient.txt','r');
          fopen('path3\var_gradient.txt','r');
          fopen('path4\var_gradient.txt','r');
          fopen('path5\var_gradient.txt','r');
          fopen('path6\var_gradient.txt','r');
          fopen('path7\var_gradient.txt','r');
          fopen('path8\var_gradient.txt','r');
          fopen('path9\var_gradient.txt','r');
          ];
fileID_right = [fopen('path1\GE.txt','r');
          fopen('path2\GE.txt','r');
          fopen('path3\GE.txt','r');
          fopen('path4\GE.txt','r');
          fopen('path5\GE.txt','r');
          fopen('path6\GE.txt','r');
          fopen('path7\GE.txt','r');
          fopen('path8\GE.txt','r');
          fopen('path9\GE.txt','r');
          ];
formatSpec = '%f';
Y1_left = fscanf(fileID_left(1),formatSpec);
Y2_left = fscanf(fileID_left(2),formatSpec);
Y3_left = fscanf(fileID_left(3),formatSpec);
Y4_left = fscanf(fileID_left(4),formatSpec);
Y5_left = fscanf(fileID_left(5),formatSpec);
Y6_left = fscanf(fileID_left(6),formatSpec);
Y7_left = fscanf(fileID_left(7),formatSpec);
Y8_left = fscanf(fileID_left(8),formatSpec);
Y9_left = fscanf(fileID_left(9),formatSpec);
Y1_right = fscanf(fileID_right(1),formatSpec);
Y2_right = fscanf(fileID_right(2),formatSpec);
Y3_right = fscanf(fileID_right(3),formatSpec);
Y4_right = fscanf(fileID_right(4),formatSpec);
Y5_right = fscanf(fileID_right(5),formatSpec);
Y6_right = fscanf(fileID_right(6),formatSpec);
Y7_right = fscanf(fileID_right(7),formatSpec);
Y8_right = fscanf(fileID_right(8),formatSpec);
Y9_right = fscanf(fileID_right(9),formatSpec);
fclose(fileID_left(1));
fclose(fileID_left(2));
fclose(fileID_left(3));
fclose(fileID_left(4));
fclose(fileID_left(5));
fclose(fileID_left(6));
fclose(fileID_left(7));
fclose(fileID_left(8));
fclose(fileID_left(9));
fclose(fileID_right(1));
fclose(fileID_right(2));
fclose(fileID_right(3));
fclose(fileID_right(4));
fclose(fileID_right(5));
fclose(fileID_right(6));
fclose(fileID_right(7));
fclose(fileID_right(8));
fclose(fileID_right(9));

X = (0.0:0.05:0.4);
Y_left = [Y1_left,Y2_left,Y3_left,Y4_left,Y5_left,Y6_left,Y7_left,Y8_left,Y9_left];
Y_right = [Y1_right,Y2_right,Y3_right,Y4_right,Y5_right,Y6_right,Y7_right,Y8_right,Y9_right];

% Create figure
figure1 = figure;
% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');
set(axes1,'FontSize',17,'YColor','black');

% Create multiple lines using matrix input to plot,
% color('r','b','g','m','c'), marker('o','d','s','+','*')
yyaxis left
gca1 = boxplot(Y_left,X,'Colors','b','Symbol','b+');
set(findobj(gca1,'type','line'),'linew',2.0);
yyaxis right
gca2 = boxplot(Y_right,X,'Colors','r','Symbol','r+');
set(findobj(gca2,'type','line'),'linew',1.0);

% Create xlabel & ylabel
xlabel('Random Label Probability','FontSize',20,'Interpreter','latex');
yyaxis left
ylabel('Gradient Varience','FontSize',20,'Interpreter','latex','Color','b');
set(gca,'YColor','b')
yyaxis right
ylabel('Generalization Error','FontSize',20,'Interpreter','latex','Color','r');
set(gca,'YColor','r')

% Create title
title(titles,'FontSize',25,'FontWeight','bold');
box(axes1,'on');

hold(axes1,'off');