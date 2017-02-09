%-------------------------------------------------------
%   LECTURA DE DATOS DE ENTRENAMIENTO
%-------------------------------------------------------
clc
clear all %#ok<*CLALL>
RGB1 = imread('c_0.png');       %Ejemplo inicial Clase "Circulo"
RGB2 = imread('t_0.png');       %Ejemplo inicial Clase "Tache/Cruz"
I = rgb2gray(RGB1);             %Convierte a matriz B/N
G = rgb2gray(RGB2);             %Convierte a matriz B/N
I = double(I(:));               %Convierte a vector analizable (1 sample)
G = double(G(:));               %Convierte a vector analizable (1 sample)
entradas = [I G];
salidas = [0 1];
data = ['Circulo';'Tache  '];
%Lectura de los 45 ejemplos de entrenamiento de los círculos
for i = 1:45
    A = imread(['c_',num2str(i),'.png']);
    A = rgb2gray(A);
    A = double(A(:));
    entradas = [entradas A]; %#ok<*AGROW>
    salidas = [salidas 0];
    data = [data;'Circulo'];
end
%Lectura de los 45 ejemplos de entrenamiento de las cruces
for i = 1:45
    A = imread(['t_',num2str(i),'.png']);
    A = rgb2gray(A);
    A = double(A(:));
    entradas = [entradas A];
    salidas = [salidas 1];
    data = [data;'Tache  '];
end
celldata = cellstr(data);

%-------------------------------------------------------
%   CREACIÓN Y CONFIGURACIÓN DE LA RNA
%-------------------------------------------------------
%Declaración de la red neuronal multicapa
%net = cascadeforwardnet(10,'traingd');
%net = fitnet(10,'traingd','mse');
%net = feedforwardnet(10,'traingd');        %También se puede usar esta
net = patternnet([10,10],'traingd','mse');  %traingd es para usar gradiente descendiente con backpropagation
net.layers{1}.transferFcn = 'tansig';       %Función de transferencia de la primera capa en tangente
net.layers{2}.transferFcn = 'logsig';       %Función de transferencia de la primera capa en logaritmo
net.trainParam.epochs = 2000;               %Epochs de entrenamiento

[net,entrenamiento] = train(net,entradas,salidas);
resultados = net(entradas);
errores = gsubtract(salidas,resultados);
rendimiento = perform(net,salidas,resultados);
view(net)
figure, plotroc(salidas,resultados)

%-------------------------------------------------------
%   SIMULACIÓN, CLASIFICACIÓN Y RESULTADOS
%-------------------------------------------------------
RGB3 = imread('circle.png');    %Prueba de simulación
H = rgb2gray(RGB3);
H = double(H(:));
resultado = sim(net,H);
fprintf('Resultados: %.4f\n',resultado);
if(resultado<0.5)
    disp('Es un circulo');
else
    disp('Es una tache');
end
pesos = net.iw{1,1};
bias = net.b{1};

%-------------------------------------------------------
%   ESPECIFICIDAD, SENSIBILIDAD Y ROC
%-------------------------------------------------------
[c,cm,ind,per] = confusion(salidas,resultados);
FNR = cm(1,2)/sum(cm(:,2));
FPR = cm(2,1)/sum(cm(:,1));
TPR = cm(1,1)/sum(cm(:,1));
TNR = cm(2,2)/sum(cm(:,2));
Sensibilidad = cm(1,1)/sum(cm(1,:));
Especificidad = cm(2,2)/sum(cm(2,:));
[X,Y,T,AUC] = perfcurve(celldata,resultados,'Tache  ');
fprintf('FNR: %.4f | FPR: %.4f | TPR: %.4f | TNR: %.4f\n',FNR,FPR,TPR,TNR);
fprintf('Sensibilidad: %.4f\n',Sensibilidad);
fprintf('Especificidad: %.4f\n',Especificidad);
fprintf('El valor del AUC es: %.4f\n',AUC);


