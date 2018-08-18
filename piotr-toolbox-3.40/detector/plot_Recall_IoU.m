%% ANMERKUNG: Vorab bitte einmal eine Dateu mit gespeicherten Evaluationsvariablen oeffnen!!! ('...WsVariables.mat')




%% Variablen deklarieren
% Basispfad zur gewuenschten MATLAB-Evaluationsdatei
% 'MATLAB_comp4_detectionBBs_test_person.txt' einfuegen!!!
base_path = ...
    '/media/gueste/TrekStor/Masterarbeit/Ausgelagerte_Trainings/2_train-all-T_D7/2_Tr16-2_OF_D7_320x320/evalOutput/singleScale/KAIST_2_Tr16-2_OF_D7_320x320_iter_210/detections_for_matlab';

% Konstanten
plot_von = .5;
plot_bis = .99;
teilung  = .005;


%% Neue Pfade Berechnen
gtDir_NEU = sprintf('%s/data/KAIST/data-kaist/test-all/annotations', getenv('HOME'));
var_load   = sprintf('%s/%s', base_path, 'MATLAB_comp4_detectionBBs_test_person-all_WsVariables.mat');
bbsNm_NEU = sprintf('%s/%s', base_path, 'MATLAB_comp4_detectionBBs_test_person.txt');


%% Daten Laden
load(var_load);
[gt_orig,dt_orig] = bbGt('loadAll',gtDir_NEU,bbsNm_NEU,pLoad);  % gt := gt0, dt := dt0


%% Berechnung starten
iou_thr = [plot_von: teilung :plot_bis];
recall = zeros(0,size(iou_thr,2));
for i = 1:size(iou_thr,2)
    [gt_neu,dt_neu] = bbGt('evalRes',gt_orig,dt_orig,iou_thr(i),0);
    [rec,prec,scorePR,refPrec] = bbGt('compRoc',gt_neu,dt_neu,0,ref);
    
    % Letzten Wert speichern
    recall(i) = rec(end);
end


%% Plot erzeugen
f = figure(show);
%f.Position(3) = 1000; % figure breiter machen
plot(iou_thr, recall * 100, 'b', 'LineWidth', 2);
axis([plot_von 1 0 100]); grid on;
title("Recall w.r.t. Intersection over Union");
xlabel("Intersection over Union"); ylabel("Recall [%]");
set(gca,'FontSize',16)


%% Plot speichern
savefig([base_path '/Recall_IoU' '.fig']);
