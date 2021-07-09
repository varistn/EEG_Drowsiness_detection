%% Author: Varis (Nam) Pornpatanapaisarnkul
% This code is for MSE 491 Project
%% Description
% This code is for calculating and simulating a real-time PSD for resting EEG data
% EEGLAB is used to obtain and pre-process some of the data 
% User can determine how many epoch to run and what channel to use
% All plots are autosave, comment them out if not desired 
%% Clear data
clc;clearvars;close all;
%% User inputs
% choose subject, from 1 to 10
subject = 1;
% the maximum number of epoch is 475 for window size of 5 secs, [-1 4]
number_epoch_desired = 100;
% choose 72 to use all channels
% choose 8 for selected channels
number_of_channel = 8;
% window size the the epoch window
window_size = [-1 4];
% real time window size
real_time_window = 50;
% automatically save graphs as image? yes or no
autosave_image = 'no';
% channel names and locations in the data
Fp1 = 1;  AF7 = 2;  AF3 = 3;  F1 = 4;   F3 = 5;   F5 = 6;   F7 = 6;   
FT7 = 8;  FC5 = 9;  FC3 = 10; FC1 = 11; C1 = 12;  C3 = 13;  C5 = 14;  
T7 = 15;  TP7 = 16; CP5 = 17; CP3 = 18; CP1 = 19; P1 = 20;  P3 = 21; 
P5 = 22;  P7 = 23;  P9 = 24;  PO7 = 25; PO3 = 26; O1 = 27;  Iz = 28; 
Oz = 29;  POz = 30; Pz = 31;  CPz = 32; FPz = 33; FP2 = 34; AF8 = 35; 
AF4 = 36; AFz = 37; Fz = 38;  F2 = 39;  F4 = 40;  F6 = 41;  F8 = 42;  
FT8 = 43; FC6 = 44; FC4 = 45; FC2 = 46; FCz = 47; Cz = 48;  C2 = 49;  
C4 = 50;  C6 = 51;  T8 = 52;  TP8 = 53; CP6 = 54; CP4 = 55; CP2 = 56;
P2 = 57;  P4 = 58;  P6 = 59;  P8 = 60;  P10 = 61; PO8 = 62; PO4 = 63; 
O2 = 64; 
% some reference nodes
M1 = 65;  M2 = 66;  NAS = 67; 
% EOG channels
LVEOG = 68; RVLOG = 69; LHEOG = 70; RHEOG = 71; NFPz = 72;

% selected channel names and locations in the data
if number_of_channel ~= 72
    selected_channel = [T7 T8 P7 P8 O1 O2 Oz Cz];
    selected_channel_name = {"T7","T8","P7","P8","O1","O2","Oz","Cz"};
end
% for outputting channel
if number_of_channel == 72
    desired_channel = Oz; % channel Oz
else
    desired_channel = 7; % channel Oz
end

%% DATA WITHOUT EEGLAB
%{
% Data Sourse
% http://clopinet.com/causality/data/nolte/
% Each data set is an EEG measurement of a subject with eyes closed using 19 
% channels according to the standard 10-20 system. The sampling rate is 256Hz. 
% If you divide a data set into blocks of 4 seconds (i.e. 1024 data points) 
% then each block is a continuous measurement which is cleaned of apparent 
% artefacts.

% subject 1
% fid=fopen('sub1.bin','r');
% data=reshape(fread(fid,'float'),[],19);
% subject 2
% fid=fopen('sub2.bin','r');
% data=reshape(fread(fid,'float'),[],19);

% figure
% plot(data)
% title('Raw data, 19 channels')
%}
%% Acquiring data with EEGLAB code
EEG.etc.eeglabvers = '2019.0'; % this tracks which version of EEGLAB is being used, you may ignore it
if subject == 1
    % subject 1
    EEG = pop_biosig('C:\Users\namto\Documents\MSE 491 - Special Topic in Mechatronic Systems Engineering\Project\EEG_Cat_Study5_Resting_S1.bdf');
elseif subject == 2
    % subject 2
    EEG = pop_biosig('C:\Users\namto\Documents\MSE 491 - Special Topic in Mechatronic Systems Engineering\Project\EEG_Cat_Study5_Resting_S2.bdf');
elseif subject == 3
    % subject 3
    EEG = pop_biosig('C:\Users\namto\Documents\MSE 491 - Special Topic in Mechatronic Systems Engineering\Project\EEG_Cat_Study5_Resting_S3.bdf');
elseif subject == 4
    % subject 4
    EEG = pop_biosig('C:\Users\namto\Documents\MSE 491 - Special Topic in Mechatronic Systems Engineering\Project\EEG_Cat_Study5_Resting_S4.bdf');
elseif subject == 5
    % subject 5
    EEG = pop_biosig('C:\Users\namto\Documents\MSE 491 - Special Topic in Mechatronic Systems Engineering\Project\EEG_Cat_Study5_Resting_S5.bdf');
elseif subject == 6
    % subject 6
    EEG = pop_biosig('C:\Users\namto\Documents\MSE 491 - Special Topic in Mechatronic Systems Engineering\Project\EEG_Cat_Study5_Resting_S6.bdf');
elseif subject == 7
    % subject 7
    EEG = pop_biosig('C:\Users\namto\Documents\MSE 491 - Special Topic in Mechatronic Systems Engineering\Project\EEG_Cat_Study5_Resting_S7.bdf');
elseif subject == 8
    % subject 8
    EEG = pop_biosig('C:\Users\namto\Documents\MSE 491 - Special Topic in Mechatronic Systems Engineering\Project\EEG_Cat_Study5_Resting_S8.bdf');
elseif subject == 9
    % subject 9
    EEG = pop_biosig('C:\Users\namto\Documents\MSE 491 - Special Topic in Mechatronic Systems Engineering\Project\EEG_Cat_Study5_Resting_S9.bdf');
elseif subject == 10
    % subject 10
    EEG = pop_biosig('C:\Users\namto\Documents\MSE 491 - Special Topic in Mechatronic Systems Engineering\Project\EEG_Cat_Study5_Resting_S10.bdf');
end
EEG.setname = 'RestingEEG';
EEG = eeg_checkset( EEG );
% lookup channel location
EEG = pop_chanedit(EEG, 'lookup','C:\\Users\\namto\\Documents\\MATLAB\\eeglab2019_0\\plugins\\dipfit\\standard_BESA\\standard-10-5-cap385.elp');
EEG = eeg_checkset( EEG );
%% Pre-processing
% filter 60Hz noise
figure(1)
EEG = pop_eegfiltnew(EEG, 'locutoff',59,'hicutoff',61,'revfilt',1,'plotfreqz',1);
EEG = eeg_checkset( EEG );
close;
% filter 1Hz
figure(2)
EEG = pop_eegfiltnew(EEG, 'locutoff',1,'plotfreqz',1);
EEG = eeg_checkset( EEG );
close;
% re-referencing, using average reference
EEG = pop_reref( EEG, []);
EEG = eeg_checkset( EEG );
%% Epoching data
%for first event
EEG_eyesopen = pop_epoch( EEG, {  '101'  },window_size, 'newname', 'Person1resting epochs 101', 'epochinfo', 'yes');
EEG_eyesopen = eeg_checkset( EEG_eyesopen );
% for second event
EEG_eyesclosed = pop_epoch( EEG, {  '201'  },window_size, 'newname', 'Person1resting epochs 201', 'epochinfo', 'yes');
EEG_eyesclosed = eeg_checkset( EEG_eyesclosed );
% remove baseline for the entire epoch
% EEG_eyesclosed = pop_rmbase( EEG_eyesclosed, [] ,[]);
% EEG_eyesclosed = eeg_checkset( EEG_eyesclosed );
% EEG_eyesopen = pop_rmbase( EEG_eyesopen, [] ,[]);
% EEG_eyesopen = eeg_checkset( EEG_eyesopen );
% run ICA, use fastica to reduce time
% EEG_eyesclosed = pop_runica(EEG_eyesclosed, 'icatype', 'fastica');
% EEG_eyesclosed = eeg_checkset( EEG_eyesclosed );
%% Auto artefacts rejection with ADJUST (doesn't work)
% pop_ADJUST_interface([] ,EEG_eyesclosed,[] );
% reject= fileread('report.txt');
%% Manual artefacts rejection (doesn't work)
%{
% check components eegplot
%pop_eegplot( EEG, 0, 1, 1);
% check components map
%pop_selectcomps(EEG, [1:71] );
% first rejection
% EEG = pop_subcomp( EEG, [1   2   3   6  12  16 21  28  44  53  71], 0);
% EEG.setname='Resting EEG epochs pruned with ICA';
% % rerun ICA
% EEG = pop_runica(EEG, 'icatype', 'fastica');
% EEG = eeg_checkset( EEG );
% second rejection
%pop_selectcomps(EEG, [1:60] );
% EEG = pop_subcomp( EEG, [6  15  16  26], 0);
% EEG.setname='Resting EEG epochs pruned with ICA 2nd';
% EEG = eeg_checkset( EEG );
% rerun ICA
% EEG = pop_runica(EEG, 'icatype', 'fastica');
% EEG = eeg_checkset( EEG );
% figure; pop_spectopo(EEG_eyesopen, 1, [0  508996.0938], 'EEG' , 'freqrange',[0 50],'electrodes','off');
% EEG = eeg_checkset( EEG );
%}
%% Preallocating array/initialized variable for epoch removal
total_epoch_removed_eyesopen = zeros(number_epoch_desired);
total_epoch_removed_eyesclosed = zeros(number_epoch_desired);
channel_removed_eyesclosed = zeros(number_of_channel);
channel_removed_eyesopen = zeros(number_of_channel);
epoch_removed_eyesopen = 0;
epoch_removed_counter = 0;
%% PSD calculations
% loop each epoch to simulate real time analysis
for j = 1:number_epoch_desired
    %% Info for FFT calc.
    % Sampling rate Hz
    Fs = 256;
    data_size_eyesclosed = length(EEG_eyesclosed.data(1,1:end,1));
    data_size_eyesopen = length(EEG_eyesopen.data(1,1:end,1));
    time = (0:data_size_eyesclosed-1)/Fs;
    desired_period = sum(abs(window_size));

    %%
    % WITH EEGLAB
    % get channel locations
    for i = 1:length(EEG.chanlocs)
        channel_loc{i,1} = EEG.chanlocs(i).labels;
    end
    
    %% FFT data
    % rearrange the matrix for selected channel
    if number_of_channel ~= 72
        for i = selected_channel
            % only take the selected data
            EEG_eyesclosed_zero(:,i) = EEG_eyesclosed.data(i,:,j);
            EEG_eyesopen_zero(:,i) = EEG_eyesopen.data(i,:,j);
        end
        % remove all the array of zeros
        % and reshape the the array
        EEG_eyesclosed_nonzero = nonzeros(EEG_eyesclosed_zero);
        EEG_eyesclosed_nonzero = reshape(EEG_eyesclosed_nonzero,data_size_eyesclosed,number_of_channel);
        EEG_eyesopen_nonzero = nonzeros(EEG_eyesopen_zero);
        EEG_eyesopen_nonzero = reshape(EEG_eyesopen_nonzero,data_size_eyesclosed,number_of_channel);
    end
    % FFT
    for i = 1:number_of_channel
        EEG_eyesclosed_fft(:,i) = fft(EEG_eyesclosed.data(i,:,j));
        EEG_eyesopen_fft(:,i) = fft(EEG_eyesopen.data(i,:,j));
    end
    NFFT = size(EEG_eyesclosed_fft,1);
    frequencies = Fs/2*linspace(0,1,floor(NFFT/2)+1);
    
    %% Power and amplitude
    % single-sided amplitude spectral
    % Calculate amplitude of the fft
    % Note: There is a scaling factor 1/NFFT
    % Note2: we usually don't multiply first element (DC component) 
    % and last element (NFFT/2+1) (Nyquist component) by 2
    for i = 1:number_of_channel
        P_test_eyesclosed = EEG_eyesclosed_fft(1:floor(NFFT/2)+1,i);
        P_test_eyesopen = EEG_eyesopen_fft(1:floor(NFFT/2)+1,i);
        P_eyesclosed(:,i) = abs(P_test_eyesclosed)/NFFT;
        P_eyesopen(:,i) = abs(P_test_eyesopen)/NFFT;
        P_test2_eyesclosed = P_eyesclosed(:,i);
        P_test2_eyesopen = P_eyesopen(:,i);
        P_test2_eyesclosed(2:end-1) = 2*P_test2_eyesclosed(2:end-1);
        P_test2_eyesopen(2:end-1) = 2*P_test2_eyesopen(2:end-1);
        P_eyesclosed(:,i) = P_test2_eyesclosed;
        P_eyesopen(:,i) = P_test2_eyesopen;
    end
    
    %% PSD
    % Calculate the power spectral density of the signals
    % Note: there is a scaling factor of 1/(Fs*NFFT) to compute the PSD
    for i = 1:number_of_channel
        P_test_eyesclosed = EEG_eyesclosed_fft(1:floor(NFFT/2)+1,i);
        P_test_eyesopen = EEG_eyesopen_fft(1:floor(NFFT/2)+1,i);
        PSD_eyesclosed(:,i) = abs(P_test_eyesclosed).^2/(Fs*NFFT);
        PSD_eyesopen(:,i) = abs(P_test_eyesopen).^2/(Fs*NFFT);
        P_test2_eyesclosed = PSD_eyesclosed(:,i);
        P_test2_eyesopen = PSD_eyesopen(:,i);
        P_test2_eyesclosed(2:end-1) = 2*P_test2_eyesclosed(2:end-1);
        P_test2_eyesopen(2:end-1) = 2*P_test2_eyesopen(2:end-1);
        PSD_eyesclosed(:,i) = P_test2_eyesclosed;       
        PSD_eyesopen(:,i) = P_test2_eyesopen;
    end

    % calculate total power from the signal in frequency domain
    for i = 1:number_of_channel
       totpower_eyesclosed(i) = sum(PSD_eyesclosed(:,i))*Fs/NFFT; 
       totpower_eyesopen(i) = sum(PSD_eyesopen(:,i))*Fs/NFFT; 
    end

    %% Frequency range
    % corresponding frequencies in the frequency vector and calculate
    % the power and the relative power
    % delta signal 1 to 3 HZ
    deltastart = find(frequencies == 1);
    deltaend = find(frequencies == 3);
    % theta signal 4 to 7 HZ
    thetastart = find(frequencies == 4);
    thetaend = find(frequencies == 7);
    % alpha signal 8 to 12 HZ
    alphastart = find(frequencies == 8);
    alphaend = find(frequencies == 12);
    % beta signal 13 to 28 HZ
    betastart = find(frequencies == 13);
    betaend = find(frequencies == 28);
    % gamma signal 29 to 100 HZ
    gammastart = find(frequencies == 29);
    gammaend = find(frequencies == 100);
    
    %% Channel_label
    if number_of_channel == 72
        channel_label = channel_loc;
    else
        channel_label = selected_channel_name;
    end
    
    %% Relative power calculations
    for i = 1:number_of_channel
        deltapower_eyesclosed(:,i) = sum(PSD_eyesclosed(deltastart:deltaend,i)*Fs/NFFT); %absolute power
        deltarelative_eyesclosed(:,i) = deltapower_eyesclosed(:,i)/totpower_eyesclosed(:,i)*100; %relative power
        thetapower_eyesclosed(:,i) = sum(PSD_eyesclosed(thetastart:thetaend,i)*Fs/NFFT); 
        thetarelative_eyesclosed(:,i) = thetapower_eyesclosed(:,i)/totpower_eyesclosed(:,i)*100; 
        alphapower_eyesclosed(:,i) = sum(PSD_eyesclosed(alphastart:alphaend,i)*Fs/NFFT); 
        alpharelative_eyesclosed(:,i) = alphapower_eyesclosed(:,i)/totpower_eyesclosed(:,i)*100;
        betapower_eyesclosed(:,i) = sum(PSD_eyesclosed(betastart:betaend,i)*Fs/NFFT); 
        betarelative_eyesclosed(:,i) = betapower_eyesclosed(:,1)/totpower_eyesclosed(:,i)*100; 
        gammapower_eyesclosed(:,i) = sum(PSD_eyesclosed(gammastart:gammaend,i)*Fs/NFFT); 
        gammarelative_eyesclosed(:,i) = gammapower_eyesclosed(:,i)/totpower_eyesclosed(:,i)*100; 
        
        deltapower_eyesopen(:,i) = sum(PSD_eyesopen(deltastart:deltaend,i)*Fs/NFFT); %absolute power
        deltarelative_eyesopen(:,i) = deltapower_eyesopen(:,i)/totpower_eyesopen(:,i)*100; %relative power
        thetapower_eyesopen(:,i) = sum(PSD_eyesopen(thetastart:thetaend,i)*Fs/NFFT); 
        thetarelative_eyesopen(:,i) = thetapower_eyesopen(:,i)/totpower_eyesopen(:,i)*100; 
        alphapower_eyesopen(:,i) = sum(PSD_eyesopen(alphastart:alphaend,i)*Fs/NFFT); 
        alpharelative_eyesopen(:,i) = alphapower_eyesopen(:,i)/totpower_eyesopen(:,i)*100;
        betapower_eyesopen(:,i) = sum(PSD_eyesopen(betastart:betaend,i)*Fs/NFFT); 
        betarelative_eyesopen(:,i) = betapower_eyesopen(:,1)/totpower_eyesopen(:,i)*100; 
        gammapower_eyesopen(:,i) = sum(PSD_eyesopen(gammastart:gammaend,i)*Fs/NFFT); 
        gammarelative_eyesopen(:,i) = gammapower_eyesopen(:,i)/totpower_eyesopen(:,i)*100; 

        %% Epoch removal
        % determine if there's an epoch to be removed eyesopen
        % sum of relative power should not exceed 100%
        sum_of_power_eyesopen = deltarelative_eyesopen(:,i)+thetarelative_eyesopen(:,i)+alpharelative_eyesopen(:,i)+betarelative_eyesopen(:,i)+gammarelative_eyesopen(:,i);
        sum_of_power_eyesclosed = deltarelative_eyesclosed(:,i)+thetarelative_eyesclosed(:,i)+alpharelative_eyesclosed(:,i)+betarelative_eyesclosed(:,i)+gammarelative_eyesclosed(:,i);
%         if sum_of_power_eyesopen > totpower_eyesopen(i)
        if sum_of_power_eyesopen > 100
            epoch_removed_counter = epoch_removed_counter+1;
            if epoch_removed_counter ~= 1
                previous_epoch_removed_eyesopen = epoch_removed_eyesopen;
            end
            epoch_removed_eyesopen = j;
            channel_removed_eyesopen(i) = i;
            total_epoch_removed_eyesopen(j) = j;
            if j < number_epoch_desired && j > 4
                % using linear interpolation
                % (higher order interpolation will be better)
%                 deltarelative_eyesopen(:,i) = 2*deltarelative_epoch_eyesopen(i,j-1) - deltarelative_epoch_eyesopen(i,j-2);           
%                 thetarelative_eyesopen(:,i) = 2*thetarelative_epoch_eyesopen(i,j-1) - thetarelative_epoch_eyesopen(i,j-2);
%                 alpharelative_eyesopen(:,i) = 2*alpharelative_epoch_eyesopen(i,j-1) - alpharelative_epoch_eyesopen(i,j-2);
%                 betarelative_eyesopen(:,i) = 2*betarelative_epoch_eyesopen(i,j-1) - betarelative_epoch_eyesopen(i,j-2);
%                 gammarelative_eyesopen(:,i) = 2*gammarelative_epoch_eyesopen(i,j-1) - gammarelative_epoch_eyesopen(i,j-2);
                % using polyfit and polyval
                % polynomial degrees
                n = 1;
                x = [j-4 j-3 j-2 j-1];
                y = [deltarelative_epoch_eyesopen(i,j-4) deltarelative_epoch_eyesopen(i,j-3) deltarelative_epoch_eyesopen(i,j-2) deltarelative_epoch_eyesopen(i,j-1)];
                interpo_eq = polyfit(x,y,n);
                deltarelative_eyesopen(:,i) = polyval(interpo_eq,j);
                y = [thetarelative_epoch_eyesopen(i,j-4) thetarelative_epoch_eyesopen(i,j-3) thetarelative_epoch_eyesopen(i,j-2) thetarelative_epoch_eyesopen(i,j-1)];
                interpo_eq = polyfit(x,y,n);
                thetarelative_eyesopen(:,i) = polyval(interpo_eq,j);
                y = [alpharelative_epoch_eyesopen(i,j-4) alpharelative_epoch_eyesopen(i,j-3) alpharelative_epoch_eyesopen(i,j-2) alpharelative_epoch_eyesopen(i,j-1)];
                interpo_eq = polyfit(x,y,n);
                alpharelative_eyesopen(:,i) = polyval(interpo_eq,j);
                y = [betarelative_epoch_eyesopen(i,j-4) betarelative_epoch_eyesopen(i,j-3) betarelative_epoch_eyesopen(i,j-2) betarelative_epoch_eyesopen(i,j-1)];
                interpo_eq = polyfit(x,y,n);
                betarelative_eyesopen(:,i) = polyval(interpo_eq,j);
                y = [gammarelative_epoch_eyesopen(i,j-4) gammarelative_epoch_eyesopen(i,j-3) gammarelative_epoch_eyesopen(i,j-2) gammarelative_epoch_eyesopen(i,j-1)];
                interpo_eq = polyfit(x,y,n);
                gammarelative_eyesopen(:,i) = polyval(interpo_eq,j);
            end
            fprintf("remove epoch number %d (eyesopen) %s\n",j,channel_label{i});
%         
%         elseif epoch_removed_counter ~= 1
%             if epoch_removed_eyesopen > 0 && j > previous_epoch_removed_eyesopen+1
%                 % try to avg the removed epoch in real time (doesn't work)    
%                 deltarelative_epoch_eyesopen(i,epoch_removed_eyesopen) = (deltarelative_epoch_eyesopen(i,previous_epoch_removed_eyesopen-1)+deltarelative_epoch_eyesopen(i,j-1))/2;
%             end
%         elseif epoch_removed_eyesopen > 0 && j > epoch_removed_eyesopen+1
%                 deltarelative_epoch_eyesopen(i,epoch_removed_eyesopen) = (deltarelative_epoch_eyesopen(i,epoch_removed_eyesopen-1)+deltarelative_epoch_eyesopen(i,j-1))/2;
        end
       
        % determine if there's an epoch to be removed eyesclosed
%         if sum_of_power_eyesclosed > totpower_eyesclosed(i)
        if sum_of_power_eyesclosed > 100
            channel_removed_eyesclosed(i) = i;
            total_epoch_removed_eyesopen(j) = j;
            if j < number_epoch_desired && j > 4
                % using polyfit and polyval
                % polynomial degrees
                n = 1;
                x = [j-4 j-3 j-2 j-1];
                y = [deltarelative_epoch_eyesclosed(i,j-4) deltarelative_epoch_eyesclosed(i,j-3) deltarelative_epoch_eyesclosed(i,j-2) deltarelative_epoch_eyesclosed(i,j-1)];
                interpo_eq = polyfit(x,y,n);
                deltarelative_eyesclosed(:,i) = polyval(interpo_eq,j);
                y = [thetarelative_epoch_eyesclosed(i,j-4) thetarelative_epoch_eyesclosed(i,j-3) thetarelative_epoch_eyesclosed(i,j-2) thetarelative_epoch_eyesclosed(i,j-1)];
                interpo_eq = polyfit(x,y,n);
                thetarelative_eyesclosed(:,i) = polyval(interpo_eq,j);
                y = [alpharelative_epoch_eyesclosed(i,j-4) alpharelative_epoch_eyesclosed(i,j-3) alpharelative_epoch_eyesclosed(i,j-2) alpharelative_epoch_eyesclosed(i,j-1)];
                interpo_eq = polyfit(x,y,n);
                alpharelative_eyesclosed(:,i) = polyval(interpo_eq,j);
                y = [betarelative_epoch_eyesclosed(i,j-4) betarelative_epoch_eyesclosed(i,j-3) betarelative_epoch_eyesclosed(i,j-2) betarelative_epoch_eyesclosed(i,j-1)];
                interpo_eq = polyfit(x,y,n);
                betarelative_eyesclosed(:,i) = polyval(interpo_eq,j);
                y = [gammarelative_epoch_eyesclosed(i,j-4) gammarelative_epoch_eyesclosed(i,j-3) gammarelative_epoch_eyesclosed(i,j-2) gammarelative_epoch_eyesclosed(i,j-1)];
                interpo_eq = polyfit(x,y,n);
                gammarelative_eyesclosed(:,i) = polyval(interpo_eq,j);
            end
            fprintf("remove epoch number %d (eyesclosed) %s\n",j,channel_label{i});
        end
        %% Storing each epoch
        deltarelative_epoch_eyesclosed(i,j) = deltarelative_eyesclosed(:,i);
        thetarelative_epoch_eyesclosed(i,j) = thetarelative_eyesclosed(:,i);
        alpharelative_epoch_eyesclosed(i,j) = alpharelative_eyesclosed(:,i);
        betarelative_epoch_eyesclosed(i,j) = betarelative_eyesclosed(:,i);
        gammarelative_epoch_eyesclosed(i,j) = gammarelative_eyesclosed(:,i);

        deltarelative_epoch_eyesopen(i,j) = deltarelative_eyesopen(:,i);
        thetarelative_epoch_eyesopen(i,j) = thetarelative_eyesopen(:,i);
        alpharelative_epoch_eyesopen(i,j) = alpharelative_eyesopen(:,i);
        betarelative_epoch_eyesopen(i,j) = betarelative_eyesopen(:,i);
        gammarelative_epoch_eyesopen(i,j) = gammarelative_eyesopen(:,i);

    end
    
    %% Ploting 
    % box representing the region of each frequency region
    x_delta = [1 1 3 3];
    y_delta = [0 150 150 0];
    x_theta = [4 4 7 7];
    y_theta = [0 150 150 0];
    x_alpha = [8 8 12 12];
    y_alpha = [0 150 150 0];
    x_beta = [13 13 28 28];
    y_beta = [0 150 150 0];
    x_gamma = [29 29 100 100];
    y_gamma = [0 150 150 0];
    
    fprintf('Epoch %d\n',j);
    
    for i = 1:number_of_channel
        % only plot PSD for the first epoch
        if j == 1 
            figure
            % plot for eyesclosed
            plot(frequencies,PSD_eyesclosed(:,i),'Linewidth',1.5);
            xlim([0 50]);
            ylim([0 150]);
            xlabel('Frequency (Hz)'); 
            ylabel('Power Spectral Density (\muV^2/Hz)');
            title(sprintf('Frequency plot %s',channel_label{i}));
            grid on
            % texts showing the relative power
            text(1,100,strcat(['Delta=' num2str(deltarelative_eyesclosed(:,i)) '%']));
            text(4,75,strcat(['Theta=' num2str(thetarelative_eyesclosed(:,i)) '%']));
            text(8,50,strcat(['Alpha=' num2str(alpharelative_eyesclosed(:,i)) '%']));
            text(15,25,strcat(['Beta=' num2str(betarelative_eyesclosed(:,i)) '%']));
            text(33,25,strcat(['Gamma=' num2str(gammarelative_eyesclosed(:,i)) '%']));

            % make sure the color is pretty
            d=patch(x_delta,y_delta,'blue');
            d.FaceAlpha=0.2;
            t=patch(x_theta,y_theta,'green');
            t.FaceAlpha=0.2;
            a=patch(x_alpha,y_alpha,'red');
            a.FaceAlpha=0.2;
            b=patch(x_beta,y_beta,'yellow');
            b.FaceAlpha=0.2;
            g=patch(x_gamma,y_gamma,'cyan');
            g.FaceAlpha=0.2;
            
            if strcmp(autosave_image, 'yes')
                filename = sprintf('Autosaved PSD channel %s.png',channel_label{i});
                saveas(gcf,filename)
            end
            
            % comparing PSD for eyesclosed vs eyesopen
%             figure
%             hold on
%             plot(frequencies,PSD_eyesclosed(:,i),'r','Linewidth',1.5);
%             plot(frequencies,PSD_eyesopen(:,i),'b','Linewidth',1.5);
%             xlim([0 20]);
%             ylim([0 50]);
%             xlabel('Frequency (Hz)'); 
%             ylabel('Power Spectral Density (\muV^2/Hz)');
%             legend('eyesclosed','eyesopen','alpha band');
%             title(sprintf('Frequency plot %s',channel_label{i}));
%             grid on
%             a=patch(x_alpha,y_alpha,'red');
%             a.FaceAlpha=0.2;
        end
        
        if deltarelative_eyesclosed(i) > thetarelative_eyesclosed(i) && deltarelative_eyesclosed(i) > alpharelative_eyesclosed(i) && deltarelative_eyesclosed(i) > betarelative_eyesclosed(i) && deltarelative_eyesclosed(i) > gammarelative_eyesclosed(i)
            fprintf('Delta is dominant (%d%%) in %s\n',round(deltarelative_eyesclosed(i)),channel_label{i});
        end
        if thetarelative_eyesclosed(i) > deltarelative_eyesclosed(i) && thetarelative_eyesclosed(i) > alpharelative_eyesclosed(i) && thetarelative_eyesclosed(i) > betarelative_eyesclosed(i) && thetarelative_eyesclosed(i) > gammarelative_eyesclosed(i)
            fprintf('Theta is dominant (%d%%) in %s\n',round(thetarelative_eyesclosed(i)),channel_label{i});
        end
        if alpharelative_eyesclosed(i) > deltarelative_eyesclosed(i) && alpharelative_eyesclosed(i) > thetarelative_eyesclosed(i) && alpharelative_eyesclosed(i) > betarelative_eyesclosed(i) && alpharelative_eyesclosed(i) > gammarelative_eyesclosed(i)
            fprintf('Alpha is dominant (%d%%) in %s\n',round(alpharelative_eyesclosed(i)),channel_label{i});
        end
        if betarelative_eyesclosed(i) > deltarelative_eyesclosed(i) && betarelative_eyesclosed(i) > thetarelative_eyesclosed(i) && betarelative_eyesclosed(i) > alpharelative_eyesclosed(i) && betarelative_eyesclosed(i) > gammarelative_eyesclosed(i)
            fprintf('Beta is dominant (%d%%) in %s\n',round(betarelative_eyesclosed(i)),channel_label{i});
        end
        if gammarelative_eyesclosed(i) > deltarelative_eyesclosed(i) && gammarelative_eyesclosed(i) > thetarelative_eyesclosed(i) && gammarelative_eyesclosed(i) > alpharelative_eyesclosed(i) && gammarelative_eyesclosed(i) > betarelative_eyesclosed(i)
            fprintf('Gamma is dominant (%d%%) in %s\n',round(gammarelative_eyesclosed(i)),channel_label{i});
        end
    end
    
    %% "Real-time" plot
%     if j > 0 && j <= 50
        hold on
        if j <= real_time_window
            figure(1)
            axis([0 real_time_window 0 100])
        elseif j > real_time_window && j <= real_time_window*2
            figure(1)
            axis([real_time_window real_time_window*2 0 100])
        elseif j > real_time_window*2 && j <= real_time_window*3
            figure(1)
            axis([real_time_window*2 real_time_window*3 0 100])    
        end
%         plot(j,deltarelative_eyesclosed(1,desired_channel),'k--o')
        plot(j,thetarelative_eyesclosed(1,desired_channel),'r--o')
        plot(j,alpharelative_eyesclosed(1,desired_channel),'b--o','Linewidth',1.5)
%         plot(j,betarelative_eyesclosed(1,desired_channel),'m--o')
%         plot(j,gammarelative_eyesclosed(1,desired_channel),'c--o')
        hold off
        title(sprintf('Alpha and Theta power for %d epochs',number_epoch_desired));
%         ylim([0 100])
        xlabel('Epoch')
        ylabel('Relative Power in %')
        % legend intentionally swap
        legend('Alpha','Theta')
%         legend('Delta','Theta','Alpha','Beta','Gamma')
        grid on
        drawnow limitrate;
        if strcmp(autosave_image, 'yes')
            if j == real_time_window || j == real_time_window*2 || j == real_time_window*3
                filename = sprintf('Autosaved %d to %d epoch.png',j-real_time_window, j);
                saveas(gcf,filename)
            end
        end
    drawnow
end
%% For ploting all epochs
j = 1:number_epoch_desired;

%% Show removed epochs
total_epoch_removed_eyesopen = nonzeros(total_epoch_removed_eyesopen);
total_epoch_removed_eyesclosed = nonzeros(total_epoch_removed_eyesclosed);
channel_removed_eyesopen = nonzeros(channel_removed_eyesopen);
channel_removed_eyesclosed = nonzeros(channel_removed_eyesclosed);

if length(total_epoch_removed_eyesopen) > 0
    disp(['Removed epoch number (eyesopen)']);
    disp(total_epoch_removed_eyesopen')
    % interpolate using avg (not real-time!!)
%     k = epoch_removed_eyesopen;
%     l = channel_removed_eyesopen;
%     % use & because number_epoch_desired is a vector
%     if k < number_epoch_desired & k > 1
%         deltarelative_epoch_eyesopen(l,k) = (deltarelative_epoch_eyesopen(l,k-1)+deltarelative_epoch_eyesopen(l,k+1))/2;
%         thetarelative_epoch_eyesopen(l,k) = (thetarelative_epoch_eyesopen(l,k-1)+thetarelative_epoch_eyesopen(l,k+1))/2;
%         alpharelative_epoch_eyesopen(l,k) = (alpharelative_epoch_eyesopen(l,k-1)+alpharelative_epoch_eyesopen(l,k+1))/2;
%         betarelative_epoch_eyesopen(l,k) = (betarelative_epoch_eyesopen(l,k-1)+betarelative_epoch_eyesopen(l,k+1))/2;
%         gammarelative_epoch_eyesopen(l,k) = (gammarelative_epoch_eyesopen(l,k-1)+gammarelative_epoch_eyesopen(l,k+1))/2;
%     end
end

if length(total_epoch_removed_eyesclosed) > 0
    disp('Removed epoch number (eyesclosed)');
    disp(total_epoch_removed_eyesclosed')
    % interpolate using avg (not real-time!!)
%     k = epoch_removed_eyesclosed;
%     l = channel_removed_eyesclosed;
%     % use & because number_epoch_desired is a vector
%     if k < number_epoch_desired & k > 0
%         deltarelative_epoch_eyesclosed(l,k) = (deltarelative_epoch_eyesclosed(l,k-1)+deltarelative_epoch_eyesclosed(l,k+1))/2;
%         thetarelative_epoch_eyesclosed(l,k) = (thetarelative_epoch_eyesclosed(l,k-1)+thetarelative_epoch_eyesclosed(l,k+1))/2;
%         alpharelative_epoch_eyesclosed(l,k) = (alpharelative_epoch_eyesclosed(l,k-1)+alpharelative_epoch_eyesclosed(l,k+1))/2;
%         betarelative_epoch_eyesclosed(l,k) = (betarelative_epoch_eyesclosed(l,k-1)+betarelative_epoch_eyesclosed(l,k+1))/2;
%         gammarelative_epoch_eyesclosed(l,k) = (gammarelative_epoch_eyesclosed(l,k-1)+gammarelative_epoch_eyesclosed(l,k+1))/2;
%     end
end

%% Threshold Calc.
threshold_eyesclosed = mean(alpharelative_epoch_eyesclosed(desired_channel,:));
threshold_eyesopen = mean(alpharelative_epoch_eyesopen(desired_channel,:));
threshold = (threshold_eyesclosed + threshold_eyesopen)/2;
fprintf('for %d epochs, calculated Alpha threshold is %f\n',number_epoch_desired,threshold);

%% Uncomment the line below to comment out all plot
% %{
%% All waves
% eyesclosed
figure
hold on
plot(j,deltarelative_epoch_eyesclosed(desired_channel,:),'k--o')
plot(j,thetarelative_epoch_eyesclosed(desired_channel,:),'r--o')
plot(j,alpharelative_epoch_eyesclosed(desired_channel,:),'b-o','Linewidth',1.5)
plot(j,betarelative_epoch_eyesclosed(desired_channel,:),'m--o')
plot(j,gammarelative_epoch_eyesclosed(desired_channel,:),'c--o')
xlabel('Epoch')
ylabel('Relative Power %')
title(sprintf('Eyesclosed channel %s',channel_label{desired_channel}));
legend('Delta','Theta','Alpha','Beta','Gamma');
grid on
if strcmp(autosave_image, 'yes')
    saveas(gcf,'Autosaved all waves eyeclosed.png')
end
hold off

% eyesopen
figure
hold on
plot(j,deltarelative_epoch_eyesopen(desired_channel,:),'k--o')
plot(j,thetarelative_epoch_eyesopen(desired_channel,:),'r--o')
plot(j,alpharelative_epoch_eyesopen(desired_channel,:),'b-o','Linewidth',1.5)
plot(j,betarelative_epoch_eyesopen(desired_channel,:),'m--o')
plot(j,gammarelative_epoch_eyesopen(desired_channel,:),'c--o')
xlabel('Epoch')
ylabel('Relative Power %')
title(sprintf('Eyesopen channel %s',channel_label{desired_channel}));
legend('Delta','Theta','Alpha','Beta','Gamma');
grid on
if strcmp(autosave_image, 'yes')
    saveas(gcf,'Autosaved all waves eyesopen.png')
end
hold off

%% Alpha eyes closed vs open
figure
hold on
% threshold is calculated by taking the mean of the alpha when eyesclose
% and eyesopen
% sub 1 
% threshold = 25.741684;
yline(threshold,'k--','Linewidth',1.5);
plot(j,alpharelative_epoch_eyesclosed(desired_channel,:),'b-o','Linewidth',1)
plot(j,alpharelative_epoch_eyesopen(desired_channel,:),'r-o','Linewidth',1)
xlabel('Epoch')
ylabel('Relative Power %')
title(sprintf('Alpha eyesclose vs eyesopen %s',channel_label{desired_channel}));
legend('Threshold','Alpha eyesclosed','Alpha eyesopen');
grid on
if strcmp(autosave_image, 'yes')
    saveas(gcf,'Autosaved alpha eyes closed vs open.png')
end
hold off

%% Alpha vs Theta
% eyesclosed
figure
hold on
plot(j,alpharelative_epoch_eyesclosed(desired_channel,:),'b-o','Linewidth',1)
plot(j,thetarelative_epoch_eyesclosed(desired_channel,:),'r-o','Linewidth',1)
title(sprintf('Eyeclosed %s',channel_label{desired_channel}));
xlabel('Epoch')
ylabel('Relative Power %')
legend('Alpha','Theta')
grid on
a=patch([15 15 25 25],[0 50 50 0],'red');
a.FaceAlpha=0.2;
if strcmp(autosave_image, 'yes')
    saveas(gcf,'Autosaved alpha vs theta eyesclosed.png')
end
hold off

% eyesopen
figure
hold on
plot(j,alpharelative_epoch_eyesopen(desired_channel,:),'b-o','Linewidth',1)
plot(j,thetarelative_epoch_eyesopen(desired_channel,:),'r-o','Linewidth',1)
title(sprintf('Eyesopen %s',channel_label{desired_channel}));
xlabel('Epoch')
ylabel('Relative Power %')
legend('Alpha','Theta');
grid on
if strcmp(autosave_image, 'yes')
    saveas(gcf,'Autosaved alpha vs theta eyesopen.png')
end
hold off

%% Average Alpha and Theta
% calculate of the average of all channel
avg_theta_eyesopen=mean(thetarelative_epoch_eyesopen);
avg_alpha_eyesopen=mean(alpharelative_epoch_eyesopen);
avg_theta_eyesclosed=mean(thetarelative_epoch_eyesclosed);
avg_alpha_eyesclosed=mean(alpharelative_epoch_eyesclosed);

% eyesclosed
figure
hold on
plot(j,avg_alpha_eyesclosed,'b-o','Linewidth',1)
plot(j,avg_theta_eyesclosed,'r-o','Linewidth',1)
title('Eyesclosed all channel Avg')
xlabel('Epoch')
ylabel('Relative Power %')
legend('Alpha','Theta');
if strcmp(autosave_image, 'yes')
    saveas(gcf,'Autosaved alpha vs theta avg eyesclosed.png')
end
grid on
hold off

% eyesopen
figure
hold on
plot(j,avg_alpha_eyesopen,'b-o','Linewidth',1)
plot(j,avg_theta_eyesopen,'r-o','Linewidth',1)
title('Eyesopen all channel Avg')
xlabel('Epoch')
ylabel('Relative Power %')
legend('Alpha','Theta');
grid on
if strcmp(autosave_image, 'yes')
    saveas(gcf,'Autosaved alpha vs theta avg eyesopen.png')
end
hold off
%}
%% GIF gen. (currently not working)
% h = figure;
% axis tight manual % this ensures that getframe() returns a consistent size
% filename = 'testAnimated.gif';
% for j = 1:number_epoch_desired;
% %     hold on
%     plot(j,thetarelative_epoch_eyesclosed(7,j),'r--o')
% %     plot(j,alpharelative_epoch_eyesclosed(7,j),'b--o')
%     xlim([0 100])
%     drawnow 
%     % Capture the plot as an image 
% %     frame = getframe(h); 
% %     im = frame2im(frame); 
% %     [imind,cm] = rgb2ind(im,256); 
% %     hold off
%     % Write to the GIF File 
%     frame = getframe(h);
%     im{j} = frame2im(frame);
%     filename = 'testAnimated.gif'; % Specify the output file name
%     for idx = 1:nImages
%         [A,map] = rgb2ind(im{idx},256);
%         if idx == 1
%             imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',1);
%         else
%             imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',1);
%         end
%     end
% end

%% MATLAB Color ref.
% b blue
% g green
% r red
% c cyan
% m magenta
% y yellow
% k black
% w white