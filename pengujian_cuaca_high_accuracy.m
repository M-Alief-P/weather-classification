function pengujian_cuaca_high_accuracy()
    clc; clear; close all; warning off all;

    % 1. KONFIGURASI
    main_folder = 'Dataset'; 
    data_type = 'Test';
    
    % Loading model hasil pelatihan yang sudah dioptimasi
    if ~exist('net_cuaca_v2.mat', 'file')
        error('File net_cuaca_v2.mat tidak ditemukan! Jalankan pelatihan_cuaca_high_accuracy dulu.');
    end
    load net_cuaca_v2.mat; % Memuat 'net', 'settings_input', dan 'classList'
    
    data_uji = [];
    target_uji = [];
    
    disp('--- MEMULAI PENGUJIAN (OPTIMASI TINGGI) ---');
    
    % 2. LOOPING SETIAP KELAS
    for k = 1:length(classList)
        className = classList{k};
        folderPath = fullfile(main_folder, data_type, className);
        files = dir(fullfile(folderPath, '*.jpg'));
        
        if isempty(files)
            disp(['Peringatan: Folder kosong ' className]);
            continue;
        end
        
        disp(['Menguji Kelas: ' className ' (' num2str(numel(files)) ' gambar)']);
        
        for n = 1:numel(files)
            try
                img = imread(fullfile(folderPath, files(n).name));
                img = imresize(img, [256 256]);
                
                % Penanganan format citra
                if size(img,3) == 3
                    gray = rgb2gray(img);
                    rgb_img = img;
                else
                    gray = img;
                    rgb_img = cat(3, img, img, img);
                end
                
                % --- EKSTRAKSI FITUR (Harus sama dengan Pelatihan) ---
                MeanR = mean2(rgb_img(:,:,1));
                MeanG = mean2(rgb_img(:,:,2));
                MeanB = mean2(rgb_img(:,:,3));
                
                % Fitur Tekstur (GLCM)
                glcm = graycomatrix(gray, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
                stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
                
                CiriCON = mean(stats.Contrast);
                CiriCOR = mean(stats.Correlation);
                CiriASM = mean(stats.Energy);
                CiriIDM = mean(stats.Homogeneity);
                
                % Fitur Entropy
                h = imhist(gray); h = h / sum(h);
                CiriENTR = -sum(h(h>0) .* log2(h(h>0)));
                
                fitur_temp = [MeanR; MeanG; MeanB; CiriENTR; CiriASM; CiriCON; CiriCOR; CiriIDM];
                data_uji = [data_uji, fitur_temp];
                target_uji = [target_uji, k]; 
                
            catch
                continue;
            end
        end
    end

    % --- 3. PROSES PREDIKSI (KUNCI AKURASI) ---
    
    % WAJIB: Normalisasi data uji menggunakan parameter dari data latih
    data_uji_norm = mapminmax('apply', data_uji, settings_input);
    
    % Simulasi JST
    output_vec = net(data_uji_norm);
    
    % Mengonversi output vektor probabilitas menjadi indeks kelas (1, 2, atau 3)
    output_class = vec2ind(output_vec);

    % --- 4. EVALUASI DAN METRIK ---
    confMat = confusionmat(target_uji, output_class);
    numClasses = length(classList);
    
    % Tampilkan Performa Per Kelas
    fprintf('\n--- HASIL EVALUASI PER KELAS ---\n');
    for i = 1:numClasses
        TP = confMat(i,i);
        FP = sum(confMat(:,i)) - TP;
        FN = sum(confMat(i,:)) - TP;

        precision = TP / (TP + FP);
        recall = TP / (TP + FN);
        f1 = 2 * (precision * recall) / (precision + recall);

        fprintf('Kelas [%s] -> Precision: %.2f, Recall: %.2f, F1-Score: %.2f\n', ...
            classList{i}, precision, recall, f1);
    end
    
    % Visualisasi Confusion Matrix
    figure('Name', 'Confusion Matrix Testing', 'Color', 'w');
    confusionchart(confMat, classList, 'Title', 'Confusion Matrix Testing (Optimized)');
    
    accuracy = sum(diag(confMat)) / sum(confMat(:)) * 100;
    disp('--------------------------------');
    disp(['Akurasi Total Pengujian: ' num2str(accuracy, '%.2f') '%']);
    disp('--------------------------------');
end