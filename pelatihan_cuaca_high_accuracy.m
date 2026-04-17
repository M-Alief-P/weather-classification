function pelatihan_cuaca_high_accuracy()
    clc; clear; close all; warning off all;

    % 1. KONFIGURASI FOLDER
    main_folder = 'Dataset'; 
    data_type = 'Train';
    classList = {'Cloudy', 'Rain', 'Shine'}; 
    data_latih = [];
    target_latih = [];
    
    disp('--- MEMULAI PROSES PELATIHAN (OPTIMASI TINGGI) ---');
    
    for k = 1:length(classList)
        className = classList{k};
        folderPath = fullfile(main_folder, data_type, className);
        files = dir(fullfile(folderPath, '*.jpg'));
        
        if isempty(files), continue; end
        disp(['Memproses Kelas: ' className ' (' num2str(numel(files)) ' gambar)']);
        
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
                
                % --- EKSTRAKSI FITUR (Sama seperti sebelumnya) ---
                MeanR = mean2(rgb_img(:,:,1));
                MeanG = mean2(rgb_img(:,:,2));
                MeanB = mean2(rgb_img(:,:,3));
                
                % GLCM Optimization
                glcm = graycomatrix(gray, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
                stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
                
                % Ambil rata-rata dari 4 sudut
                CiriCON = mean(stats.Contrast);
                CiriCOR = mean(stats.Correlation);
                CiriASM = mean(stats.Energy);
                CiriIDM = mean(stats.Homogeneity);
                
                % Entropy Histogram
                h = imhist(gray); h = h / sum(h);
                CiriENTR = -sum(h(h>0) .* log2(h(h>0)));
                
                fitur_temp = [MeanR; MeanG; MeanB; CiriENTR; CiriASM; CiriCON; CiriCOR; CiriIDM];
                data_latih = [data_latih, fitur_temp];
                target_latih = [target_latih, k]; 
                
            catch
                continue;
            end
        end
    end

    % --- TAHAP OPTIMASI JST ---
    
    % 2. Normalisasi Data (Kunci Akurasi Tinggi)
    % Menskalakan semua fitur ke rentang [-1, 1] agar bobot JST stabil
    [input_norm, settings_input] = mapminmax(data_latih, -1, 1);
    
    % 3. One-Hot Encoding Target
    % Mengubah target [1, 2, 3] menjadi matriks biner (e.g., [1;0;0])
    target_vec = ind2vec(target_latih);
    
    % 4. Membuat Pattern Recognition Network
    % Menggunakan patternnet (pengganti newff yang lebih optimal untuk klasifikasi)
    net = patternnet(30); % Menambah neuron menjadi 30 untuk kapasitas belajar lebih besar
    
    % Konfigurasi Pelatihan
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-7;
    net.trainParam.lr = 0.01;
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 10/100;

    % 5. Proses Training
    [net, tr] = train(net, input_norm, target_vec);
    
    % Simpan Model dan Parameter Normalisasi
    save net_cuaca_v2.mat net settings_input classList;
    
    % 6. Evaluasi
    output_vec = net(input_norm);
    output_class = vec2ind(output_vec);
    
    confMat = confusionmat(target_latih, output_class);
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
    
    accuracy = sum(diag(confMat)) / sum(confMat(:)) * 100;
    
    disp(['Akurasi Training Setelah Optimasi: ' num2str(accuracy) '%']);
end