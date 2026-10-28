
 // =====================================================
 // SECTION 4
 // Harmonic extraction, THD and filter switch selection
 // =====================================================

 // Sampled PCC waveform
 x = pcc_data.values;
 x = matrix(x, 1, -1);

 // Sampling parameters
 fs = 10000;
 N = size(x, "*");

 // Fundamental frequency
 f1 = 50;

 // =====================================================
 // 1. FFT
 // =====================================================

 Y = fft(x);

 // Frequency axis
 f = (0:N-1) * (fs/N);

 // Positive-frequency half
 half = 1:floor(N/2);
 f_half = f(half);
 Y_half = Y(half);

 // Single-sided peak-amplitude spectrum
 P = 2 * abs(Y_half) / N;

 // =====================================================
 // 2. Harmonic frequencies
 // =====================================================

 harmonic_orders = [1 3 5 7];
 harmonic_freqs = harmonic_orders * f1;

 // =====================================================
 // 3. Extract harmonic magnitudes
 // =====================================================

 V_peak = zeros(1, 4);
 V_rms = zeros(1, 4);

 for i = 1:size(harmonic_orders, "*")

     target_freq = harmonic_freqs(i);

     // Find nearest FFT frequency bin
     [min_value, k] = min(abs(f_half - target_freq));

     // Peak and RMS magnitudes
     V_peak(i) = P(k);
     V_rms(i) = V_peak(i) / sqrt(2);

 end

 // =====================================================
 // 4. THD calculation
 // =====================================================

 V1_rms = V_rms(1);
 harmonic_rms = V_rms(2:4);

 if V1_rms > 0 then
     THD = sqrt(sum(harmonic_rms.^2)) / V1_rms * 100;
 else
     THD = %nan;
 end

 // =====================================================
 // 5. Display harmonic results
 // =====================================================

 disp("==================================");
 disp("HARMONIC ANALYSIS");
 disp("==================================");

 disp("Harmonic orders:");
 disp(harmonic_orders);

 disp("Harmonic frequencies (Hz):");
 disp(harmonic_freqs);

 disp("Harmonic RMS magnitudes (V):");
 disp(V_rms);

 disp("THD (%):");
 disp(THD);
  // =====================================================
 //  First graphing -zayaz
 // =====================================================
 scf(1);
 clf(1);
 plot(f_half, P);
 xtitle("PCC Voltage Spectrum before Control", "Frequency (Hz)", "Peak Voltage (V)");
 a = gca(); //plt restriction 
 a.data_bounds = [0 0; 500 450];
 disp(max(P))

 // =====================================================
 // 6. Detect dominant harmonic
 // =====================================================

 // Consider only 3rd, 5th and 7th harmonics
 [dominant_rms, dominant_index] = max(V_rms(2:4));

 // Capacitor values in farads
 C_values = [20 20 32 40] * 1e-6;

 // Inductor value in henries
 L = 10e-3;

 // Switch outputs: 1 = ON, 0 = OFF
 switch_output = zeros(1, 4);

 if dominant_rms <= 1e-9 then

     disp("No significant 3rd, 5th or 7th harmonic detected.");
     disp("Filter switches remain OFF.");

 else

     // Map the maximum back to the original harmonic arrays
     dominant_index = dominant_index + 1;
     dominant_order = harmonic_orders(dominant_index);
     dominant_freq = harmonic_freqs(dominant_index);

     disp("==================================");
     disp("DOMINANT HARMONIC");
     disp("==================================");

     disp("Dominant harmonic order:");
     disp(dominant_order);

     disp("Dominant harmonic frequency (Hz):");
     disp(dominant_freq);

     disp("Dominant harmonic RMS magnitude (V):");
     disp(dominant_rms);

     // =================================================
     // 7. Test all 16 capacitor-switch combinations
     // =================================================

     best_error = %inf;
     best_capacitance = 0;
     best_mask = 0;

     for mask = 1:15

         // Build the four switch states from the mask
         trial_switches = zeros(1, 4);

         for j = 1:4
             trial_switches(j) = modulo(floor(mask / 2^(j-1)), 2);
         end

         // Capacitors are connected in parallel
         C_total = sum(trial_switches .* C_values);

         // Series LC resonant frequency
         f_res = 1 / (2 * %pi * sqrt(L * C_total));

         // Choose the combination nearest to the
         // dominant harmonic frequency
         frequency_error = abs(f_res - dominant_freq);

         if frequency_error < best_error then
             best_error = frequency_error;
             best_capacitance = C_total;
             best_mask = mask;
             switch_output = trial_switches;
         end

     end

     // =================================================
     // 8. Print selected switch combination
     // =================================================

     selected_f_res = 1 / ...
         (2 * %pi * sqrt(L * best_capacitance));

     disp("==================================");
     disp("RECOMMENDED FILTER SWITCH OUTPUT");
     disp("==================================");

     disp("Switch states [S1 S2 S3 S4]:");
     disp(switch_output);

     disp("S1: 20 uF capacitor");
     disp("S2: 20 uF capacitor");
     disp("S3: 32 uF capacitor");
     disp("S4: 40 uF capacitor");

     disp("Selected total capacitance (uF):");
     disp(best_capacitance * 1e6);

     disp("Filter resonant frequency (Hz):");
     disp(selected_f_res);

     disp("Frequency mismatch (Hz):");
     disp(best_error);

 end
