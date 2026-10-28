
function [switch_output, THD, f_half, P, V_rms] = analyse_pcc_Data(x)

    // ================================================
    // 1. Prepare sampled PCC waveform
    // ================================================

    x = matrix(x, 1, -1);

    fs = 10000;
    N = size(x, "*");
    f1 = 50;

    if N < 2 then
        error("PCC waveform contains insufficient samples.");
    end

    // ================================================
    // 2. FFT and single-sided spectrum
    // ================================================

    Y = fft(x);
    f = (0:N-1) * (fs/N);

    half = 1:floor(N/2);
    f_half = f(half);
    Y_half = Y(half);

    P = 2 * abs(Y_half) / N;

    // ================================================
    // 3. Extract 1st, 3rd, 5th and 7th harmonics
    // ================================================

    harmonic_orders = [1 3 5 7];
    harmonic_freqs = harmonic_orders * f1;

    V_peak = zeros(1, 4);
    V_rms = zeros(1, 4);

    for i = 1:4
        target_freq = harmonic_freqs(i);

        [min_value, k] = min(abs(f_half - target_freq));

        V_peak(i) = P(k);
        V_rms(i) = V_peak(i) / sqrt(2);
    end

    // ================================================
    // 4. Calculate THD
    // ================================================

    V1_rms = V_rms(1);

    if V1_rms > 0 then
        THD = sqrt(sum(V_rms(2:4).^2)) / V1_rms * 100;
    else
        THD = %nan;
    end

    // ================================================
    // 5. Display harmonic analysis
    // ================================================

    disp("================================");
    disp("HARMONIC ANALYSIS");
    disp("================================");

    disp("Harmonic orders:");
    disp(harmonic_orders);

    disp("Harmonic frequencies (Hz):");
    disp(harmonic_freqs);

    disp("Harmonic RMS magnitudes (V):");
    disp(V_rms);

    disp("THD (%):");
    disp(THD);

    // ================================================
    // 6. Find dominant non-fundamental harmonic
    // ================================================

    [dominant_rms, dominant_index] = max(V_rms(2:4));

    C_values = [20 20 32 40] * 1e-6;
    L = 10e-3;

    // All switches initially OFF
    switch_output = zeros(1, 4);

    if dominant_rms <= 1e-9 then

        disp("No significant 3rd, 5th or 7th harmonic.");
        disp("All filter switches remain OFF.");

    else

        dominant_index = dominant_index + 1;
        dominant_order = harmonic_orders(dominant_index);
        dominant_freq = harmonic_freqs(dominant_index);

        disp("Dominant harmonic order:");
        disp(dominant_order);

        disp("Dominant harmonic frequency (Hz):");
        disp(dominant_freq);

        disp("Dominant harmonic RMS (V):");
        disp(dominant_rms);

        // ============================================
        // 7. Evaluate all 15 nonempty combinations
        // ============================================

        best_error = %inf;
        best_capacitance = 0;

        for mask = 1:15

            trial_switches = zeros(1, 4);

            for j = 1:4
                trial_switches(j) = ...
                    modulo(floor(mask / 2^(j-1)), 2);
            end

            // Capacitors connected in parallel
            C_total = sum(trial_switches .* C_values);

            // Resonant frequency of series LC branch
            f_res = 1 / (2 * %pi * sqrt(L * C_total));

            frequency_error = abs(f_res - dominant_freq);

            if frequency_error < best_error then
                best_error = frequency_error;
                best_capacitance = C_total;
                switch_output = trial_switches;
            end

        end

        // ============================================
        // 8. Display selected switch combination
        // ============================================

        selected_f_res = ...
            1 / (2 * %pi * sqrt(L * best_capacitance));

        disp("================================");
        disp("RECOMMENDED SWITCH OUTPUT");
        disp("================================");

        disp("Switches [S1 S2 S3 S4]:");
        disp(switch_output);

        disp("Selected capacitance (uF):");
        disp(best_capacitance * 1e6);

        disp("Calculated resonant frequency (Hz):");
        disp(selected_f_res);

        disp("Frequency mismatch (Hz):");
        disp(best_error);

    end

endfunction
