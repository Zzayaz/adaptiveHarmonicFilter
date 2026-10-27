// =====================================================
// SECTION 4
// Harmonic extraction and THD calculation
// =====================================================

// Sampled PCC waveform
x = pcc_data.values;

// Sampling parameters
fs = 10000;
N = size(x,"*");

// Fundamental frequency
f1 = 50;

// -----------------------------------------------------
// FFT
// -----------------------------------------------------

Y = fft(x);

// Frequency axis
f = (0:N-1)*(fs/N);

// Positive-frequency half
half = 1:floor(N/2);

f_half = f(half);
Y_half = Y(half);

// Single-sided peak-amplitude spectrum
P = 2*abs(Y_half)/N;


// -----------------------------------------------------
// Harmonic frequencies
// -----------------------------------------------------

harmonic_orders = [1 3 5 7];

harmonic_freqs = harmonic_orders * f1;


// -----------------------------------------------------
// Extract harmonic magnitudes
// -----------------------------------------------------

V_peak = zeros(1,4);
V_rms = zeros(1,4);

for i = 1:size(harmonic_orders,"*")
    
    target_freq = harmonic_freqs(i);
    
    // Find nearest FFT frequency bin
   [min_value, k] = min(abs(f_half - target_freq));
    
    // Peak amplitude
    V_peak(i) = P(k);
    
    // RMS amplitude
    V_rms(i) = V_peak(i)/sqrt(2);
    
end


// -----------------------------------------------------
// THD
// -----------------------------------------------------

V1_rms = V_rms(1);

harmonic_rms = V_rms(2:$);

THD = sqrt(sum(harmonic_rms.^2)) / V1_rms * 100;


// -----------------------------------------------------
// Display results
// -----------------------------------------------------

disp("Harmonic orders:");
disp(harmonic_orders);

disp("Harmonic frequencies (Hz):");
disp(harmonic_freqs);

disp("Harmonic RMS magnitudes (V):");
disp(V_rms);

disp("THD (%):");
disp(THD);
