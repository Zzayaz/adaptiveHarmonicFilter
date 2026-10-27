Y = fft(v);

f = (0:N-1)*(fs/N); // generates the frequency index. 0hz->9999hz

half = 1:floor(N/2); // only need half since positive fft conjugate

f_half = f(half);
Y_half = Y(half);

P = 2*abs(Y_half)/N; // abs of Y_half implies removal of the phase component then multiplied by 2 to offset the halfing
clf();

plot(f_half, P);
//replot([0, 0; 500, 1000]);



xlabel("Frequency (Hz)");
ylabel("Magnitude");
title("Harmonic Spectrum");
