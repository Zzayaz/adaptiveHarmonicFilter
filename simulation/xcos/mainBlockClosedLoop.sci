

x = pcc_data.values;

Y = fft(x);
fs = 10000;
N = length(x);

f = (0:N-1)*(fs/N);

half = 1:floor(N/2);

f_half = f(half);
Y_half = Y(half);

P = 2*abs(Y_half)/N;

scf(1);
clf(1);
plot(pcc_data.time, x)
xtitle("PCC Voltage", "Time (s)", "Voltage (V)");


scf(2);
clf(2);
plot(f_half, P);
xtitle("PCC Voltage Spectrum", "Frequency (Hz)", "Peak Voltage (V)");
a = gca(); //plt restriction 
a.data_bounds = [0 0; 500 450];
disp(max(P))
