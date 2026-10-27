// Sampling parameters
fs = 10000;
N = 10000;

dt = 1/fs;
t = (0:N-1)*dt;

// Fundamental
V1 = 230*sqrt(2);

// Harmonic amplitudes
V3 = 46*sqrt(2);
V5 = 34.5*sqrt(2);
V7 = 23*sqrt(2);

// Harmonic frequencies
f1 = 50;
f3 = 150;
f5 = 250;
f7 = 350;

// Distorted voltage waveform
v = V1*sin(2*%pi*f1*t) ...
  + V3*sin(2*%pi*f3*t) ...
  + V5*sin(2*%pi*f5*t) ...
  + V7*sin(2*%pi*f7*t);
  
  clf();

plot(t(1:1000), v(1:1000));

xlabel("Time (s)");
ylabel("Voltage (V)");
title("Distorted 50 Hz Grid Voltage");
