pkg load interval

input = csvread("Chanel_2_600nm_0.2.csv")
eps = 1e-4

input_int = infsup(input - eps, input + eps)
figure
errorbar(mid(input_int), rad(input_int), "b")
title("Data")
xlabel("n")
ylabel("mV")
xlim([0, size(input, 1)])

print  -dpng Result10/data_int.png
