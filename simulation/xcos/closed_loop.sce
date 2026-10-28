
 // ================================================
 // CLOSED-LOOP HARMONIC FILTER SIMULATION
 // ================================================

 clear;
 clc;

 // ================================================
 // 1. Load harmonic analysis function
 // ================================================

 exec("analyse_pcc_Data.sci", -1);

 // ================================================
 // 2. Helper to update an existing context variable
 //    while preserving all other context definitions
 // ================================================

 function context = set_context_value(context, name, value)

     context = matrix(context, 1, -1);
     found = %f;

     for i = 1:size(context, "*")

         line = stripblanks(context(i));

         if length(line) >= length(name) then

             if part(line, 1:length(name)) == name then

                 if length(line) > length(name) then

                     rest = stripblanks( ...
                         part(line, length(name)+1:length(line)));

                     if length(rest) > 0 then

                         if part(rest, 1:1) == "=" then
                             context(i) = name + "=" + string(value);
                             found = %t;
                             break;
                         end

                     end

                 end

             end

         end

     end

     // Add the variable only if it was not already defined
     if ~found then
         context = [context, name + "=" + string(value)];
     end

 endfunction

 // ================================================
 // 3. Import Xcos model
 // ================================================

 model_path = "mainBlockScript.zcos";

 importXcosDiagram(model_path);

 // ================================================
 // 4. First simulation: all filter switches OFF
 // ================================================

 scs_m.props.context = set_context_value( ...
     scs_m.props.context, "S1", 0);

 scs_m.props.context = set_context_value( ...
     scs_m.props.context, "S2", 0);

 scs_m.props.context = set_context_value( ...
     scs_m.props.context, "S3", 0);

 scs_m.props.context = set_context_value( ...
     scs_m.props.context, "S4", 0);

 [Info, status] = xcos_simulate(scs_m, 4);

 if ~status then
     error("Initial Xcos simulation failed.");
 end

 // Retrieve the initial PCC waveform
 x_before = pcc_data.values;

 // ================================================
 // 5. Analyze initial waveform
 // ================================================

 [switch_output, THD_before, ...
  f_before, P_before, V_rms_before] = ...
     analyse_pcc_Data(x_before);

 disp("================================");
 disp("SELECTED CAPACITOR SWITCHES");
 disp("================================");

 disp(switch_output);

 // ================================================
 // 6. Update the four Xcos context variables
 // ================================================

 scs_m.props.context = set_context_value( ...
     scs_m.props.context, "S1", switch_output(1));

 scs_m.props.context = set_context_value( ...
     scs_m.props.context, "S2", switch_output(2));

 scs_m.props.context = set_context_value( ...
     scs_m.props.context, "S3", switch_output(3));

 scs_m.props.context = set_context_value( ...
     scs_m.props.context, "S4", switch_output(4));

 disp("Updated Xcos context:");
 disp(scs_m.props.context);

 // ================================================
 // 7. Second simulation: selected filter combination
 // ================================================

 [Info, status] = xcos_simulate(scs_m, 4);

 if ~status then
     error("Filtered Xcos simulation failed.");
 end

 // Retrieve the filtered PCC waveform
 x_after = pcc_data.values;

 // ================================================
 // 8. Analyze filtered waveform
 // ================================================

 [switch_check, THD_after, ...
  f_after, P_after, V_rms_after] = ...
     analyse_pcc_Data(x_after);

 // ================================================
 // 9. Display comparison
 // ================================================

 disp("================================");
 disp("CLOSED-LOOP RESULTS");
 disp("================================");

 disp("Applied switch states [S1 S2 S3 S4]:");
 disp(switch_output);

 disp("THD before filtering (%):");
 disp(THD_before);

 disp("THD after filtering (%):");
 disp(THD_after);

 disp("THD reduction (percentage points):");
 disp(THD_before - THD_after);

 if THD_before <> 0 & ~isnan(THD_before) then
     disp("Relative THD reduction (%):");
     disp(100 * (THD_before - THD_after) / THD_before);
 end

 disp("RMS harmonics before filtering [1 3 5 7]:");
 disp(V_rms_before);

 disp("RMS harmonics after filtering [1 3 5 7]:");
 disp(V_rms_after);

 // ================================================
 // 10. Plot PCC waveforms
 // ================================================


 // ================================================
 // 10. Plot first 5 cycles of PCC waveforms
 // ================================================

 fs = 10000;
 f1 = 50;

 samples_per_cycle = fs / f1;
 cycles_to_plot = 5;
 N_plot = min(cycles_to_plot * samples_per_cycle, ...
              min(size(x_before, "*"), size(x_after, "*")));

 time = (0:N_plot-1) / fs;

 scf(10);
 clf(10);

 subplot(2,1,1);
 plot(time, x_before(1:N_plot));
 xtitle("PCC waveform before filtering - 5 cycles", ...
        "Time (s)", "Voltage (V)");

 subplot(2,1,2);
 plot(time, x_after(1:N_plot));
 xtitle("PCC waveform after filtering - 5 cycles", ...
        "Time (s)", "Voltage (V)");


 // ================================================
 // 11. Plot harmonic spectra
 // ================================================

 scf(11);
 clf(11);

 // Both simulations should use the same sample count
 if size(P_before, "*") == size(P_after, "*") then

     plot(f_before', [P_before' P_after']);

     xtitle("PCC spectrum before and after filtering", ...
            "Frequency (Hz)", "Peak voltage (V)");

     legend(["Before filtering"; "After filtering"]);

     a = gca();
     a.data_bounds = [0 0; ...
         500 max([P_before P_after])];

 else

     disp("Spectrum lengths differ; plotting separately.");

     subplot(2, 1, 1);
     plot(f_before, P_before);
     xtitle("Spectrum before filtering", ...
            "Frequency (Hz)", "Peak voltage (V)");

     subplot(2, 1, 2);
     plot(f_after, P_after);
     xtitle("Spectrum after filtering", ...
            "Frequency (Hz)", "Peak voltage (V)");

 end
