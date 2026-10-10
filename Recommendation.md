# Utility Recommendation: Washer Energy Feedback

**Decision:** use this project as a feasibility study and a guide for a small submetered pilot. The current disaggregation models are not ready for customer-level appliance estimates.

## Evidence and suitable uses

Directly metered analysis across 19 REFIT homes finds 6,581 heuristic washer events, with a median duration of 68 minutes and median energy of 0.533 kWh. Mean observed washer demand is highest around 08:00 and on Saturday. These patterns can help design an opt-in research pilot and investigate wash-program energy, after checking inconsistent labels.

The models perform poorly on later periods in three known homes. On 55,823 common test minutes, Seq2Point MAE is 1.370 W and Seq2Seq MAE is 1.395 W, against 1.150 W for always predicting zero. Activity F1 is 0.046 and 0.000; estimated washer energy is about 79% below the measured total. The small overall errors reflect mostly off-state minutes, not reliable appliance identification.

Quality filtering is a major limitation: only 114 of 3,714 paired active test minutes survive the 61-minute window rules. The evidence supports pipeline development and data-quality investigation, but not appliance billing, individualized savings estimates, automated control or fault alerts. Details and figures are in [RESULTS.md](RESULTS.md).

## One energy-saving opportunity

**Pilot lower-temperature washing for suitable loads.** Detected events commonly peak near 2 kW, and the illustrated event contains a high-power stage consistent with water heating. That interpretation needs program metadata. [ENERGY STAR's laundry guidance](https://www.energystar.gov/products/clothes_washers) supports cold-water washing where appropriate; its published savings figures are not assumed to apply to these homes.

Invite volunteers with temperature-selectable machines to try an appropriate lower-temperature program, following garment and machine instructions. Record program, temperature, load mass, repeat washes and cleaning satisfaction. Measure washer electricity directly during comparable baseline and intervention weeks, preferably using a randomized crossover. Compare kWh per kg of laundry and weekly washer kWh, report uncertainty, and exclude or separately report incomplete cycles. Measure any separate electric water heating if it contributes to the intervention.

Savings must come from the pilot measurements, not this model. Moving a wash to another hour may change peak demand or cost, but does not by itself demonstrate reduced electricity consumption.

## Limits and data needed for confidence

Overlapping appliance signatures, unmetered loads, missing observations and uncertain labels limit inference. One-minute averaging hides brief transitions; 15/30-minute readings hide more of the cycle. The current centered windows also require future readings, making inference retrospective or delayed. No coarser-resolution model or unseen-household test has been evaluated.

For an Indian utility, collect synchronized aggregate and washer submeters from a representative local sample across seasons, with meter-quality flags, appliance models, heater availability, programs, temperatures and household routines. Verify timestamp conventions and include appliance replacements and behavior changes. UK usage patterns and heating signatures should not be assumed to transfer.

Proceed beyond the pilot only after independent homes and actual utility meter resolutions show useful active-event recovery and energy estimates against audited submeter readings, with uncertainty bounds and improvement over simple baselines.


## Future Experiments

- **LSTM / Bidirectional LSTM:** Explore recurrent neural networks to capture long-term temporal dependencies and appliance usage patterns.
- **CNN-LSTM Hybrid:** Combine CNN-based feature extraction with LSTM layers to learn both local power signatures and sequential dependencies.
- **Transformer-Based Models:** Experiment with attention mechanisms to identify important temporal relationships in aggregate power consumption.
- **Temporal Convolutional Networks (TCN):** Investigate dilated causal convolutions for capturing longer-duration appliance activity patterns.
- **Hyperparameter Optimization:** Experiment with different window sizes, learning rates, loss functions, and sampling strategies to improve disaggregation accuracy.
- **Cross-Household Generalization:** Evaluate model performance on unseen households to assess robustness and real-world applicability.