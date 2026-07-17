# Hyperparameter & Ablation Results (Current)

This report summarizes all hyperparameter/ablation runs available so far.

## Implementation Summary

- **Heads sweep:** uses existing ranked heads (k≈25), evaluates DPSWeightedSoft on 50 samples.
- **Group-size sweep:** uses existing k=250 (≈10 users) and k=25 (≈100 users), evaluates on 50 samples.
- **Gamma sweep:** adaptive α vs fixed α=0.5, full dev where available.
- **Heads ablation:** randomize head weights and random mask; includes reference DPSWeightedSoft where found.
- **Routing ablation:** soft vs hard routing with 200 samples (per-task).

## Key Takeaways (Auto)

- Heads sweep: 160 (best by accuracy).
- Group size sweep: 100 (best by accuracy).
- Gamma sweep: adaptive wins on 2 task(s) (LaMP-3, LaMP-5), fixed α=0.5 wins on 2 task(s) (LaMP-1, LaMP-2), ties on 0 task(s) (n/a).
- Heads ablation: n/a (no true-head references found).
- Routing ablation: soft wins on 2 task(s) (LaMP-4, LaMP-7), hard wins on 3 task(s) (LaMP-1, LaMP-2, LaMP-5), ties on 1 task(s) (LaMP-3).

## Heads Sweep (50-sample quick)

| heads | task   | samples | metrics               | pred_file                                                                                   |
| ----- | ------ | ------- | --------------------- | ------------------------------------------------------------------------------------------- |
| 10    | LaMP-1 | 2550    | acc=0.6278, f1=0.6130 | decore/outputs/hparam/heads_quick/h10/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json  |
| 20    | LaMP-1 | 2550    | acc=0.6271, f1=0.6124 | decore/outputs/hparam/heads_quick/h20/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json  |
| 40    | LaMP-1 | 2550    | acc=0.6290, f1=0.6147 | decore/outputs/hparam/heads_quick/h40/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json  |
| 80    | LaMP-1 | 2550    | acc=0.6282, f1=0.6141 | decore/outputs/hparam/heads_quick/h80/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json  |
| 160   | LaMP-1 | 2550    | acc=0.6302, f1=0.6167 | decore/outputs/hparam/heads_quick/h160/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json |

Figure: `decore/outputs/hparam/figures/heads/heads_sweep.png`

## Group Size Sweep (50-sample quick)

| group_size | task   | samples | metrics               | pred_file                                                                                       |
| ---------- | ------ | ------- | --------------------- | ----------------------------------------------------------------------------------------------- |
| 10         | LaMP-1 | 2550    | acc=0.6271, f1=0.6124 | decore/outputs/hparam/groupsize_quick/g10/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json  |
| 100        | LaMP-1 | 2550    | acc=0.6353, f1=0.6285 | decore/outputs/hparam/groupsize_quick/g100/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json |

Figure: `decore/outputs/hparam/figures/groupsize/groupsize_sweep.png`

## Gamma Sweep (full dev where available)

| task   | setting         | samples | metrics                             | pred_file                                                                                                   |
| ------ | --------------- | ------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| LaMP-1 | adaptive        | 2500    | acc=0.6308, f1=0.6192               | decore/outputs/hparam/gamma_full/adaptive/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json              |
| LaMP-1 | fixed_alpha_0.5 | 2500    | acc=0.6316, f1=0.6216               | decore/outputs/hparam/gamma_full/fixed_alpha_0.5/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json       |
| LaMP-2 | adaptive        | 1445    | acc=0.4519, f1=0.3807               | decore/outputs/hparam/gamma_full/lamp2/adaptive/pred_LAMP_2_LLaMA3-8b-Instruct__DPSWeightedSoft.json        |
| LaMP-2 | fixed_alpha_0.5 | 1384    | acc=0.4595, f1=0.3897               | decore/outputs/hparam/gamma_full/lamp2/fixed_alpha_0.5/pred_LAMP_2_LLaMA3-8b-Instruct__DPSWeightedSoft.json |
| LaMP-3 | adaptive        | 2500    | mae=0.4382, rmse=0.9237             | decore/outputs/hparam/gamma_full/lamp3/adaptive/pred_LAMP_3_LLaMA3-8b-Instruct__DPSWeightedSoft.json        |
| LaMP-3 | fixed_alpha_0.5 | 2221    | mae=0.4451, rmse=0.9347             | decore/outputs/hparam/gamma_full/lamp3/fixed_alpha_0.5/pred_LAMP_3_LLaMA3-8b-Instruct__DPSWeightedSoft.json |
| LaMP-5 | adaptive        | 2500    | r1=0.3460, rL=0.2991, meteor=0.3887 | decore/outputs/hparam/gamma_full/lamp5/adaptive/pred_LAMP_5_LLaMA3-8b-Instruct__DPSWeightedSoft.json        |
| LaMP-5 | fixed_alpha_0.5 | 1046    | r1=0.3444, rL=0.2949, meteor=0.3943 | decore/outputs/hparam/gamma_full/lamp5/fixed_alpha_0.5/pred_LAMP_5_LLaMA3-8b-Instruct__DPSWeightedSoft.json |

Figure: `decore/outputs/hparam/figures/gamma/gamma_sweep.png` (LaMP-1)

## Heads Ablation (full dev where available)

| task   | setting      | samples | metrics                             | pred_file                                                                                                   |
| ------ | ------------ | ------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| LaMP-1 | random_heads | 2529    | acc=0.6220, f1=0.6067               | decore/outputs/hparam/ablation_full/random_heads/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json       |
| LaMP-1 | random_mask  | 2500    | acc=0.6292, f1=0.6150               | decore/outputs/hparam/ablation_full/random_mask/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json        |
| LaMP-2 | random_heads | 1467    | acc=0.4560, f1=0.3851               | decore/outputs/hparam/ablation_full/lamp2/random_heads/pred_LAMP_2_LLaMA3-8b-Instruct__DPSWeightedSoft.json |
| LaMP-2 | random_mask  | 1384    | acc=0.4523, f1=0.3839               | decore/outputs/hparam/ablation_full/lamp2/random_mask/pred_LAMP_2_LLaMA3-8b-Instruct__DPSWeightedSoft.json  |
| LaMP-3 | random_heads | 2500    | mae=0.4434, rmse=0.9351             | decore/outputs/hparam/ablation_full/lamp3/random_heads/pred_LAMP_3_LLaMA3-8b-Instruct__DPSWeightedSoft.json |
| LaMP-3 | random_mask  | 2498    | mae=0.4386, rmse=0.9206             | decore/outputs/hparam/ablation_full/lamp3/random_mask/pred_LAMP_3_LLaMA3-8b-Instruct__DPSWeightedSoft.json  |
| LaMP-5 | random_heads | 2500    | r1=0.3322, rL=0.2872, meteor=0.3848 | decore/outputs/hparam/ablation_full/lamp5/random_heads/pred_LAMP_5_LLaMA3-8b-Instruct__DPSWeightedSoft.json |
| LaMP-5 | random_mask  | 933     | r1=0.3725, rL=0.3216, meteor=0.4059 | decore/outputs/hparam/ablation_full/lamp5/random_mask/pred_LAMP_5_LLaMA3-8b-Instruct__DPSWeightedSoft.json  |

Figures: `decore/outputs/hparam/figures/ablation/ablation_metrics.png`, `decore/outputs/hparam/figures/ablation/ablation_head_heatmaps.png`

## Routing Ablation (soft vs hard, 200 samples)

| task   | routing | samples | metrics                             | pred_file                                                                                              |
| ------ | ------- | ------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------ |
| LaMP-1 | hard    | 400     | acc=0.6300, f1=0.6276               | decore/outputs/hparam/routing_ablation/lamp1/hard/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedHard.json |
| LaMP-1 | soft    | 400     | acc=0.6150, f1=0.6115               | decore/outputs/hparam/routing_ablation/lamp1/soft/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json |
| LaMP-2 | hard    | 200     | acc=0.4200, f1=0.3151               | decore/outputs/hparam/routing_ablation/lamp2/hard/pred_LAMP_2_LLaMA3-8b-Instruct__DPSWeightedHard.json |
| LaMP-2 | soft    | 200     | acc=0.4100, f1=0.3025               | decore/outputs/hparam/routing_ablation/lamp2/soft/pred_LAMP_2_LLaMA3-8b-Instruct__DPSWeightedSoft.json |
| LaMP-3 | hard    | 200     | mae=0.4050, rmse=0.8860             | decore/outputs/hparam/routing_ablation/lamp3/hard/pred_LAMP_3_LLaMA3-8b-Instruct__DPSWeightedHard.json |
| LaMP-3 | soft    | 200     | mae=0.4050, rmse=0.8860             | decore/outputs/hparam/routing_ablation/lamp3/soft/pred_LAMP_3_LLaMA3-8b-Instruct__DPSWeightedSoft.json |
| LaMP-4 | hard    | 148     | r1=0.1767, rL=0.1584, meteor=0.1591 | decore/outputs/hparam/routing_ablation/lamp4/hard/pred_LAMP_4_LLaMA3-8b-Instruct__DPSWeightedHard.json |
| LaMP-4 | soft    | 200     | r1=0.1927, rL=0.1753, meteor=0.1825 | decore/outputs/hparam/routing_ablation/lamp4/soft/pred_LAMP_4_LLaMA3-8b-Instruct__DPSWeightedSoft.json |
| LaMP-5 | hard    | 200     | r1=0.3716, rL=0.3123, meteor=0.4134 | decore/outputs/hparam/routing_ablation/lamp5/hard/pred_LAMP_5_LLaMA3-8b-Instruct__DPSWeightedHard.json |
| LaMP-5 | soft    | 200     | r1=0.3468, rL=0.2956, meteor=0.4231 | decore/outputs/hparam/routing_ablation/lamp5/soft/pred_LAMP_5_LLaMA3-8b-Instruct__DPSWeightedSoft.json |
| LaMP-7 | hard    | 200     | r1=0.3187, rL=0.2707, meteor=0.2620 | decore/outputs/hparam/routing_ablation/lamp7/hard/pred_LAMP_7_LLaMA3-8b-Instruct__DPSWeightedHard.json |
| LaMP-7 | soft    | 200     | r1=0.3277, rL=0.2799, meteor=0.2714 | decore/outputs/hparam/routing_ablation/lamp7/soft/pred_LAMP_7_LLaMA3-8b-Instruct__DPSWeightedSoft.json |

Figure: `decore/outputs/hparam/routing_ablation/routing_ablation.png`

## Coverage & Missing Points

- Heads sweep missing: none
- Group size sweep missing: 50, 200, 400
- Gamma sweep missing tasks: LaMP-4, LaMP-7
- Heads ablation missing tasks: LaMP-4, LaMP-7
- Routing ablation missing soft/hard pairs: none
