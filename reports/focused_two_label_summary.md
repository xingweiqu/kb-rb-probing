# Focused Two-Label Summary (strict rows)

本文件仅做聚焦整理：筛选严格满足计数阈值与非 MCQ/非 random split 的既有结果；不重跑模型、不生成新 decision rule、不输出研究结论。

## wrong_claim_susceptible

- valid delta transfer rows (n=8):
  - dataset=gpt_test; split=realization::train_natural__test_symbolic; train=gpt_test; test=gpt_test; position=pre_answer; feature_mode=delta::wrongclaim_bare-original; layer=2; hidden_f1=0.7456189937817976; train_pos/neg=12/78; test_pos/neg=25/65 (outputs/first_pass_probing/gpt_test/pre_answer/delta::wrongclaim_bare-original/wrong_claim_susceptible/realization::train_natural__test_symbolic/summary.json)
  - dataset=gpt_test; split=realization::train_natural__test_symbolic; train=gpt_test; test=gpt_test; position=final_input; feature_mode=delta::wrongclaim_bare-original; layer=28; hidden_f1=0.7025240384615384; train_pos/neg=12/78; test_pos/neg=25/65 (outputs/first_pass_probing/gpt_test/final_input/delta::wrongclaim_bare-original/wrong_claim_susceptible/realization::train_natural__test_symbolic/summary.json)
  - dataset=probe_ready; split=realization::train_natural__test_symbolic; train=probe_ready; test=probe_ready; position=final_input; feature_mode=delta::wrongclaim_bare-original; layer=32; hidden_f1=0.6344025773917916; train_pos/neg=32/58; test_pos/neg=16/74 (outputs/first_pass_probing/probe_ready/final_input/delta::wrongclaim_bare-original/wrong_claim_susceptible/realization::train_natural__test_symbolic/summary.json)
  - dataset=gpt_test; split=realization::train_symbolic__test_natural; train=gpt_test; test=gpt_test; position=final_input; feature_mode=delta::wrongclaim_bare-original; layer=22; hidden_f1=0.6199324324324325; train_pos/neg=25/65; test_pos/neg=12/78 (outputs/first_pass_probing/gpt_test/final_input/delta::wrongclaim_bare-original/wrong_claim_susceptible/realization::train_symbolic__test_natural/summary.json)
  - dataset=probe_ready; split=realization::train_natural__test_symbolic; train=probe_ready; test=probe_ready; position=pre_answer; feature_mode=delta::wrongclaim_bare-original; layer=16; hidden_f1=0.608957219251337; train_pos/neg=32/58; test_pos/neg=16/74 (outputs/first_pass_probing/probe_ready/pre_answer/delta::wrongclaim_bare-original/wrong_claim_susceptible/realization::train_natural__test_symbolic/summary.json)
  - dataset=probe_ready; split=realization::train_symbolic__test_natural; train=probe_ready; test=probe_ready; position=pre_answer; feature_mode=delta::wrongclaim_bare-original; layer=6; hidden_f1=0.5824668053111501; train_pos/neg=16/74; test_pos/neg=32/58 (outputs/first_pass_probing/probe_ready/pre_answer/delta::wrongclaim_bare-original/wrong_claim_susceptible/realization::train_symbolic__test_natural/summary.json)
  - dataset=probe_ready; split=realization::train_symbolic__test_natural; train=probe_ready; test=probe_ready; position=final_input; feature_mode=delta::wrongclaim_bare-original; layer=5; hidden_f1=0.5482295482295483; train_pos/neg=16/74; test_pos/neg=32/58 (outputs/first_pass_probing/probe_ready/final_input/delta::wrongclaim_bare-original/wrong_claim_susceptible/realization::train_symbolic__test_natural/summary.json)
  - dataset=gpt_test; split=realization::train_symbolic__test_natural; train=gpt_test; test=gpt_test; position=pre_answer; feature_mode=delta::wrongclaim_bare-original; layer=24; hidden_f1=0.54398894518655; train_pos/neg=25/65; test_pos/neg=12/78 (outputs/first_pass_probing/gpt_test/pre_answer/delta::wrongclaim_bare-original/wrong_claim_susceptible/realization::train_symbolic__test_natural/summary.json)

- 是否有 natural->symbolic: True
- 是否有 symbolic->natural: True

## removal_dependent

- valid delta transfer rows (n=4):
  - dataset=gpt_test; split=realization::train_natural__test_symbolic; train=gpt_test; test=gpt_test; position=pre_answer; feature_mode=delta::premise_removal-original; layer=17; hidden_f1=0.825; train_pos/neg=37/53; test_pos/neg=59/31 (outputs/first_pass_probing/gpt_test/pre_answer/delta::premise_removal-original/removal_dependent/realization::train_natural__test_symbolic/summary.json)
  - dataset=gpt_test; split=realization::train_natural__test_symbolic; train=gpt_test; test=gpt_test; position=final_input; feature_mode=delta::premise_removal-original; layer=13; hidden_f1=0.7948016415868673; train_pos/neg=37/53; test_pos/neg=59/31 (outputs/first_pass_probing/gpt_test/final_input/delta::premise_removal-original/removal_dependent/realization::train_natural__test_symbolic/summary.json)
  - dataset=gpt_test; split=realization::train_symbolic__test_natural; train=gpt_test; test=gpt_test; position=final_input; feature_mode=delta::premise_removal-original; layer=12; hidden_f1=0.6592244418331374; train_pos/neg=59/31; test_pos/neg=37/53 (outputs/first_pass_probing/gpt_test/final_input/delta::premise_removal-original/removal_dependent/realization::train_symbolic__test_natural/summary.json)
  - dataset=gpt_test; split=realization::train_symbolic__test_natural; train=gpt_test; test=gpt_test; position=pre_answer; feature_mode=delta::premise_removal-original; layer=26; hidden_f1=0.599802371541502; train_pos/neg=59/31; test_pos/neg=37/53 (outputs/first_pass_probing/gpt_test/pre_answer/delta::premise_removal-original/removal_dependent/realization::train_symbolic__test_natural/summary.json)

- 是否有 natural->symbolic: True
- 是否有 symbolic->natural: True

## cross_dataset_transfer_v2 delta

- no delta cross-dataset result

## Signal value distributions (from atomic_labels_*.json)

- probe_ready removal_dependent pos/neg (from labels): 179/1

### probe_ready / removal_dependent / signal=removal_drop
- n_with_signal: 180
- quantiles: q00=-1.0000, q01=0.0000, q05=0.0000, q10=0.0000, q20=0.0000, q50=0.0000, q80=0.0000, q90=1.0000, q95=1.0000, q99=1.0000, q100=1.0000
- hist: [-1.00,-0.75):1; [-0.75,-0.50):0; [-0.50,-0.25):0; [-0.25,-0.10):0; [-0.10,0.00):0; [0.00,0.10):159; [0.10,0.25):0; [0.25,0.50):0; [0.50,0.75):0; [0.75,1.00):20
- top10 (fid, value): probe_nat_hybrid_006:1.0000, probe_nat_kb_014:1.0000, probe_nat_rb_004:1.0000, probe_nat_rb_009:1.0000, probe_sym_kb_009:1.0000, probe_sym_kb_010:1.0000, probe_sym_kb_011:1.0000, probe_sym_kb_012:1.0000, probe_sym_kb_013:1.0000, probe_sym_kb_014:1.0000
- bottom10 (fid, value): probe_nat_rb_017:-1.0000, probe_nat_hybrid_001:0.0000, probe_nat_hybrid_002:0.0000, probe_nat_hybrid_003:0.0000, probe_nat_hybrid_004:0.0000, probe_nat_hybrid_005:0.0000, probe_nat_hybrid_007:0.0000, probe_nat_hybrid_008:0.0000, probe_nat_hybrid_009:0.0000, probe_nat_hybrid_010:0.0000

### probe_ready / wrong_claim_susceptible / signal=wrongclaim_drop
- n_with_signal: 180
- quantiles: q00=0.0000, q01=0.0000, q05=0.0000, q10=0.0000, q20=0.0000, q50=0.0000, q80=1.0000, q90=1.0000, q95=1.0000, q99=1.0000, q100=1.0000
- hist: [-1.00,-0.75):0; [-0.75,-0.50):0; [-0.50,-0.25):0; [-0.25,-0.10):0; [-0.10,0.00):0; [0.00,0.10):132; [0.10,0.25):0; [0.25,0.50):0; [0.50,0.75):0; [0.75,1.00):48
- top10 (fid, value): probe_nat_hybrid_002:1.0000, probe_nat_hybrid_006:1.0000, probe_nat_hybrid_007:1.0000, probe_nat_hybrid_010:1.0000, probe_nat_hybrid_011:1.0000, probe_nat_hybrid_016:1.0000, probe_nat_hybrid_018:1.0000, probe_nat_hybrid_020:1.0000, probe_nat_hybrid_022:1.0000, probe_nat_hybrid_023:1.0000
- bottom10 (fid, value): probe_nat_hybrid_001:0.0000, probe_nat_hybrid_003:0.0000, probe_nat_hybrid_004:0.0000, probe_nat_hybrid_005:0.0000, probe_nat_hybrid_008:0.0000, probe_nat_hybrid_009:0.0000, probe_nat_hybrid_012:0.0000, probe_nat_hybrid_013:0.0000, probe_nat_hybrid_014:0.0000, probe_nat_hybrid_015:0.0000

### gpt_test / removal_dependent / signal=removal_drop
- n_with_signal: 180
- quantiles: q00=-1.0000, q01=-1.0000, q05=0.0000, q10=0.0000, q20=0.0000, q50=1.0000, q80=1.0000, q90=1.0000, q95=1.0000, q99=1.0000, q100=1.0000
- hist: [-1.00,-0.75):5; [-0.75,-0.50):0; [-0.50,-0.25):0; [-0.25,-0.10):0; [-0.10,0.00):0; [0.00,0.10):79; [0.10,0.25):0; [0.25,0.50):0; [0.50,0.75):0; [0.75,1.00):96
- top10 (fid, value): HYB-N-002:1.0000, HYB-N-006:1.0000, HYB-N-007:1.0000, HYB-N-010:1.0000, HYB-N-013:1.0000, HYB-N-018:1.0000, HYB-N-020:1.0000, HYB-N-021:1.0000, HYB-N-023:1.0000, HYB-N-024:1.0000
- bottom10 (fid, value): HYB-N-011:-1.0000, RB-N-006:-1.0000, RB-N-017:-1.0000, RB-N-020:-1.0000, RB-S-024:-1.0000, HYB-N-001:0.0000, HYB-N-003:0.0000, HYB-N-004:0.0000, HYB-N-005:0.0000, HYB-N-008:0.0000

### gpt_test / wrong_claim_susceptible / signal=wrongclaim_drop
- n_with_signal: 180
- quantiles: q00=-1.0000, q01=-1.0000, q05=0.0000, q10=0.0000, q20=0.0000, q50=0.0000, q80=1.0000, q90=1.0000, q95=1.0000, q99=1.0000, q100=1.0000
- hist: [-1.00,-0.75):5; [-0.75,-0.50):0; [-0.50,-0.25):0; [-0.25,-0.10):0; [-0.10,0.00):0; [0.00,0.10):138; [0.10,0.25):0; [0.25,0.50):0; [0.50,0.75):0; [0.75,1.00):37
- top10 (fid, value): HYB-N-001:1.0000, HYB-N-013:1.0000, HYB-N-016:1.0000, HYB-S-003:1.0000, HYB-S-008:1.0000, HYB-S-014:1.0000, HYB-S-015:1.0000, HYB-S-028:1.0000, HYB-S-029:1.0000, KB-S-001:1.0000
- bottom10 (fid, value): HYB-N-005:-1.0000, HYB-N-011:-1.0000, HYB-N-015:-1.0000, RB-S-002:-1.0000, RB-S-024:-1.0000, HYB-N-002:0.0000, HYB-N-003:0.0000, HYB-N-004:0.0000, HYB-N-006:0.0000, HYB-N-007:0.0000

## removal_dependent pos/neg 计数异常的阈值/同值说明

以下为 **extreme-bin** 阈值切分在当前信号分布上的机械结果（用于解释为何会出现 pos=179/neg=1 这种极端计数；不涉及任何结论判断）。

- probe_ready / removal_dependent: signal=removal_drop, pos_thr=0.000000, neg_thr=0.000000, n=180
  - n(signal>=pos_thr)=179, n(signal<=neg_thr)=160
  - ties: n_eq_pos_thr=159, n_eq_neg_thr=159, n_unique_values=3
  - unique_values_preview(first<=20): [-1.0, 0.0, 1.0]
- gpt_test / removal_dependent: signal=removal_drop, pos_thr=1.000000, neg_thr=0.000000, n=180
  - n(signal>=pos_thr)=96, n(signal<=neg_thr)=84
  - ties: n_eq_pos_thr=96, n_eq_neg_thr=79, n_unique_values=3
  - unique_values_preview(first<=20): [-1.0, 0.0, 1.0]
- probe_ready / wrong_claim_susceptible: signal=wrongclaim_drop, pos_thr=1.000000, neg_thr=0.000000, n=180
  - n(signal>=pos_thr)=48, n(signal<=neg_thr)=132
  - ties: n_eq_pos_thr=48, n_eq_neg_thr=132, n_unique_values=2
  - unique_values_preview(first<=20): [0.0, 1.0]
- gpt_test / wrong_claim_susceptible: signal=wrongclaim_drop, pos_thr=1.000000, neg_thr=0.000000, n=180
  - n(signal>=pos_thr)=37, n(signal<=neg_thr)=143
  - ties: n_eq_pos_thr=37, n_eq_neg_thr=138, n_unique_values=3
  - unique_values_preview(first<=20): [-1.0, 0.0, 1.0]
