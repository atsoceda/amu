# Cross-Country Commitment Screen Report

Generated: 2026-06-29T18:01:52+0900
Model: `google/gemma-3-270m`

## Verdict

The screen found strong same-country movement, but cross-country wrong-answer induction remains a serious confound.

## Baseline Snapshots

- `france_association` (ambiguous_association): top5 ` the`, ` Paris`, ` France`, ` that`, ` Marseille`
- `france_association_plain` (ambiguous_association): top5 ` Paris`, ` the`, ` Lyon`, ` France`, ` Bordeaux`
- `france_famous_city` (near_match): top5 ` the`, ` a`, ` known`, ` one`, ` Paris`
- `france_capital` (direct_capital): top5 ` Paris`, ` the`, ` a`, ` one`, ` known`
- `germany_association` (ambiguous_association): top5 ` the`, ` Germany`, ` Bavaria`, ` Munich`, ` Berlin`
- `germany_association_plain` (ambiguous_association): top5 ` Munich`, ` Berlin`, ` Stuttgart`, ` Cologne`, ` Frankfurt`
- `germany_famous_city` (near_match): top5 ` a`, ` the`, ` known`, ` one`, ` home`
- `germany_capital` (direct_capital): top5 ` Berlin`, ` the`, ` a`, ` one`, ` Frankfurt`
- `italy_association` (ambiguous_association): top5 ` the`, ` Rome`, ` Italy`, ` Naples`, ` Bologna`
- `spain_association` (ambiguous_association): top5 ` the`, ` Madrid`, ` Spain`, ` Andalusia`, ` Catalonia`
- `syntax_stall_control` (syntax_control): top5 ` the`, ` “`, ` that`, ` "`, ` a`
- `generic_stall_control` (syntax_control): top5 ` the`, ` a`, ` “`, ` that`, ` `

## Best Same-Country Effects

- `germany_commitment_combo_top8` on `germany_association`: applied=16/16; delta_margin=5.480; expected_delta=3.714; stall_delta=-1.766; wrong_delta=0.343; specificity=-0.071; expected_beats_stall=True
- `spain_commitment_combo_top8` on `spain_association`: applied=16/16; delta_margin=4.493; expected_delta=2.121; stall_delta=-2.373; wrong_delta=-0.972; specificity=-0.614; expected_beats_stall=True
- `italy_commitment_combo_top8` on `italy_association`: applied=16/16; delta_margin=4.216; expected_delta=2.444; stall_delta=-1.772; wrong_delta=1.071; specificity=0.055; expected_beats_stall=True
- `italy_commitment_combo_top5` on `italy_association`: applied=10/10; delta_margin=3.738; expected_delta=1.944; stall_delta=-1.793; wrong_delta=0.734; specificity=-0.136; expected_beats_stall=True
- `spain_commitment_combo_top3` on `spain_association`: applied=6/6; delta_margin=3.494; expected_delta=1.610; stall_delta=-1.884; wrong_delta=-1.407; specificity=-1.266; expected_beats_stall=True
- `germany_single_stall_r6_l17_p9_f405_zero` on `germany_association`: applied=1/1; delta_margin=3.442; expected_delta=2.374; stall_delta=-1.069; wrong_delta=0.261; specificity=-0.428; expected_beats_stall=True
- `spain_commitment_combo_top5` on `spain_association`: applied=10/10; delta_margin=3.267; expected_delta=0.637; stall_delta=-2.630; wrong_delta=-1.777; specificity=-1.620; expected_beats_stall=True
- `spain_commitment_combo_top1` on `spain_association`: applied=2/2; delta_margin=2.805; expected_delta=1.369; stall_delta=-1.436; wrong_delta=-0.486; specificity=-1.292; expected_beats_stall=True
- `spain_single_stall_r1_l16_p9_f436_zero` on `spain_association`: applied=1/1; delta_margin=2.745; expected_delta=1.634; stall_delta=-1.111; wrong_delta=0.017; specificity=-0.595; expected_beats_stall=True
- `spain_stall_positive_top1_zero` on `spain_association`: applied=1/1; delta_margin=2.745; expected_delta=1.634; stall_delta=-1.111; wrong_delta=0.017; specificity=-0.595; expected_beats_stall=True
- `italy_commitment_combo_top3` on `italy_association`: applied=6/6; delta_margin=2.690; expected_delta=1.814; stall_delta=-0.876; wrong_delta=1.013; specificity=-1.372; expected_beats_stall=True
- `italy_stall_positive_top3_zero` on `italy_association`: applied=3/3; delta_margin=2.647; expected_delta=2.284; stall_delta=-0.364; wrong_delta=1.514; specificity=-0.456; expected_beats_stall=True
- `italy_single_stall_r2_l16_p9_f2361_zero` on `italy_association`: applied=1/1; delta_margin=2.544; expected_delta=2.515; stall_delta=-0.029; wrong_delta=1.651; specificity=-0.450; expected_beats_stall=True
- `spain_stall_positive_top3_zero` on `spain_association`: applied=3/3; delta_margin=2.533; expected_delta=0.967; stall_delta=-1.565; wrong_delta=-0.712; specificity=-0.776; expected_beats_stall=True
- `italy_stall_positive_top8_zero` on `italy_association`: applied=8/8; delta_margin=2.510; expected_delta=1.568; stall_delta=-0.942; wrong_delta=0.295; specificity=-0.752; expected_beats_stall=True
- `germany_stall_positive_top8_zero` on `germany_association`: applied=8/8; delta_margin=2.491; expected_delta=0.804; stall_delta=-1.687; wrong_delta=-1.086; specificity=-0.600; expected_beats_stall=False

## Leakage Screen

- `germany_commitment_combo_top8` on `italy_association` (italy/ambiguous_association): applied=16/16; delta_margin=0.667; wrong_delta=7.261; specificity=-9.230
- `germany_commitment_combo_top8` on `spain_association` (spain/ambiguous_association): applied=16/16; delta_margin=0.103; wrong_delta=7.162; specificity=-10.097
- `germany_commitment_combo_top8` on `france_association` (france/ambiguous_association): applied=16/16; delta_margin=0.341; wrong_delta=6.731; specificity=-7.951
- `spain_commitment_combo_top8` on `germany_association` (germany/ambiguous_association): applied=16/16; delta_margin=-0.606; wrong_delta=5.953; specificity=-8.583
- `spain_answer_positive_top8_double` on `germany_association` (germany/ambiguous_association): applied=8/8; delta_margin=-0.706; wrong_delta=5.787; specificity=-7.235
- `germany_answer_positive_top8_double` on `spain_association` (spain/ambiguous_association): applied=8/8; delta_margin=-0.598; wrong_delta=5.355; specificity=-7.174
- `germany_answer_positive_top8_double` on `italy_association` (italy/ambiguous_association): applied=8/8; delta_margin=-0.511; wrong_delta=5.267; specificity=-6.541
- `germany_answer_positive_top8_double` on `france_association` (france/ambiguous_association): applied=8/8; delta_margin=-0.780; wrong_delta=4.832; specificity=-5.776
- `spain_commitment_combo_top8` on `france_association` (france/ambiguous_association): applied=16/16; delta_margin=0.478; wrong_delta=4.678; specificity=-5.551
- `germany_answer_positive_top5_double` on `spain_association` (spain/ambiguous_association): applied=5/5; delta_margin=-0.471; wrong_delta=4.677; specificity=-6.198
- `spain_answer_positive_top8_double` on `france_association` (france/ambiguous_association): applied=8/8; delta_margin=0.192; wrong_delta=4.520; specificity=-4.460
- `spain_commitment_combo_top8` on `italy_association` (italy/ambiguous_association): applied=16/16; delta_margin=0.531; wrong_delta=4.488; specificity=-5.305
- `germany_answer_positive_top5_double` on `france_association` (france/ambiguous_association): applied=5/5; delta_margin=-0.413; wrong_delta=4.439; specificity=-4.893
- `germany_answer_positive_top5_double` on `italy_association` (italy/ambiguous_association): applied=5/5; delta_margin=-0.146; wrong_delta=4.348; specificity=-5.305
- `germany_commitment_combo_top5` on `france_association` (france/ambiguous_association): applied=10/10; delta_margin=-0.375; wrong_delta=4.324; specificity=-5.410
- `spain_answer_positive_top8_double` on `italy_association` (italy/ambiguous_association): applied=8/8; delta_margin=-0.717; wrong_delta=4.188; specificity=-4.391

## Aggregate Screen

- `france__control__syntax_control` n=56; applied_rows=56; mean delta_margin=0.145; positive=31/56; expected beats stall=0/56; mean wrong_delta=0.083; mean specificity=-0.019
- `france__france__ambiguous_association` n=56; applied_rows=28; mean delta_margin=0.083; positive=10/28; expected beats stall=5/28; mean wrong_delta=-0.526; mean specificity=-0.464
- `france__france__direct_capital` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=28/28; mean wrong_delta=0.000; mean specificity=0.000
- `france__france__near_match` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=0/28; mean wrong_delta=0.000; mean specificity=0.000
- `france__germany__ambiguous_association` n=56; applied_rows=28; mean delta_margin=-0.801; positive=8/28; expected beats stall=0/28; mean wrong_delta=0.426; mean specificity=-2.429
- `france__germany__direct_capital` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=28/28; mean wrong_delta=0.000; mean specificity=0.000
- `france__germany__near_match` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=0/28; mean wrong_delta=0.000; mean specificity=0.000
- `france__italy__ambiguous_association` n=28; applied_rows=28; mean delta_margin=-0.074; positive=9/28; expected beats stall=0/28; mean wrong_delta=0.824; mean specificity=-1.836
- `france__spain__ambiguous_association` n=28; applied_rows=28; mean delta_margin=-0.611; positive=8/28; expected beats stall=0/28; mean wrong_delta=0.465; mean specificity=-2.193
- `germany__control__syntax_control` n=56; applied_rows=56; mean delta_margin=0.401; positive=32/56; expected beats stall=0/56; mean wrong_delta=0.155; mean specificity=-0.006
- `germany__france__ambiguous_association` n=56; applied_rows=36; mean delta_margin=-0.246; positive=6/36; expected beats stall=11/36; mean wrong_delta=1.128; mean specificity=-1.642
- `germany__france__direct_capital` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=28/28; mean wrong_delta=0.000; mean specificity=0.000
- `germany__france__near_match` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=0/28; mean wrong_delta=0.000; mean specificity=0.000
- `germany__germany__ambiguous_association` n=56; applied_rows=36; mean delta_margin=0.247; positive=20/36; expected beats stall=10/36; mean wrong_delta=-0.170; mean specificity=-0.411
- `germany__germany__direct_capital` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=28/28; mean wrong_delta=0.000; mean specificity=0.000
- `germany__germany__near_match` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=0/28; mean wrong_delta=0.000; mean specificity=0.000
- `germany__italy__ambiguous_association` n=28; applied_rows=28; mean delta_margin=-0.002; positive=12/28; expected beats stall=0/28; mean wrong_delta=1.238; mean specificity=-1.915
- `germany__spain__ambiguous_association` n=28; applied_rows=28; mean delta_margin=-0.115; positive=11/28; expected beats stall=0/28; mean wrong_delta=1.179; mean specificity=-2.126
- `italy__control__syntax_control` n=56; applied_rows=56; mean delta_margin=0.138; positive=30/56; expected beats stall=0/56; mean wrong_delta=0.232; mean specificity=-0.071
- `italy__france__ambiguous_association` n=56; applied_rows=35; mean delta_margin=0.113; positive=17/35; expected beats stall=12/35; mean wrong_delta=0.466; mean specificity=-0.754
- `italy__france__direct_capital` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=28/28; mean wrong_delta=0.000; mean specificity=0.000
- `italy__france__near_match` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=0/28; mean wrong_delta=0.000; mean specificity=0.000
- `italy__germany__ambiguous_association` n=56; applied_rows=35; mean delta_margin=0.026; positive=15/35; expected beats stall=7/35; mean wrong_delta=0.256; mean specificity=-0.802
- `italy__germany__direct_capital` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=28/28; mean wrong_delta=0.000; mean specificity=0.000
- `italy__germany__near_match` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=0/28; mean wrong_delta=0.000; mean specificity=0.000
- `italy__italy__ambiguous_association` n=28; applied_rows=28; mean delta_margin=0.858; positive=18/28; expected beats stall=7/28; mean wrong_delta=0.284; mean specificity=-0.353
- `italy__spain__ambiguous_association` n=28; applied_rows=28; mean delta_margin=0.092; positive=16/28; expected beats stall=0/28; mean wrong_delta=0.334; mean specificity=-0.892
- `spain__control__syntax_control` n=56; applied_rows=56; mean delta_margin=0.240; positive=32/56; expected beats stall=0/56; mean wrong_delta=0.167; mean specificity=0.003
- `spain__france__ambiguous_association` n=56; applied_rows=34; mean delta_margin=-0.077; positive=8/34; expected beats stall=8/34; mean wrong_delta=0.700; mean specificity=-1.182
- `spain__france__direct_capital` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=28/28; mean wrong_delta=0.000; mean specificity=0.000
- `spain__france__near_match` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=0/28; mean wrong_delta=0.000; mean specificity=0.000
- `spain__germany__ambiguous_association` n=56; applied_rows=34; mean delta_margin=-0.254; positive=4/34; expected beats stall=6/34; mean wrong_delta=0.822; mean specificity=-1.647
- `spain__germany__direct_capital` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=28/28; mean wrong_delta=0.000; mean specificity=0.000
- `spain__germany__near_match` n=28; applied_rows=0; mean delta_margin=0.000; positive=0/28; expected beats stall=0/28; mean wrong_delta=0.000; mean specificity=0.000
- `spain__italy__ambiguous_association` n=28; applied_rows=28; mean delta_margin=-0.034; positive=12/28; expected beats stall=0/28; mean wrong_delta=0.691; mean specificity=-1.222
- `spain__spain__ambiguous_association` n=28; applied_rows=28; mean delta_margin=1.069; positive=20/28; expected beats stall=10/28; mean wrong_delta=-0.227; mean specificity=-0.526

## Individual Feature Candidates

- `germany_single_stall_r6_l17_p9_f405_zero` (single_stall_suppression): score=3.015; same_delta=3.442; same_specificity=-0.428; cross_wrong=0.000; control_delta=0.000; applied=6/12
- `italy_single_stall_r2_l16_p9_f2361_zero` (single_stall_suppression): score=2.094; same_delta=2.544; same_specificity=-0.450; cross_wrong=0.000; control_delta=0.000; applied=6/12
- `spain_single_stall_r1_l16_p9_f436_zero` (single_stall_suppression): score=2.023; same_delta=2.745; same_specificity=-0.595; cross_wrong=0.127; control_delta=0.000; applied=6/12
- `spain_single_answer_r3_l12_p9_f1343_double` (single_answer_amplification): score=0.445; same_delta=0.322; same_specificity=0.168; cross_wrong=-0.053; control_delta=0.045; applied=6/12
- `italy_single_stall_r7_l7_p9_f89_zero` (single_stall_suppression): score=0.157; same_delta=0.158; same_specificity=-0.001; cross_wrong=-0.009; control_delta=0.000; applied=6/12
- `italy_single_answer_r6_l15_p9_f180_double` (single_answer_amplification): score=0.125; same_delta=0.191; same_specificity=0.070; cross_wrong=0.137; control_delta=-0.002; applied=6/12
- `germany_single_stall_r5_l7_p9_f89_zero` (single_stall_suppression): score=0.012; same_delta=0.082; same_specificity=-0.070; cross_wrong=-0.002; control_delta=0.000; applied=6/12
- `italy_single_stall_r8_l10_p9_f26_zero` (single_stall_suppression): score=-0.043; same_delta=0.045; same_specificity=-0.031; cross_wrong=0.056; control_delta=0.000; applied=6/12
- `italy_single_answer_r8_l11_p9_f2745_double` (single_answer_amplification): score=-0.101; same_delta=0.200; same_specificity=-0.028; cross_wrong=0.232; control_delta=0.041; applied=6/12
- `spain_single_answer_r6_l10_p9_f556_double` (single_answer_amplification): score=-0.111; same_delta=0.942; same_specificity=-0.043; cross_wrong=0.929; control_delta=0.080; applied=6/12
- `italy_single_answer_r2_l12_p9_f1884_double` (single_answer_amplification): score=-0.113; same_delta=0.098; same_specificity=-0.146; cross_wrong=0.034; control_delta=0.032; applied=6/12
- `france_single_stall_r7_l0_p9_f774_zero` (single_stall_suppression): score=-0.123; same_delta=0.077; same_specificity=-0.058; cross_wrong=0.142; control_delta=0.000; applied=6/12
- `italy_single_stall_r3_l10_p9_f9037_zero` (single_stall_suppression): score=-0.135; same_delta=0.136; same_specificity=-0.271; cross_wrong=-0.082; control_delta=0.000; applied=6/12
- `france_single_stall_r5_l7_p9_f89_zero` (single_stall_suppression): score=-0.142; same_delta=-0.002; same_specificity=-0.140; cross_wrong=-0.043; control_delta=0.000; applied=6/12
- `spain_single_stall_r6_l7_p9_f89_zero` (single_stall_suppression): score=-0.166; same_delta=-0.026; same_specificity=-0.140; cross_wrong=-0.010; control_delta=0.000; applied=6/12
- `germany_single_stall_r8_l10_p9_f26_zero` (single_stall_suppression): score=-0.190; same_delta=-0.017; same_specificity=-0.101; cross_wrong=0.072; control_delta=0.000; applied=6/12

## Interpretation

The high bar is not merely moving one city token or suppressing `the`. A meaningful result should show analogous same-country movement across country-answer prompts while failing the obvious confounds: wrong-city induction and syntax-control movement.

Rows with `applied=0/...` are positional non-applications, not evidence that the representation is absent. This screen therefore treats them as a transfer-design limitation and bases the verdict on applied rows.
