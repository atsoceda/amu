# Fixed-b positive control + near-boundary screen

- Candidates screened: 12
- Near-boundary (gap ≥ -1.5): 2
- Test pairs: med_doctor_surgeon, science_scientist_biologist, psych_psychologist_counselor, science_scientist_chemist
- Best mix: 0.25
- Assay validated: False

## Closest baseline gaps (target − source under fixed b)

- `science_scientist_biologist`: gap=-0.750 near=True free_src='a scientist.' free_tgt='a biologist.'
- `science_scientist_chemist`: gap=-1.125 near=True free_src='a scientist.' free_tgt='a chemist.'
- `legal_lawyer_solicitor`: gap=-3.625 near=False free_src='a lawyer.' free_tgt='a lawyer.'
- `med_doctor_surgeon`: gap=-4.625 near=False free_src='a doctor.' free_tgt='a surgeon.'
- `legal_lawyer_barrister`: gap=-4.750 near=False free_src='a lawyer.' free_tgt='a lawyer.'
- `psych_psychologist_counselor`: gap=-5.125 near=False free_src='a psychologist.' free_tgt='a counselor.'
- `psych_psychologist_therapist`: gap=-5.750 near=False free_src='a psychologist.' free_tgt='a counselor.'
- `teach_teacher_tutor`: gap=-5.750 near=False free_src='a teacher.' free_tgt='a psychologist.'

## Oracle patch by mix

- mix=0.25: n=4 match=0.00 (CP95 upper=0.60) ΔΔ mean=0.062 [-0.008,0.133]
- mix=0.5: n=4 match=0.00 (CP95 upper=0.60) ΔΔ mean=0.031 [-0.030,0.092]
- mix=0.75: n=4 match=0.00 (CP95 upper=0.60) ΔΔ mean=0.031 [-0.030,0.092]
- mix=1.0: n=4 match=0.00 (CP95 upper=0.60) ΔΔ mean=0.062 [-0.008,0.133]

## Random patch controls

- mix=0.25: n=20 match=0.00 ΔΔ mean=-0.025 [-0.091,0.041]
- mix=0.5: n=20 match=0.00 ΔΔ mean=-0.106 [-0.225,0.012]
- mix=0.75: n=20 match=0.00 ΔΔ mean=-0.062 [-0.217,0.092]
- mix=1.0: n=20 match=0.00 ΔΔ mean=-0.044 [-0.191,0.103]

Oracle MLP-in activation patching under fixed native article does not clearly move nouns; fixed-b assay may be insensitive or model may lack separable noun control at these sites.
