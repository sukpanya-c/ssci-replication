# Results Summary

This file is generated from the current output tables and model summaries. It is intended as a drafting aid for the manuscript and submission package, not as a replacement for the underlying analysis files.

## Key empirical numbers to cite

- Break diagnostics: mean FX depreciation rises from 0.051 to 1.297, while mean monthly inflation rises from 0.330 to 1.531 and mean CPI inflation (y/y) rises from 4.087 to 20.869.
- Benchmark fx_dep -> inflation_mom pass-through at horizon 0 rises from -0.049 [-0.137, 0.039] in the pre-2022 sample to 0.494 [0.397, 0.591] in the post-2022 sample. The post-2022 horizon-1 estimate is 0.438 [0.279, 0.596].
- Oil_shock -> fx_dep becomes positive in the stressed regime, with post-2022 estimates of 0.043 [0.012, 0.074] at horizon 0 and 0.066 [0.004, 0.128] at horizon 2.
- Oil_shock -> inflation_mom remains positive in the stressed regime and reaches 0.078 [0.027, 0.130] by horizon 3.
- TVP-SV-VAR impact responses are strongest on 2022-03-31 for oil_shock -> fx_dep (0.070 [-0.172, 0.362]) and oil_shock -> inflation_mom (-0.014 [-0.234, 0.180]). By 2023-10-31 they are smaller but still positive, and by 2026-02-28 they are near zero for oil_shock -> fx_dep (0.039 [-0.081, 0.188]) and oil_shock -> inflation_mom (-0.024 [-0.183, 0.078]).

## Interpretation

The strongest supported result remains the post-2022 amplification of exchange-rate pass-through. In the benchmark local projections, depreciation becomes far more inflationary after the January 2022 break, which is the cleanest evidence for the paper's debt-distress-amplified pass-through argument.

The oil shock channel matters mainly through the exchange rate and only in the stressed regime. That is why the manuscript should continue to frame the contribution as state-dependent shock transmission rather than as a broad claim that geopolitical risk directly causes Laos inflation every month.

## Robustness summary

- Alternative inflation response: post-2022 fx_dep -> CPI_Inf at horizon 1 is 0.543 [0.213, 0.872], which preserves the stronger stressed-regime pass-through result on a slower-moving inflation measure.
- Truncated sample check: post-2022 fx_dep -> inflation_mom at horizon 0 remains 0.512 [0.407, 0.617] when the sample is truncated at 2025-12.
- Alternative GPR construction: the monthly GPR_ME_alt -> oil estimate is -5.467 (SE 10.253, p=0.594) in the pre-2022 sample and -0.030 (SE 4.040, p=0.994) in the post-2022 sample, so the alternative construction does not materially strengthen the direct monthly GPR-to-oil relation.

## What should not be overstated

- The direct reduced-form monthly GPR -> oil link is weak: the GPR_ME coefficient is -6.568 (p=0.605) before 2022 and 0.060 (p=0.990) after 2022.
- The main contribution is not a sharp causal identification of Middle East conflict on Laos inflation.
- The February 2026 scenario is a sample-endpoint IRF date, not realized ex post evidence.

## Generated paper-ready outputs

- Paper tables directory: `output/paper_tables`
- Paper figures directory: `output/paper_figures`
