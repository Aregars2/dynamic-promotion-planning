# Final targeted code audit before manuscript lock

Audit date: 2026-08-13. This was an inspection of the current empirical-Bayes workflow and its persisted artifacts. No PPML model, calibration, optimizer, or policy artifact was rerun or changed.

> **Implementation status (2026-08-13):** The price-reference issue in Section
> 1 has since been implemented in source as a price-consistent policy-profile
> change. The corrected policy artifact will be written separately under
> `artifacts/policy/empirical_bayes_price_consistent/` when Notebook 06 is
> rerun; the audit findings describe the pre-correction artifact.

## 1. Price-reference audit

**MODEL ISSUE — STOP BEFORE CHANGING.** The common-origin PPML baseline and policy-profit calculation use different price references, and the gap is material.

`build_common_origin_ex_ante_frame` takes the last pre-origin row of each store–product panel, freezes that row's `regular_price` and `price_imputed_indicator`, and sets `model_unit_price = regular_price`, promotion = 0, and post-promotion = 0. Thus `Q_base,it` is store-specific baseline demand evaluated at its origin-known regular price. Sources: [demand.py](../src/dynamic_promotion_planning/demand.py#L618) and [demand.py](../src/dynamic_promotion_planning/demand.py#L687).

Policy profit instead uses the scalar product-level `regular_price` in the behavioral draws. It is the median `regular_price_model` among regular, non-post-promotion calibration observations; selected-policy price is `p_i0 (1-d_it)`. Sources: [calibration.py](../src/dynamic_promotion_planning/calibration.py#L1170) and [policy.py](../src/dynamic_promotion_planning/policy.py#L535).

| UPC | Policy p_i0 | Frozen mean | Median | Min | Max | (p_i0 − median) / median |
|---|---:|---:|---:|---:|---:|---:|
| 1600066510 | 2.290 | 2.506 | 2.490 | 2.290 | 2.730 | -8.0% |
| 1600066590 | 3.150 | 3.272 | 3.250 | 2.980 | 3.370 | -3.1% |
| 1600066610 | 3.220 | 3.375 | 3.350 | 3.160 | 3.470 | -3.9% |
| 3800000110 | 1.644 | 2.169 | 2.150 | 2.050 | 2.230 | -23.5% |
| 3800000120 | 2.060 | 2.856 | 2.840 | 2.480 | 2.940 | -27.5% |
| 3800000127 | 2.620 | 3.520 | 3.490 | 3.290 | 3.630 | -24.9% |
| 3800001520 | 3.420 | 4.164 | 4.130 | 3.890 | 4.290 | -17.2% |
| 3800001611 | 2.850 | 3.333 | 3.310 | 3.150 | 3.420 | -13.9% |

Using the persisted EB policy and common-origin prediction artifacts, absolute relative gaps have mean **15.3%**, median **15.5%**, 95th percentile **26.6%**, and maximum **27.5%**. This is too large to describe as harmless standardization: baseline quantities are generated at panel-specific prices while revenue/cost are valued at a lower product-level calibration-period reference. No change was made.

## 2. Policy-definition audit

**PASS.** `piM` is a sequential capacity-constrained weekly policy. Each weekly choice conditions on inherited draw-specific state, and the selected action then updates `I_(t+1)=rI_t+d_t`. It therefore propagates displacement from earlier promotions, while omitting only the current candidate's continuation cost at choice time. Source: [policy.py](../src/dynamic_promotion_planning/policy.py#L940).

`piN` uses the explicit surrogate transition `I_(t+1)=rI_t`: candidate depth is not added while its contemporaneous price/promotion effects remain. With fresh start `I_1=0`, its surrogate state remains zero. Its selected calendar is replayed under the full transition. Sources: [policy.py](../src/dynamic_promotion_planning/policy.py#L535), [policy.py](../src/dynamic_promotion_planning/policy.py#L1404), and [analysis.toml](../config/analysis.toml).

Manuscript wording: *The myopic policy conditions each weekly choice on displacement inherited from earlier promotions but does not internalize the continuation effect of the current action. The naive forward-looking policy optimizes the full calendar under the surrogate transition `I_(t+1)=rI_t` and is evaluated under the common full transition.*

## 3. Gross-margin units

**PASS.** Raw Dominick's `PROFIT` is retained as `gross_margin_pct_observed` only where transaction price is observed. In the live data it is percentage points (processed-data 95th percentile 33.95). [Notebook 01](../notebooks/01_sample_construction.ipynb#L991).

`_construct_observed_cost` detects percentage-point input when the finite absolute 95th percentile exceeds 1.5; live data therefore use `g_ist=PROFIT_ist/100` and `c_ist=p_ist(1-g_ist)`. Valid margins satisfy `-1.00 <= g <= 0.95`; cost must also be finite and strictly positive. Missing/invalid costs are set missing and excluded from the product median—never imputed or forward-filled. Sources: [calibration.py](../src/dynamic_promotion_planning/calibration.py#L449) and [analysis.toml](../config/analysis.toml).

The policy helper defensively accepts either percent or fraction input. [policy.py](../src/dynamic_promotion_planning/policy.py#L264). Each selected product has a finite stored cost. The code permits pooled-median cost only if a product median is non-finite/non-positive; none of the eight stored costs equals the pooled median (2.342964), so this fallback was not used.

Manuscript wording: *We convert the observed Dominick's gross-margin field from percentage points to a fraction, set `c_ist=p_ist(1-g_ist)`, and exclude missing, invalid (`g` outside [-1,0.95]), or non-positive implied-cost observations when forming product-level cost medians.*

## 4. Behavioral estimator definitions

**PASS.** The raw product bootstrap resamples eligible `store_upc` panels with replacement; a sampled panel's full price/event moment contribution is repeated by its multiplicity. [calibration.py](../src/dynamic_promotion_planning/calibration.py#L763). Events use residual changes relative to mean weeks -3,-2,-1 (at least two observed preweeks); post-lag displacement is the negative residual event effect. [calibration.py](../src/dynamic_promotion_planning/calibration.py#L642) and [calibration.py](../src/dynamic_promotion_planning/calibration.py#L692).

For each bootstrap draw:
- `beta_price=sum(wxy)/sum(wx²)` on regular/non-post panel price moments; `epsilon=clip(-beta_price,0.10,5.00)`.
- `gamma` is the weighted mean of `current_lift_log - beta_price*current_price_change`, clipped to [-1,3].
- `psi` is the weighted no-intercept slope of post-1 dip on event depth, clipped to [0,3].
- `r=clip(post2_slope/post1_slope,0.05,0.95)` only if both slopes are finite, post-1 slope >=0.05, and `0 < post2 <= 1.25*post1`. Otherwise the draw expands over {0.15,0.35,0.60} plus clipped pooled persistence. A failed product bootstrap uses the pooled-fallback grid.

Sources: [calibration.py](../src/dynamic_promotion_planning/calibration.py#L763), [calibration.py](../src/dynamic_promotion_planning/calibration.py#L1013), [calibration.py](../src/dynamic_promotion_planning/calibration.py#L1111), and [analysis.toml](../config/analysis.toml). Product bootstrap fallback applies for fewer than 4 common panels or fewer than 80 depth events. EB subsequently mixes aligned product/pooled parent draws; it does not alter the raw estimators.

## 5. Eligibility and action-support rules

**PASS.** Of 344 cleaned products, sequential filters retain 175 with >=100 training weeks, 133 with >=40 calibration weeks, 106 with >=40 test weeks, 105 after >=10 training stores, >=8 distinct prices, >=30 promotion rows, and >=100 regular rows, and 97 after imputed-price share <=25%; the >=10% observed price-range requirement leaves 97. The eight final products are the highest-training-unit eligible products. Sources: [demand.py](../src/dynamic_promotion_planning/demand.py#L99), [analysis.toml](../config/analysis.toml), and `results/empirical_bayes/tables/product_selection_flow.csv`.

Stores require >=100/40/40 train/calibration/test weeks and coverage of at least `ceil(.75*8)=6` selected products in each period; panels then require >=100/30/30 weeks. [Notebook 02](../notebooks/02_demand_estimation.ipynb#L2253).

Week 219 is excluded because every price and movement value is zero (16,971 rows, 86 stores, 250 UPCs): a structural-zero anomaly. [Notebook 01](../notebooks/01_sample_construction.ipynb#L380) and [analysis.toml](../config/analysis.toml).

Recorded-promotion observations with depth in [0.05,0.60] are mapped to `round(depth/.05)*.05`, clipped to that interval; actions equal bin centres. A positive bin needs >=15 rows and >=3 store–product panels. Retain at most three, ordered by support count, then panel count, then shallower bin centre. Sources: [action_support.py](../src/dynamic_promotion_planning/action_support.py#L13), [action_support.py](../src/dynamic_promotion_planning/action_support.py#L31), and [action_support.py](../src/dynamic_promotion_planning/action_support.py#L131).

## 6. Policy action granularity

**PASS.** The decision is one product-level promotion depth per product-week, common across stores. Store heterogeneity enters only through the store-level common-origin baseline demand that is aggregated to a product-week policy baseline. Sources: [Notebook 06](../notebooks/06_policy_optimization.ipynb#L410) and [policy.py](../src/dynamic_promotion_planning/policy.py#L535).

Optional limitation: *The policy captures store heterogeneity in baseline demand but not store-level promotional targeting.*

## 7. 48-week baseline construction

**PASS.** All 48 weeks come from one PPML fit to pre-origin observations. The complete panel-week grid advances only deterministic calendar controls; origin-known regular price and row-quality status are frozen, and promotion/post-promotion/depth are zero. No realized future sales, price, promotion, missingness, or availability enters weeks 13–48. Sources: [demand.py](../src/dynamic_promotion_planning/demand.py#L618) and [demand.py](../src/dynamic_promotion_planning/demand.py#L687).

## 8. Calibration-versus-cooldown extrapolation

**WORDING FIX ONLY.** Calibration events require a three-week pre-gap and three-week post-gap, while feasible policy calendars require only a two-week cooldown after a promotion. Sources: [calibration.py](../src/dynamic_promotion_planning/calibration.py#L608) and [policy.py](../src/dynamic_promotion_planning/policy.py#L497).

Use: *The reduced-form displacement transition is calibrated from isolated events but is applied to all feasible policy calendars, including sequences permitted by the two-week cooldown.*

## 9. Uncertainty-table point-estimate definition

**WORDING FIX ONLY.** The canonical fixed-calendar uncertainty export has no `Point estimate` column. It has paired parent-bootstrap contrasts and p05, p50, p95 only. Sources: [run_final_robustness_reporting.py](../scripts/run_final_robustness_reporting.py#L101) and `results/empirical_bayes/robustness/fixed_calendar_behavioral_uncertainty_peak_summary.csv`. Therefore a Table 5 column called `Point estimate` is assembled outside the canonical export and should be removed or explicitly defined.

If it is meant to summarize the parent-bootstrap distribution, label it **Weighted mean** (parent probabilities are equal). For `delta_total`, mean / median / p05 / p95 are:

| B | Mean | Median | p05 | p95 |
|---:|---:|---:|---:|---:|
| 1 | 873.75 | 865.77 | 249.70 | 1576.81 |
| 2 | 1454.99 | 1163.11 | 75.66 | 4236.71 |
| 3 | 1579.79 | 1456.03 | 580.62 | 2891.32 |
| 8 | 1672.95 | 1549.84 | 678.07 | 2974.00 |

These are behavioral-parameter uncertainty conditional on selected calendars, not confidence intervals or integrated main-policy values.

## 10. Required action before manuscript lock

1. **MODEL ISSUE — STOP BEFORE CHANGING:** decide how to reconcile or explicitly justify the material baseline-price/profit-price mismatch in Section 1. Any scientific change would require downstream policy recomputation; none was made here.
2. **WORDING FIX ONLY:** remove/rename any Table 5 `Point estimate` that is not explicitly defined. The canonical output supports p05/p50/p95; call an added mean `Weighted mean`.
3. **WORDING FIX ONLY:** state the isolated-event/two-week-cooldown extrapolation plainly.

All other audited implementation claims match the frozen current code.
