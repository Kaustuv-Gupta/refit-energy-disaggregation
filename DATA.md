# Data Coverage and Quality

This report separates the **raw House 1 cleaning exercise** from the **official cleaned data used for EDA and modeling**. Raw counts and cleaning diagnostics below come from saved outputs in [notebook 01](notebooks/01_data_cleaning.ipynb). Additional timestamp and window-selection audits were calculated from the supplied CSVs while preparing these reports.

## Sources, units and provenance

The assignment uses `REFIT_RAW_081116.7z` and `CLEAN_REFIT_081116.7z`. The dataset describes aggregate and appliance-level active power in W from 20 UK homes, nominally sampled every eight seconds. One-minute analysis values are arithmetic mean W; kWh values integrate those minute means.

The [raw-data documentation](https://pure.strath.ac.uk/ws/portalfiles/portal/62090176/RAW_READ_ME_081116.txt) distinguishes unavailable sensor readings encoded as zero in Part 1 from NaN in Part 2. It also describes asynchronous sensor updates and irregular recording. Zero therefore cannot always be interpreted as genuine off-state consumption.

The official cleaned release already applies timestamp/appliance corrections and replaces appliance spikes above 4,000 W with zero. Its `Issues` flag indicates that summed appliance readings exceed aggregate power. See the [cleaned-data documentation](https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt). The [dataset paper](https://www.nature.com/articles/sdata2016122) describes forward filling gaps shorter than two minutes and zero filling longer gaps. These processed labels can contain artificial off states; observed zeros do not establish true appliance inactivity.

## Raw House 1: files and findings

| Finding | RAW_House1_Part1.csv | RAW_House1_Part2.csv |
| --- | ---: | ---: |
| Rows | 4,377,877 | 4,140,886 |
| Columns | 12 | 12 |
| First raw Time | 2013-10-09 14:06:17 | 2014-06-27 21:57:50 |
| Last raw Time | 2014-10-01 01:27:07 | 2015-07-10 11:56:32 |
| Invalid parsed timestamps | 0 | 0 |
| Exact duplicate rows | 595,935 | 0 |
| Duplicate timestamp rows beyond the first | 645,514 | 0 |
| Backward timestamp steps in file order | 2 | 0 |
| Median interval between sorted unique timestamps | 5 seconds | 7 seconds |
| 95th percentile unique-timestamp interval | 15 seconds | 8 seconds |
| Gaps longer than 60 seconds between unique timestamps | 780 | 3,603 |

The files overlap from late June to early October 2014. After timestamp consolidation within each part, combining them introduces 116,037 duplicate timestamp rows, including five exact duplicates. Consolidating again leaves **7,757,212 unique timestamps** spanning 2013-10-09 14:06:17 to 2015-07-10 11:56:32.

Across the combined sorted unique timestamps, the median interval is seven seconds; 390 intervals exceed five minutes. The longest gap is **3,594,413 seconds**, approximately 41.6 days. These statistics describe timestamp availability, not whether every channel has a valid reading. The supporting [timestamp audit](artifacts/refit_outputs/01_raw_timestamp_audit.csv) records the per-file checks.

Part 1 reports no NaNs, but its zero encoding makes that an incomplete measure of availability. Part 2 explicitly contains:

| Channel | Missing rows | Missing fraction |
| --- | ---: | ---: |
| Aggregate | 188,973 | 4.56% |
| Appliance1 | 437,890 | 10.57% |
| Appliance2 | 459,468 | 11.10% |
| Appliance5, washing machine | 497,223 | 12.01% |
| Appliance9 | 1,911,811 | 46.17% |

The raw extremes are implausible for ordinary individual appliances: washer readings reach 32,767 W in Part 1 and 73,725 W in Part 2; other channels reach 98,301 W. Aggregate power reaches 29,159 W in Part 2. They warrant a sensor/encoding audit rather than an assumption that every large reading is genuine.

## Raw cleaning: implemented decisions

1. **Parse and sort.** Convert `Time` with invalid entries coerced to NaT, drop invalid timestamps, and sort chronologically using a stable sort.
2. **Resolve duplicate timestamps.** Convert numeric fields, temporarily mask zeros, take the maximum available reading per power channel and the last available Unix value at each timestamp, then replace missing grouped numeric values with zero. Apply the same rule after concatenating the parts.
3. **Resample.** Set `Time` as the index and calculate one-minute means.
4. **Fill and flag.** Forward fill up to five missing minute slots per column. Remaining missing entries form an `outage_mask`; they stay NaN in `df_cleaned`.
5. **Inspect extremes.** Print minima and maxima, then clip negative numeric values to zero. The current raw code does not remove positive spikes.
6. **Visualize.** Plot the first 1,000 cleaned minutes for aggregate power and Appliance1/Appliance2. The cleaned CSV export is currently commented out.

The saved resampling output reports **115,078** missing minute slots in the most-missing column before filling and **112,374** minute rows with at least one NaN after filling. These are different definitions, so their difference is not a defensible count of completely repaired rows.

### Consequences of the current raw rules

Maximum-based duplicate consolidation favors larger readings and may combine channel values that never occurred in the same original row. It can retain spikes and inflate energy. A last-valid reading or agreement-based rule should be compared before treating the output as trustworthy.

The zero replacement in step 2 converts channel-level missing readings into off states. Consequently, later outage counts chiefly detect empty timestamp bins and can understate sensor-level missingness. The five-minute filling limit also fills the beginning of a longer outage; it is a cap on filled entries, not a policy that fills only gaps whose entire duration is at most five minutes.

The outage mask is retained only in notebook memory, and a matching before-cleaning plot is absent. Raw positive washer maxima remain 16,387.25 W after one-minute resampling. These shortcomings apply to the custom raw-cleaning exercise; that output is not used to train the models.

## Official cleaned data: preprocessing

Notebooks 02 and 03 read the cleaned CSVs in chunks of 250,000 rows. They discard unparseable timestamps, coerce numeric fields, and mask values outside these inclusive ranges:

- Aggregate: **0-25,000 W**.
- Each washer meter: **0-4,000 W**.

House 4's two washer channels are summed only when both are present. Its combined washer demand can legitimately exceed the single-meter 4,000 W bound.

The loader groups readings into minute bins using sums and valid-value counts, then combines these across chunk boundaries before dividing. This avoids giving small partial chunks the same weight as full ones. Aggregate and washer means use their own valid readings in the same minute bin; they are not guaranteed to come from exactly simultaneous sensor samples.

The timeline is expanded to every minute between the first and last retained observations. Empty readings remain NaN, with no additional forward filling or zero filling. Existing zeros and filled values from the official release remain. Minute means are based on available readings without a minimum sample-count threshold, so a sparsely observed minute may still count as valid.

Any positive `Issues` value makes the minute flagged. Missing issue values are currently treated as unflagged; this is an assumption requiring review. EDA counts flagged minutes but includes them in summaries and cycle detection. Models exclude windows containing any flagged minute.

## EDA households and coverage

House 12 has no identified washer channel, and there is no House 14. Coverage endpoints below are minute bins from [02_coverage_quality.csv](artifacts/refit_outputs/02_coverage_quality.csv). Dates differ by home.

| House | Washer channel(s) | First date | Last date | Missing paired minutes | Missing % | Below 20 W among paired minutes % | Flagged minutes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 5 | 2013-10-09 | 2015-07-10 | 116,318 | 12.64 | 98.12 | 16,686 |
| 2 | 2 | 2013-09-17 | 2015-05-28 | 218,914 | 24.62 | 95.31 | 13,458 |
| 3 | 6 | 2013-09-25 | 2015-06-02 | 100,810 | 11.39 | 94.34 | 60,633 |
| 4 | 4, 5 | 2013-10-11 | 2015-07-07 | 124,967 | 13.69 | 98.54 | 24,364 |
| 5 | 3 | 2013-09-26 | 2015-07-06 | 87,220 | 9.34 | 92.10 | 70,888 |
| 6 | 2 | 2013-11-28 | 2015-06-28 | 126,771 | 15.25 | 98.12 | 14,524 |
| 7 | 5 | 2013-11-01 | 2015-07-08 | 111,042 | 12.58 | 91.63 | 35,837 |
| 8 | 4 | 2013-11-01 | 2015-05-10 | 82,063 | 10.27 | 96.68 | 12,331 |
| 9 | 3 | 2013-12-17 | 2015-07-08 | 125,823 | 15.38 | 97.32 | 16,534 |
| 10 | 5 | 2013-11-20 | 2015-06-30 | 94,671 | 11.20 | 91.27 | 13,975 |
| 11 | 3 | 2014-06-03 | 2015-06-30 | 61,848 | 10.95 | 98.65 | 9,775 |
| 13 | 3 | 2014-01-17 | 2015-05-31 | 165,478 | 23.05 | 94.78 | 22,109 |
| 15 | 3 | 2013-12-17 | 2015-07-08 | 111,626 | 13.66 | 97.07 | 11,448 |
| 16 | 5 | 2014-01-10 | 2015-07-08 | 135,663 | 17.33 | 96.30 | 6,317 |
| 17 | 4 | 2014-03-06 | 2015-06-19 | 41,627 | 6.15 | 97.60 | 27,136 |
| 18 | 5 | 2014-03-07 | 2015-05-24 | 37,229 | 5.84 | 98.80 | 34,909 |
| 19 | 2 | 2014-03-06 | 2015-06-20 | 43,779 | 6.46 | 96.13 | 26,075 |
| 20 | 4 | 2014-03-20 | 2015-06-23 | 39,510 | 5.96 | 97.56 | 9,932 |
| 21 | 3 | 2014-03-07 | 2015-07-10 | 66,188 | 9.38 | 96.88 | 34,727 |

There are **14,960,575 household-minute slots**, of which **13,069,028** have both aggregate and washer values. Missing paired minutes total **1,891,547 (12.64%)**. A further quality dimension is the **461,658 issue-flagged minutes**; these can overlap with missing minutes and must not be added to the missing total as though disjoint.

A missing paired minute means aggregate or washer power is unavailable after range filtering; it is not exclusively a recording outage. The pooled below-20-W share is approximately **96.04%**, weighted by valid paired minutes. Below threshold includes low standby draw and is not an exact-zero percentage.

## Model period and filtering losses

Models use Houses 1-3 from **2014-06-01 00:00 to 2014-10-31 23:59**, with 220,320 minute slots per home.

| House | Usable paired, issue-free minutes | Missing or excluded % | Issue-flagged minutes |
| --- | ---: | ---: | ---: |
| 1 | 209,641 | 4.85 | 3,122 |
| 2 | 213,728 | 2.99 | 4,438 |
| 3 | 206,464 | 6.29 | 7,176 |

Source: [03_data_coverage.csv](artifacts/model_results/03_data_coverage.csv). Here `paired_valid_minutes` includes issue exclusion, unlike the similarly named paired-count field in the Seq2Seq coverage file.

Window selection is much stricter than single-minute availability. The [additional selection audit](artifacts/model_results/03_window_selection_audit.csv) uses the current loader, chronological split and Seq2Point eligibility rule:

| Split, pooled Houses 1-3 | Active paired minutes before issue/window filtering | Active paired minutes without their own issue flag | Active eligible midpoints after full-window filtering |
| --- | ---: | ---: | ---: |
| Train | 18,684 | 14,310 | 234 |
| Validation | 3,588 | 2,637 | 184 |
| Test | 3,714 | 2,920 | 114 |

Only **1.25%** of paired active training midpoints and **3.07%** of paired active test midpoints survive. A minute can have a valid unflagged washer label yet be removed because aggregate data are missing or an issue flag occurs elsewhere in its 61-minute input window.

This selection favors off-state periods. Removing flags indiscriminately would admit questionable labels; the next step is to inspect complete cycles and the reasons each window fails, then revise eligibility using justified quality rules.

## Remaining concerns

- Official forward-filled or zero-replaced measurements can resemble long plateaus or genuine off periods. 
- Per-home coverage differs, issue flags are retained in EDA, and House 4 combines two machines. Event frequency and energy differences cannot alone establish efficiency differences.
- Appliance replacement, moved monitors and behavior changes may change label meaning or signatures. They need explicit metadata checks.
- Raw and cleaned timestamp conventions need reconciliation before local-time usage advice.
- The current models are validated only on selected later periods in three known UK homes. Broader and locally collected data are needed before utility deployment.
