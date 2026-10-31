# Source provenance and scientific scope

The NIMS source is **Dataset of La(Fe,Si)13-based magnetocaloric alloys**, Anton Bolyachkin and Srinithi Ashok Krishnaswamy (2024), https://doi.org/10.48505/nims.4909, licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

The included `magnetic_data/nims_lafesi_source_1113.csv` is a CSV serialization of the 1113-row source workbook `1-13_dataset.xlsx`. Source record: https://mdr.nims.go.jp/datasets/d7b80863-6b14-41c2-ab01-8debc43945d5?locale=en. File endpoint: https://mdr.nims.go.jp/filesets/a16fd3fa-5839-4799-a69d-2ddef6162e83/download. Retrieved 31 October 2026. CSV conversion and the subsequent cohort, payoff and group fields are modifications for this reanalysis; the source authors do not endorse them. All row IDs are retained. Only article metadata are forward-filled; physical values and processing entries are not imputed across rows.

The HEA input is the empirical strategy portfolio in the author's published version 8.0.0 archive, https://doi.org/10.5281/zenodo.23602541. Its finite-game eligibility and utility definitions are preserved. It is distinct from the full target-auditing cohort.

New code is MIT licensed. The NIMS source and its derivatives retain CC BY 4.0 attribution. Published papers cited in the manuscript are not redistributed. No new alloy measurements are claimed. The H–C case is nested within the NIMS five-dopant case. A fixed catalog maximum is not a chemical optimum or a causal dopant effect.

The original long calibration runs remain available in version 8 and are separately identified as historical evidence. This runner regenerates the new finite-game completion, magnetic data analysis, source-grouped representation check, 4000-world paired acquisition comparison and six figures. It does not claim to rerun the original long calibration studies.
