# Dhar and Hoch (1996) reconstruction protocol

This is a separate outcome-blind reconstruction of the five one-week,
store-randomized Coupon versus Bonus Buy field tests described in Dhar and Hoch
(1996), *Price Discrimination Using In-Store Merchandising*.

`run_dhar_hoch_1996_outcome_blind.py` permits only `UPC`, `STORE`, `WEEK`,
`QTY`, `PRICE`, `SALE`, and `OK`. It rejects `MOVE`, profit fields, and any
derived outcome. It first checks the published Cheerios 15 oz, week 177 clue,
then scans weeks 151--203 plus adjacent weeks for one-week support.

The search ranks UPC-week candidates using the documented treatment-side
architecture: approximately 43 `B` stores, approximately 43 `C` stores,
approximately 86 participating stores, and no adjacent-week B/C support. It
does not use published sales changes. A populated `SALE` is affirmative
promotion evidence; a blank `SALE` is recorded as missing, never as a control.

Candidate files are not treatment labels. A reconstruction can be frozen only
after product identity, one-week timing, and store-level B/C assignments are
independently supported. The outcome-validation stage remains prohibited until
that freeze passes.
