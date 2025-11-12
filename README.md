Code Ocean pipeline capsule
===========================

This folder contains a minimal capsule to run one session end-to-end:
download the NWB, preprocess, and run HMM cross-validation. It is intended
for batch/array execution by providing `SESSION_ID` per run or a
`parameters.json` sweep in Code Ocean.

Entry point
 - run.sh (expects `SESSION_ID` env var)

Outputs
 - /results/sessions_preprocessed/df_<SESSION_ID>.pkl
 - /results/sessions_hmm_crossval/hmm_<SESSION_ID>.pkl
 - /results/logs/session_<SESSION_ID>.log

How it works
 - download_single_session.py: downloads a single session NWB to /data/sessions_nwb/
 - process_session.py: orchestrates download → preprocess → HMM (fast)
 - It invokes top-level scripts: ../preprocessing_nwb.py and ../hmm_crossvalidation_fast.py

Batch runs
 - Provide parameters.json with objects like {"SESSION_ID": "766640955"}
 - Use Code Ocean parameter sweep to run multiple sessions in parallel

Environment
 - See environment.yml for dependencies similar to ssm_env

Notes
 - Internet access is required (public S3).
 - If you prefer the full (non-fast) HMM, update process_session.py to
   invoke ../hmm_crossvalidation.py instead.




