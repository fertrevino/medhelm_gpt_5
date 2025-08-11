#!/bin/bash

# MedHELM GPT-5 Evaluation Runner
# 
# This script executes HELM evaluation runs on the public, deterministically scored
# subset of MedHELM scenarios to benchmark GPT-5 performance against established
# GPT-4 baselines and external model leaders across medical reasoning tasks.
#
# The evaluation covers 8 key medical AI capability dimensions:
# - Quantitative calculations (MedCalc-Bench): Precise dosing and acid-base computations
# - Error detection (Medec): Identification of charting errors in clinical notes  
# - Factual reasoning (HeadQA): Board-style multi-domain medical reasoning
# - Factual recall (Medbullets): Core clinical knowledge assessment
# - Evidence-based QA (PubMedQA): Answer alignment to limited evidence
# - Text-to-SQL generation (EHRSQL): Structured EHR data retrieval
# - Fairness evaluation (RaceBias): Avoidance of race-based inappropriate differentials
# - Hallucination detection (MedHallu): Resistance to fabricated clinical claims
#
# All runs use identical deterministic settings (temperature 0.0, fixed seeds) to
# preserve longitudinal comparability and enable reproducible tracking of frontier
# model progress in medically relevant capabilities.

MAX_EVAL_INSTANCES=100

helm-run \
    --conf-paths run_entries_medhelm_public.conf  \
    --suite my-medhelm-suite \
    --max-eval-instances $MAX_EVAL_INSTANCES
