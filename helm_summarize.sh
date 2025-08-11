#!/bin/bash

# MedHELM Results Aggregation and Summarization
#
# This script processes raw HELM evaluation outputs into normalized artifacts 
# for comparative analysis and visualization of GPT-5 performance across medical
# reasoning benchmarks.
#
# The summarization process:
# 1. Aggregates per-instance raw model outputs from helm-run execution
# 2. Applies scenario-specific metrics (exact match, execution accuracy, numerical accuracy)
# 3. Normalizes results into standardized JSON artifacts for cross-model comparison
# 4. Generates summary statistics and performance deltas relative to established baselines
# 5. Prepares data structures for interactive exploration via helm-server
#
# The --schema parameter is critical for proper result visualization - without it,
# aggregated results will not appear correctly in the HELM server interface.
# The schema ensures consistent metric interpretation and enables longitudinal
# tracking of model performance across MedHELM scenario families.

helm-summarize --suite my-medhelm-suite --schema schema_medhelm.yaml


