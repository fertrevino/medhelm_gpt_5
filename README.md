# MedHELM GPT-5

## Overview
This repository contains reproducibility artifacts for evaluating OpenAI's GPT-5 model on the MedHELM (Medical Holistic Evaluation of Language Models) benchmark as discussed in [this paper](https://www.fertrevino.com/docs/gpt5_medhelm.pdf). MedHELM is a comprehensive evaluation suite designed to assess the medical knowledge and reasoning capabilities of large language models across diverse healthcare domains.

Our work provides the first systematic evaluation of GPT-5's performance on medical tasks, comparing it against GPT-4 baselines and current leaderboard leaders. The evaluation covers eight core medical scenarios spanning quantitative reasoning, factual knowledge, clinical decision-making, and fairness considerations.

## Reproducibility Artifacts
Place or symlink the following into HELM's config search path. In this repo they are stored under `prod_env/` which serves as the default production configuration directory HELM will scan first.
```
custom_client.py                # Custom OpenAI client exposing GPT-5 model ID
run_entries_medhelm_public.conf # Scenarios run configuration file
prod_env/
  model_deployments.yaml        # GPT-5 deployment registration with decoding defaults
  model_metadata.yaml           # Model metadata for reporting and aggregation
  credentials.conf              # key:value API credentials (see below)
```
Deterministic settings: temperature=0.0, top_p=1.0, fixed seed, unchanged prompts.ides the minimal configuration assets and instructions needed to reproduce the GPT-5 integration and results reported in the paper:
“From GPT-4 to GPT-5: Measuring Progress in Medical Language Understanding Through MedHELM.”

Focus: append-only, deterministic evaluation of GPT-5 on the public, objectively scored MedHELM subset (temperature 0.0, fixed seeds, unchanged prompts) for longitudinal comparison with GPT-4 era baselines and external leaderboard leaders.

## Public Deterministic Scenarios
| Scenario | Capability Axis | Metric |
|----------|-----------------|--------|
| MedCalc-Bench | Quantitative medical calculations | Accuracy |
| Medec | Medical error flagging | Accuracy |
| HeadQA | Multi-domain factual reasoning | EM |
| Medbullets | Broad factual recall | EM |
| PubMedQA | Evidence-grounded QA | EM |
| EHRSQL | Text-to-SQL (EHR schema) | Execution Accuracy |
| RaceBias | Fairness probe | EM |
| MedHallu | Hallucination resistance | EM |

## Key GPT-5 Findings (Summary)
- New / tied highs: HeadQA (0.93 new), Medbullets (0.89 new), MedCalc-Bench (0.35 tie).
- Regressions vs best GPT-4 baseline: EHRSQL (-0.14), RaceBias (-0.18), PubMedQA (-0.07), MedHallu (-0.02), Medec (-0.03).
- Mean delta: -0.04 vs best GPT-4; -0.05 vs scenario leader → progress uneven.
- Strengths: multi-domain factual + quantitative reasoning.
- Weaknesses: schema-grounded structured generation, fairness robustness, evidence-calibrated QA, full hallucination suppression.

## System Requirements
- Python 3.9+ (recommended: Conda for environment isolation)
- OpenAI API access with GPT-5 availability
- Sufficient disk space for scenario datasets and evaluation outputs (~2GB)

## Latency Snapshot
- Mean per-instance latency: GPT-5 15.05 s vs leaders 13.56 s (ratio 1.11).
- Faster: MedCalc-Bench (0.50×), Medec (0.67×).
- Slow outliers with quality regressions: HeadQA (16.31×), EHRSQL (8.08×).
- Priority remediation: EHRSQL (quality + efficiency), RaceBias (fairness + stability).

## Repository Directories
- prod_env/: Canonical location of runnable HELM configuration artifacts (used directly without additional flags). You may point HELM here or copy these files into your own config path.
- benchmark_output/: Reference snapshot produced by `helm-run` + `helm-summarize` for the GPT-5 paper evaluation (raw scenario JSON, summary metrics). Use it to:
  - Verify local reproduction (diff your regenerated output against this snapshot).
  - Inspect latency / metric JSON schemas without re-running.
  - Serve as a frozen audit trail (do not edit; regenerate instead).
To regenerate, delete (or move) the folder, then re-run the Minimal Reproduction Workflow; a structurally equivalent directory will be recreated.

## Dependencies
Install all required Python packages via the pinned requirements file:
```bash
pip install -r requirements.txt
```
The file pins `crfm-helm==0.5.6` (core evaluation framework) to ensure metric and normalization parity. **Do not upgrade HELM** for reproducing paper results.

## Environment Setup  
The paper recommends using [Conda](https://anaconda.org/anaconda/conda) for environment isolation to avoid Python + binary dependency conflicts.

## Minimal Reproduction Workflow
1. Create/activate a Python 3.9+ virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Populate `prod_env/credentials.conf` using the actual key:value syntax:
   ```
   openaiApiKey: sk-REPLACE_OPENAI_KEY
   # Optional additional providers (remove lines you do not use):
   anthropicApiKey: sk-ant-REPLACE_ANTHROPIC_KEY
   ```
   **Note**: This file supersedes environment variables for these keys.
4. Place/symlink config artifacts (see Reproducibility Artifacts) onto HELM's search path.
5. Add the repository directory to your PATH so HELM can locate the configuration files:
   ```bash
   export PATH="/path/to/medhelm_gpt_5:$PATH"
   # Or if you're in the repository directory:
   export PATH="$(pwd):$PATH"
   ```
   **Note**: This ensures HELM can find `run_entries_medhelm_public.conf` in the repository root.
6. Execute the HELM evaluation workflow in the following **required order**:
   ```bash
   # Step 1: Run evaluations (resolve datasets, execute inference)
   helm-run --conf-paths run_entries_medhelm_public.conf --suite my-medhelm-suite
   
   # Step 2: Aggregate results (normalize outputs, compute metrics)
   helm-summarize --suite my-medhelm-suite
   
   # Step 3: Launch interactive UI (inspect results, compare models)
   helm-server
   ```
   
   **Important**: The three-step sequence is mandatory:
   - `helm-run`: Resolves datasets and executes inference for configured scenario-model pairs under fixed seeds
   - `helm-summarize`: Aggregates raw outputs into normalized artifacts for visualization and comparison 
   - `helm-server`: Launches local UI to inspect summarized results interactively
   
   You can also use the provided convenience scripts: `./helm_run.sh` followed by `./helm_summarize.sh`
   
   **Alternative approach**: To run with limited instances for testing:
   ```bash
   helm-run --conf-paths run_entries_medhelm_public.conf --suite my-medhelm-suite --max-eval-instances 100
   ```
   
7. Verify JSON scores match paper (e.g., HeadQA 0.93 EM, EHRSQL 0.18 ExecAcc).

## Verification Checklist
- **Determinism**: Repeated runs produce identical metrics (only timestamps differ)
- **Scenario integrity**: No prompt or dataset diffs vs baseline configs
- **Metrics**: Match reported accuracy / EM / execution accuracy values
- **File structure**: Ensure `run_entries_medhelm_public.conf` is in repository root, not `prod_env/`
- **Model naming**: Verify model deployment name (`openai/gpt-5`) matches across all config files

## Citation
If you use these assets or results, please cite:
```
@article{trevino2025medhelm,
  title={From GPT-4 to GPT-5: Measuring Progress in Medical Language Understanding Through MedHELM},
  author={Trevino, Fernando},
  year={2025}
}
```
Also cite the MedHELM benchmark and individual scenario datasets used in your research.

## References
- MedHELM Leaderboard: https://crfm.stanford.edu/helm/medhelm
- HELM Documentation: https://crfm-helm.readthedocs.io
- MedCalc-Bench: https://github.com/ncbi-nlp/MedCalc-Bench
- EHRSQL: https://github.com/glee4810/EHRSQL
- PubMedQA: https://github.com/pubmedqa/pubmedqa
- HEAD-QA: https://huggingface.co/datasets/head_qa
- MedHallu benchmark (paper reference)
- Medec benchmark (paper reference)

## Troubleshooting

### Common Issues
- **Import errors**: Ensure `crfm-helm==0.5.6` is installed and the virtual environment is activated
- **Missing datasets**: HELM will automatically download datasets (HuggingFace, etc.) on first run
- **API rate limits**: OpenAI API rate limits may cause timeouts; HELM includes retry logic
- **Schema validation errors**: Ensure `model_deployments.yaml` matches the model name in `run_entries_medhelm_public.conf`

### Debug Mode  
The custom client includes debug logging. Check console output for:
- `[GPT-5 INPUT]`: Request details and token budgets
- `[GPT-5 OUTPUT]`: Model responses  
- `[GPT-5 USAGE]`: Token usage including reasoning tokens
- `[GPT-5 RAW RESPONSE]`: Full API responses for debugging

## Additional Scenarios (Full Suite)
While the paper focuses on 8 public, deterministically-scored scenarios, the full evaluation suite includes additional scenarios visible in the `benchmark_output/` directory:
- **ACI Bench**: Additional clinical reasoning scenarios
- **MedDialog**: Conversational medical dialogue scenarios (healthcaremagic, icliniq subsets)  
- **MediQA**: Medical question answering tasks
- **MTSamples Replicate**: Medical transcription scenarios

These extended scenarios provide broader coverage but may include subjective scoring or access restrictions not suitable for the core reproducibility study.

## Citation

Please use the the following citation if you refer to these results in your work.

```bibtex
@article{Trevino2025_FromGPT4toGPT5,
  title   = {From GPT-4 to GPT-5: Measuring Progress in Medical Language Understanding Through MedHELM},
  author  = {Fernando Trevino},
  year    = {2025},
  month   = {August 11},
  note    = {PDF available at \url{https://www.fertrevino.com/docs/gpt5_medhelm.pdf}},
}
```
