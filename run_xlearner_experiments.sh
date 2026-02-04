#!/bin/bash

# Shell script to run XLearner with 30 trials for each cohort IN PARALLEL
# This script runs:
# 1. single_cohort_analysis.py - SHAP value computation (parallel across GPUs)
# 2. clinical_agent.py - hypothesis generation WITH SHAP info
# 3. clinical_agent.py - hypothesis generation WITHOUT SHAP info (baseline)

NUM_TRIALS=30
TOP_N_FEATURES=5
RELATIVE_CHANGE_THRESHOLD=0.05
N_HYPOTHESES=5
# MODEL="gpt-5.1-2025-11-13"
MODEL="gpt-4o-2024-08-06"
# GPU configuration - specify which GPUs to use
GPUS=(0 1 2 3)  # Modify based on available GPUs
NUM_GPUS=${#GPUS[@]}

# List of cohorts to analyze
COHORTS=(
    "crash_2"
    "ist3"
    "sprint"
    "accord"
)

# Cohorts with trial metadata for clinical_agent
# (massive_trans, responder, txa may need manual metadata)
TRIAL_COHORTS=(
    "crash_2"
    "ist3"
    "sprint"
    "accord"
)

echo "Starting XLearner experiments with ${NUM_TRIALS} trials for each cohort"
echo "Running cohorts in PARALLEL across ${NUM_GPUS} GPUs: ${GPUS[@]}"
echo "=========================================================================="

# Array to store background process PIDs
declare -a PIDS=()
declare -a COHORT_NAMES=()

# Function to run complete pipeline for one cohort
run_cohort_pipeline() {
    local cohort=$1
    local gpu=$2
    local device="cuda:${gpu}"
    
    echo "[GPU ${gpu}] Starting pipeline for ${cohort}"
    
    # # Step 1: SHAP analysis
    # echo "[GPU ${gpu}] [${cohort}] Step 1/3: Computing SHAP values..."
    # python single_cohort_analysis.py \
    #     --num_trials ${NUM_TRIALS} \
    #     --cohort_name ${cohort} \
    #     --device ${device} \
    #     --wandb \
    #     --relative_change_threshold ${RELATIVE_CHANGE_THRESHOLD} \
    #     --top_n_features ${TOP_N_FEATURES} \
    #     > logs/${cohort}_shap.log 2>&1
    
    # local shap_exit=$?
    # if [ $shap_exit -ne 0 ]; then
    #     echo "[GPU ${gpu}] [${cohort}] ✗ SHAP analysis failed (exit code: ${shap_exit})"
    #     return $shap_exit
    # fi
    # echo "[GPU ${gpu}] [${cohort}] ✓ SHAP analysis completed"
    
    # Check if this cohort has trial metadata support
    if [[ " ${TRIAL_COHORTS[@]} " =~ " ${cohort} " ]]; then
        
        SHAP_JSON="results/${cohort}/shapley/${cohort}_shap_summary_False.json"
        OUT_WITH_SHAP="results/agent/${cohort}/hypotheses_with_shap_XLearner.json"
        OUT_WITHOUT_SHAP="results/agent/${cohort}/hypotheses_without_shap_baseline.json"
        EMPTY_SHAP_JSON="results/agent/${cohort}/shapley/${cohort}_empty_shap.json"
        
        if [ -f "${SHAP_JSON}" ]; then
            # Step 2: Clinical agent WITH SHAP
            echo "[GPU ${gpu}] [${cohort}] Step 2/3: Generating hypotheses WITH SHAP..."
            python clinical_agent.py \
                --shap_json ${SHAP_JSON} \
                --out_json ${OUT_WITH_SHAP} \
                --trial_name ${cohort} \
                --n_features ${TOP_N_FEATURES} \
                --n_hypotheses ${N_HYPOTHESES} \
                --model ${MODEL} \
                --enable_verifier \
                --enable_judge \
                > logs/${cohort}_agent_with_shap.log 2>&1
            
            if [ $? -eq 0 ]; then
                echo "[GPU ${gpu}] [${cohort}] ✓ Hypotheses WITH SHAP generated"
            else
                echo "[GPU ${gpu}] [${cohort}] ✗ Error generating hypotheses WITH SHAP"
            fi
            
            # Step 3: Clinical agent WITHOUT SHAP
            echo "[GPU ${gpu}] [${cohort}] Step 3/3: Generating hypotheses WITHOUT SHAP..."
            mkdir -p $(dirname ${EMPTY_SHAP_JSON})
            echo '{"metadata": {"cohort": "'${cohort}'", "model": "baseline_no_shap"}, "features": []}' > ${EMPTY_SHAP_JSON}
            
            python clinical_agent.py \
                --shap_json ${EMPTY_SHAP_JSON} \
                --out_json ${OUT_WITHOUT_SHAP} \
                --trial_name ${cohort} \
                --n_features ${TOP_N_FEATURES} \
                --n_hypotheses ${N_HYPOTHESES} \
                --model ${MODEL} \
                --enable_verifier \
                --enable_judge \
                > logs/${cohort}_agent_without_shap.log 2>&1
            
            if [ $? -eq 0 ]; then
                echo "[GPU ${gpu}] [${cohort}] ✓ Hypotheses WITHOUT SHAP generated"
            else
                echo "[GPU ${gpu}] [${cohort}] ✗ Error generating hypotheses WITHOUT SHAP"
            fi
        else
            echo "[GPU ${gpu}] [${cohort}] ⚠ SHAP JSON not found: ${SHAP_JSON}"
        fi
    else
        echo "[GPU ${gpu}] [${cohort}] ⚠ Skipping clinical_agent (no trial metadata)"
    fi
    
    echo "[GPU ${gpu}] [${cohort}] ✓ Complete pipeline finished"
    return 0
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Launch complete pipeline for each cohort in parallel
echo ""
echo "Launching complete pipelines in parallel across ${NUM_GPUS} GPUs..."
echo "Each cohort will: Train Model → Clinical Agent WITH SHAP → Clinical Agent WITHOUT SHAP"
echo "=================================================="

gpu_idx=0
for cohort in "${COHORTS[@]}"
do
    gpu=${GPUS[$gpu_idx]}
    
    # Run complete pipeline in background
    run_cohort_pipeline ${cohort} ${gpu} &
    PIDS+=($!)
    COHORT_NAMES+=("${cohort}")
    
    echo "Launched ${cohort} pipeline on GPU ${gpu} (PID: ${PIDS[-1]})"
    
    # Cycle through available GPUs
    gpu_idx=$(( (gpu_idx + 1) % NUM_GPUS ))
    
    # Small delay to avoid overwhelming the system
    sleep 2
done

echo ""
echo "All pipelines launched. Waiting for completion..."
echo "=================================================="

# Wait for all pipelines to complete
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    cohort=${COHORT_NAMES[$i]}
    
    wait $pid
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ ${cohort} complete pipeline finished successfully"
    else
        echo "✗ ${cohort} complete pipeline failed (exit code: ${exit_code})"
    fi
done

echo ""
echo "=========================================================================="
echo "All experiments completed!"
echo ""
echo "Results structure:"
echo "  - SHAP values: results/<cohort>/shapley/<cohort>_shap_summary_True.json"
echo "  - Hypotheses WITH SHAP: results/<cohort>/hypotheses_with_shap_XLearner.json"
echo "  - Hypotheses WITHOUT SHAP: results/<cohort>/hypotheses_without_shap_baseline.json"
echo "  - Logs: logs/<cohort>_*.log"
echo "=========================================================================="
