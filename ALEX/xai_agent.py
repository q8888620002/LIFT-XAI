"""
XAI Agent — an LLM agent that uses tools to analyse a clinical trial dataset.

The LLM drives the workflow by calling tools in any order it chooses:
  - load_dataset          → inspect cohort size, features, treatment balance
  - run_shap_pipeline     → train CATE ensemble + compute Shapley values
  - load_shap_summary     → read an existing SHAP JSON (skip recomputation)
  - generate_hypotheses   → generate mechanistic hypotheses from SHAP features

Run from the project root:
    python ALEX/xai_agent.py \\
        --goal "Analyse crash_2 with DRLearner and generate mechanistic hypotheses" \\
        --cohort_name crash_2 \\
        --trial_name crash_2 \\
        --out_json docs/agent/crash_2/gpt-5-mini/seed_0/hypotheses.json \\
        --model gpt-5-mini
"""

import argparse
import json
import os
import pickle
import sys
from typing import Any

import numpy as np
import torch
from captum.attr import ShapleyValueSampling

import src.CATENets.catenets.models.torch.pseudo_outcome_nets as pseudo_outcome_nets
from src.dataset import Dataset
from src.agent_utils import (
    ensure_out_dir,
    get_model_client,
    get_trial_metadata,
    load_local_env,
    load_top_features,
    resolve_seeded_output_path,
    write_json_file,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # expose ALEX/

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _to_py(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _translate_feature_name(name: str) -> str:
    mappings = {
        "stroketype_1.0": "Stroke Type: TACI",
        "stroketype_2.0": "Stroke Type: PACI",
        "stroketype_3.0": "Stroke Type: LACI",
        "stroketype_4.0": "Stroke Type: POCI",
        "stroketype_5.0": "Stroke Type: Other",
        "infarct_0": "Infarct Visible on CT: No",
        "infarct_1.0": "Infarct Visible on CT: Possibly Yes",
        "infarct_2.0": "Infarct Visible on CT: Definitely Yes",
        "iinjurytype_1": "Injury Type: Blunt",
        "iinjurytype_2": "Injury Type: Penetrating",
    }
    return mappings.get(name, name)


def tool_load_dataset(cohort_name: str) -> dict:
    """Load a cohort dataset and return summary statistics."""
    dataset = Dataset(cohort_name, 0)
    x_train, w_train, y_train = dataset.get_data("train")
    x_test, _, _ = dataset.get_data("test")
    feature_names = dataset.get_feature_names().tolist()
    return {
        "cohort_name": cohort_name,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_features": int(x_train.shape[1]),
        "feature_names": feature_names,
        "treatment_rate_train": float(w_train.mean()),
        "outcome_rate_train": float(y_train.mean()),
        "n_categorical_groups": len(dataset.categorical_indices),
        "categorical_groups": list(dataset.categorical_indices.keys()),
    }


def tool_run_shap_pipeline(
    cohort_name: str,
    learner: str,
    num_trials: int,
    device: str,
    baseline_mode: str,
    relative_change_threshold: float,
    top_n_features: int,
) -> dict:
    """Train a CATE ensemble and compute Shapley values. Returns the path to
    the saved JSON summary and a brief digest of the top features found."""
    use_random_baseline = baseline_mode == "random"

    dataset = Dataset(cohort_name, 0)
    x_train, w_train, y_train = dataset.get_data()

    cohort_predict_results, cohort_shap_values = [], []
    baseline_indices, baseline_outputs, shap_sum_pred_corr = [], [], []

    for i in range(num_trials):
        sampled_idx = np.random.choice(len(x_train), size=int(0.9 * len(x_train)), replace=False)
        x_s, y_s, w_s = x_train[sampled_idx], y_train[sampled_idx], w_train[sampled_idx]

        learner_class = getattr(pseudo_outcome_nets, learner)
        model = learner_class(
            x_s.shape[1],
            binary_y=(len(np.unique(y_s)) == 2),
            n_layers_out=2, n_units_out=100,
            batch_size=128, n_iter=1000,
            nonlin="relu", device=device, seed=i,
        )
        model.fit(x_s, y_s, w_s)
        cohort_predict_results.append(
            model.predict(X=x_train).detach().cpu().numpy().flatten()
        )

        if not use_random_baseline:
            baseline = np.median(x_s, 0)
            baseline_index = None
            for _, idx_lst in dataset.discrete_indices.items():
                baseline[idx_lst] = 0.5 if len(idx_lst) == 1 else 1 / len(idx_lst)
        else:
            baseline_index = np.random.choice(len(x_train), 1)
            baseline = x_train[baseline_index]

        baseline_indices.append(int(baseline_index[0]) if baseline_index is not None else None)
        baseline_outputs.append(
            float(model.predict(X=baseline.reshape(1, -1)).detach().cpu().numpy().flatten()[0])
        )

        print(f"Trial {i+1}/{num_trials} — computing SHAP")
        sv_model = ShapleyValueSampling(model)
        shap_values = (
            sv_model.attribute(
                torch.tensor(x_train).to(device),
                n_samples=1000,
                baselines=torch.tensor(baseline.reshape(1, -1)).to(device),
                perturbations_per_eval=10,
                show_progress=True,
            )
            .detach().cpu().numpy()
        )
        cohort_shap_values.append(shap_values)

        corr = np.corrcoef(shap_values.sum(axis=1), cohort_predict_results[-1])[0, 1]
        shap_sum_pred_corr.append(float(corr) if not np.isnan(corr) else 0.0)

        arr = np.array(cohort_shap_values)
        mean_sv = arr.mean(axis=0)
        if i > 5:
            prev_mean = np.array(cohort_shap_values[:-1]).mean(axis=0)
            rel_change = np.mean(np.abs(mean_sv - prev_mean) / (np.abs(prev_mean) + 1e-8))
            print(f"  avg relative change = {rel_change:.6f}")
            if rel_change < relative_change_threshold:
                print(f"  Converged at trial {i+1}. Stopping early.")
                break

    # --- persist artefacts ---
    save_path = f"results/{cohort_name}/shapley"
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{cohort_name}_predict_results_{use_random_baseline}.pkl"), "wb") as f:
        pickle.dump(np.stack(cohort_predict_results), f)
    with open(os.path.join(save_path, f"{cohort_name}_shap_bootstrapped_{use_random_baseline}.pkl"), "wb") as f:
        pickle.dump(np.stack(cohort_shap_values), f)

    # --- build JSON summary ---
    sv_arr = np.stack(cohort_shap_values)
    pred_arr = np.stack(cohort_predict_results)
    feature_names = dataset.get_feature_names()

    abs_mean_per_trial = np.abs(sv_arr).mean(axis=1)
    abs_mean = abs_mean_per_trial.mean(axis=0)
    abs_std = abs_mean_per_trial.std(axis=0)
    mean_per_trial = sv_arr.mean(axis=1)
    mean_val = mean_per_trial.mean(axis=0)
    mean_std = mean_per_trial.std(axis=0)

    feature_records = sorted([
        {
            "feature_index": int(j),
            "feature": _translate_feature_name(str(fn)),
            "feature_original": str(fn),
            "shap_mean_abs": float(abs_mean[j]),
            "shap_mean_abs_std": float(abs_std[j]),
            "shap_mean": float(mean_val[j]),
            "shap_mean_std": float(mean_std[j]),
        }
        for j, fn in enumerate(feature_names)
    ], key=lambda x: x["shap_mean_abs"], reverse=True)

    cat_indices_set = set()
    categorical_aggregates = []
    for cat_name, cat_idx in dataset.categorical_indices.items():
        cat_abs = abs_mean_per_trial[:, cat_idx].sum(axis=1)
        cat_signed = mean_per_trial[:, cat_idx].sum(axis=1)
        categorical_aggregates.append({
            "feature": cat_name, "feature_original": cat_name,
            "is_categorical": True, "num_categories": len(cat_idx),
            "category_names": [str(feature_names[k]) for k in cat_idx],
            "category_indices": cat_idx,
            "shap_mean_abs": float(cat_abs.mean()), "shap_mean_abs_std": float(cat_abs.std()),
            "shap_mean": float(cat_signed.mean()), "shap_mean_std": float(cat_signed.std()),
        })
        cat_indices_set.update(cat_idx)

    non_cat = [r for r in feature_records if r["feature_index"] not in cat_indices_set]
    combined = sorted(non_cat + categorical_aggregates, key=lambda x: x["shap_mean_abs"], reverse=True)

    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    pred_pooled = pred_arr.flatten()
    summary_json = {
        "metadata": {
            "dataset": cohort_name, "model": learner,
            "trials_completed": len(cohort_shap_values),
            "total_trials_requested": num_trials,
            "baseline_mode": "random_sample" if use_random_baseline else "median",
            "relative_change_threshold": relative_change_threshold,
            "device": device,
            "baseline_indices": baseline_indices if use_random_baseline else None,
            "baseline_outputs": baseline_outputs,
        },
        "summary": {
            "num_features": int(sv_arr.shape[2]),
            "num_categorical_features": len(categorical_aggregates),
            "num_non_categorical_features": len(non_cat),
            "top_n_features": top_n_features,
            "top_features_by_mean_abs": combined[:top_n_features],
            "cate_prediction_overall": {
                "mean": float(pred_pooled.mean()), "std": float(pred_pooled.std()),
                "quantiles": {str(q): float(np.quantile(pred_pooled, q)) for q in qs},
                "positive_rate": float((pred_pooled > 0).mean()),
                "negative_rate": float((pred_pooled < 0).mean()),
            },
            "shap_sum_vs_cate_pred_corr": {
                "per_trial": _to_py(np.array(shap_sum_pred_corr)),
                "mean": float(np.nanmean(shap_sum_pred_corr)),
                "std": float(np.nanstd(shap_sum_pred_corr)),
            },
        },
        "features": combined,
        "features_all_original": feature_records,
        "per_trial": {
            "shap_abs_mean_per_trial": _to_py(abs_mean_per_trial),
            "shap_mean_per_trial": _to_py(mean_per_trial),
            "cate_prediction": {
                "mean_per_trial": _to_py(pred_arr.mean(axis=1)),
                "std_per_trial": _to_py(pred_arr.std(axis=1)),
                "quantiles_per_trial": {
                    "quantiles": [str(q) for q in qs],
                    "values": _to_py(np.quantile(pred_arr, qs, axis=1).T),
                },
                "pos_rate_per_trial": _to_py((pred_arr > 0).mean(axis=1)),
                "neg_rate_per_trial": _to_py((pred_arr < 0).mean(axis=1)),
            },
        },
    }

    json_path = os.path.join(save_path, f"{cohort_name}_shap_summary_{use_random_baseline}.json")
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)

    top_digest = [
        {"feature": r["feature"], "shap_mean_abs": r["shap_mean_abs"]}
        for r in combined[:top_n_features]
    ]
    return {
        "status": "success",
        "json_path": json_path,
        "trials_completed": len(cohort_shap_values),
        "top_features": top_digest,
        "shap_corr_mean": summary_json["summary"]["shap_sum_vs_cate_pred_corr"]["mean"],
    }


def tool_load_shap_summary(json_path: str, n_features: int = 15) -> dict:
    """Read an existing SHAP summary JSON and return metadata + top features."""
    if not os.path.exists(json_path):
        return {"error": f"File not found: {json_path}"}
    with open(json_path) as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    features = data.get("features", [])[:n_features]
    top_digest = [
        {"feature": r["feature"], "shap_mean_abs": r["shap_mean_abs"]}
        for r in features
    ]
    return {
        "status": "loaded",
        "json_path": json_path,
        "dataset": meta.get("dataset"),
        "model": meta.get("model"),
        "trials_completed": meta.get("trials_completed"),
        "baseline_mode": meta.get("baseline_mode"),
        "top_features": top_digest,
    }


def tool_generate_hypotheses(
    shap_json_path: str,
    trial_name: str,
    out_json: str,
    n_features: int,
    n_hypotheses: int,
    model_name: str,
    api_provider: str,
    seed: int,
    api_base_url: str = None,
) -> dict:
    """Generate mechanistic hypotheses from a SHAP summary using the clinical agent."""
    from clinical_agent import generate_feature_hypotheses

    client = get_model_client(api_provider, api_base_url)
    trial_meta = get_trial_metadata(trial_name)
    evidence = load_top_features(shap_json_path, n_features, dataset_override=trial_name.lower())

    study_context = {
        "dataset": evidence["dataset"],
        "learner": evidence["learner"],
        "population": trial_meta["population"],
        "treatment": trial_meta["treatment"],
        "outcome": trial_meta["outcome"],
        "source_explainer": evidence["explainer"],
        "n_hypotheses_per_feature": n_hypotheses,
        "n_features": n_features,
        "available_features": evidence.get("available_features", []),
        "use_data_summary": False,
    }

    feature_hypotheses = generate_feature_hypotheses(
        top_features=evidence["top_feature_evidence"],
        study_context=study_context,
        client=client,
        model_name=model_name,
    )

    if not feature_hypotheses:
        return {"error": "Hypothesis generation failed."}

    seeded_path = resolve_seeded_output_path(out_json, seed)
    ensure_out_dir(seeded_path)
    write_json_file(seeded_path, feature_hypotheses.model_dump())

    return {
        "status": "success",
        "out_json": seeded_path,
        "n_features_covered": len(feature_hypotheses.feature_hypotheses),
    }


# ---------------------------------------------------------------------------
# Tool registry: schema + dispatch
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_dataset",
            "description": (
                "Load a clinical trial dataset and return summary statistics: "
                "sample sizes, feature count, treatment and outcome rates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cohort_name": {
                        "type": "string",
                        "description": "Dataset name (e.g. crash_2, ist3, sprint, accord, txa).",
                    }
                },
                "required": ["cohort_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shap_pipeline",
            "description": (
                "Train a bootstrapped ensemble of CATE models and compute ensemble "
                "Shapley values. Saves a SHAP summary JSON to "
                "results/<cohort_name>/shapley/ and returns the top features found."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cohort_name": {"type": "string"},
                    "learner": {
                        "type": "string",
                        "enum": ["XLearner", "RLearner", "DRLearner", "PWLearner", "RALearner", "ULearner"],
                        "description": "CATE learner class.",
                    },
                    "num_trials": {
                        "type": "integer",
                        "description": "Number of bootstrap trials (recommended: 10–30).",
                    },
                    "device": {
                        "type": "string",
                        "description": "CUDA device string (e.g. cuda:0).",
                    },
                    "baseline_mode": {
                        "type": "string",
                        "enum": ["median", "random"],
                        "description": "SHAP baseline: 'median' (feature medians) or 'random' (a random training sample).",
                    },
                    "relative_change_threshold": {
                        "type": "number",
                        "description": "Early-stopping threshold on mean relative SHAP change (default 0.05).",
                    },
                    "top_n_features": {
                        "type": "integer",
                        "description": "How many top features to include in the summary (default 15).",
                    },
                },
                "required": ["cohort_name", "learner", "num_trials", "device"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_shap_summary",
            "description": (
                "Read an existing SHAP summary JSON (skipping recomputation) "
                "and return the top features and metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "json_path": {
                        "type": "string",
                        "description": "Path to the SHAP summary JSON file.",
                    },
                    "n_features": {
                        "type": "integer",
                        "description": "How many top features to return (default 15).",
                    },
                },
                "required": ["json_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_hypotheses",
            "description": (
                "Generate mechanistic hypotheses explaining why the top SHAP features "
                "modify the treatment effect, using the clinical agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "shap_json_path": {
                        "type": "string",
                        "description": "Path to the SHAP summary JSON.",
                    },
                    "trial_name": {
                        "type": "string",
                        "description": "Trial name for metadata lookup (e.g. crash_2, ist3).",
                    },
                    "out_json": {
                        "type": "string",
                        "description": "Output path for the hypotheses JSON.",
                    },
                    "n_features": {
                        "type": "integer",
                        "description": "Number of top features to pass to the agent.",
                    },
                    "n_hypotheses": {
                        "type": "integer",
                        "description": "Number of hypotheses to generate per feature.",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "LLM model name (e.g. gpt-5-mini).",
                    },
                    "api_provider": {
                        "type": "string",
                        "enum": ["openai", "openrouter"],
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Seed index for output subfolder.",
                    },
                    "api_base_url": {
                        "type": "string",
                        "description": "Optional custom API base URL.",
                    },
                },
                "required": ["shap_json_path", "trial_name", "out_json", "n_features",
                             "n_hypotheses", "model_name", "api_provider", "seed"],
            },
        },
    },
]

TOOL_DISPATCH: dict[str, Any] = {
    "load_dataset": tool_load_dataset,
    "run_shap_pipeline": tool_run_shap_pipeline,
    "load_shap_summary": tool_load_shap_summary,
    "generate_hypotheses": tool_generate_hypotheses,
}


def execute_tool(name: str, arguments: dict) -> str:
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**arguments)
    except Exception as exc:
        result = {"error": str(exc)}
    return json.dumps(result, default=_to_py)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an XAI (Explainable AI) research agent for clinical trial data.

You have access to tools that let you:
  1. Inspect a dataset (load_dataset)
  2. Train a CATE model and compute ensemble Shapley values (run_shap_pipeline)
  3. Load an existing SHAP summary without recomputing (load_shap_summary)
  4. Generate mechanistic hypotheses explaining the SHAP results (generate_hypotheses)

Work step-by-step. Inspect the data first, then compute or load SHAP values, then
optionally generate hypotheses. When you are done, provide a concise written summary
of your findings: which features drive treatment heterogeneity and what that implies
clinically.
"""


def run_agent(goal: str, client, model_name: str, max_iterations: int = 20) -> str:
    """Run the agentic loop until the LLM stops calling tools."""
    messages = [{"role": "user", "content": goal}]

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # No tool calls → agent is done; return its final text
        if not message.tool_calls:
            return message.content or ""

        # Append assistant message (with tool_calls)
        messages.append(message)

        # Execute each tool call and feed results back
        for tc in message.tool_calls:
            print(f"\n[agent] calling tool: {tc.function.name}({tc.function.arguments[:120]}...)")
            arguments = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, arguments)
            print(f"[agent] tool result: {result[:200]}...")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "[Agent reached max iterations without a final answer]"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="XAI LLM agent: inspect data, compute SHAP, generate hypotheses."
    )
    parser.add_argument("--cohort_name", required=True, help="Dataset name (e.g. crash_2, ist3).")
    parser.add_argument("--trial_name", help="Trial name for hypothesis metadata (e.g. crash_2, ist3).")
    parser.add_argument("--learner", default="DRLearner",
                        choices=["XLearner", "RLearner", "DRLearner", "PWLearner", "RALearner", "ULearner"])
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out_json", help="Output JSON path for hypotheses.")
    parser.add_argument("--n_features", type=int, default=15)
    parser.add_argument("--n_hypotheses", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="gpt-5-mini", help="LLM model for the agent.")
    parser.add_argument("--api_provider", default="openai", choices=["openai", "openrouter"])
    parser.add_argument("--api_base_url", help="Custom API base URL.")
    parser.add_argument("--max_iterations", type=int, default=20)
    return parser.parse_args()


def _build_task(args) -> str:
    """Construct the agent's task description from CLI arguments."""
    lines = [
        f"Analyse the {args.cohort_name} clinical trial dataset.",
        f"Train a {args.learner} ensemble ({args.num_trials} trials) on device {args.device} "
        f"and compute ensemble Shapley values.",
    ]
    if args.trial_name:
        lines.append(
            f"Then generate {args.n_hypotheses} mechanistic hypotheses per feature "
            f"for the top {args.n_features} SHAP features (trial: {args.trial_name})."
        )
        if args.out_json:
            lines.append(f"Save hypotheses to: {args.out_json} (seed {args.seed}).")
        lines.append(f"Use model {args.model} via {args.api_provider}.")
    else:
        lines.append(
            f"Summarise the top {args.n_features} features driving treatment heterogeneity."
        )
    return " ".join(lines)


def main(args):
    load_local_env()
    client = get_model_client(args.api_provider, args.api_base_url)
    task = _build_task(args)

    print(f"[agent] task: {task}\n")
    final_answer = run_agent(task, client, model_name=args.model, max_iterations=args.max_iterations)

    print("\n" + "=" * 60)
    print("AGENT FINAL ANSWER")
    print("=" * 60)
    print(final_answer)


if __name__ == "__main__":
    args = parse_args()
    main(args)
