"""pubmed_validation.py
======================
Mechanism validator that uses the parent PubMedMechanismValidator's
keyword-based PubMed query pipeline for validation, while also exposing
Semantic Scholar citation-graph utilities (fetch_trial_citation_graph,
_filter_citations_locally) for optional pre-filtering or future use.

Validation uses the parent's 3-tier PubMed query:
  Tier 1: treatment context + feature + mechanism keywords + interaction terms
  Tier 2: treatment context + feature + interaction terms
  Tier 3: treatment context + feature (broad fallback)

Usage:
    python ALEX/pubmed_validation.py \\
        --input ALEX/results/accord/gpt-5-mini/hypogenic/seed_0/hypotheses.json \\
        --dataset accord \\
        --model gpt-5-mini
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from src.agent_utils import load_local_env, write_json_file
from pubmed_mechanism_validator import PubMedMechanismValidator


class SemanticScholarMechanismValidator(PubMedMechanismValidator):
    """Extends PubMedMechanismValidator with Semantic Scholar citation-graph
    utilities.

    Validation (validate_mechanism, validate_all_mechanisms, generate_report)
    is fully inherited from PubMedMechanismValidator and uses the standard
    3-tier PubMed keyword query pipeline.

    The Semantic Scholar methods (fetch_trial_citation_graph,
    _filter_citations_locally) are available for ad-hoc analysis or future
    hybrid validation strategies but are not called during the standard
    validation flow.
    """

    # Anchor PMIDs for primary trial publications
    ANCHOR_PMIDS: Dict[str, str] = {
        "accord":  "18539917",   # Effects of intensive glucose lowering in type 2 diabetes
        "accord_glycemia": "18539917",  # Same primary ACCORD trial (glycemia arm)
        "sprint":  "26551272",   # Randomized Trial of Intensive vs Standard Blood-Pressure Control
        "ist3":    "22632908",   # Benefits and harms of intravenous thrombolysis
        "crash_2": "20554319",   # Effects of tranexamic acid on death / vascular occlusive events
    }

    def __init__(self, s2_api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.s2_api_key = s2_api_key
        # Keyed by anchor PMID so each trial graph is fetched only once per run
        self._citation_graph_cache: Dict[str, List[Dict[str, str]]] = {}

    # ------------------------------------------------------------------
    # Citation-graph retrieval
    # ------------------------------------------------------------------

    def fetch_trial_citation_graph(self, dataset: str) -> List[Dict[str, str]]:
        """Fetch every paper that has cited the primary trial via Semantic Scholar.

        Results are cached in-memory for the lifetime of the validator object
        so that subsequent validate_mechanism calls for the same dataset do not
        repeat the network round-trip.
        """
        dataset = dataset.lower().replace("-", "_")
        anchor_pmid = self.ANCHOR_PMIDS.get(dataset)
        if not anchor_pmid:
            print(
                f"  Warning: no anchor PMID for dataset '{dataset}'. "
                "Falling back to PubMed search."
            )
            return []

        if anchor_pmid in self._citation_graph_cache:
            return self._citation_graph_cache[anchor_pmid]

        print(
            f"Building citation graph for {dataset.upper()} "
            f"(anchor PMID: {anchor_pmid}) …"
        )

        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/"
            f"PMID:{anchor_pmid}/citations"
        )
        headers: Dict[str, str] = (
            {"x-api-key": self.s2_api_key} if self.s2_api_key else {}
        )
        params: Dict[str, Any] = {
            "fields": "title,abstract,externalIds,year",
            "limit": 1000,
        }

        all_papers: List[Dict[str, str]] = []
        offset = 0

        while True:
            params["offset"] = offset
            try:
                resp = requests.get(
                    url, params=params, headers=headers, timeout=15
                )
                if resp.status_code == 429:
                    print("  S2 rate-limit hit — waiting 5 s …")
                    time.sleep(5)
                    continue
                if resp.status_code != 200:
                    print(f"  S2 API error {resp.status_code}: {resp.text[:200]}")
                    break

                data = resp.json()
                citations = data.get("data", [])
                if not citations:
                    break

                for cite in citations:
                    paper = cite.get("citingPaper", {})
                    if paper.get("abstract") and paper.get("title"):
                        all_papers.append(
                            {
                                "pmid": paper.get("externalIds", {}).get(
                                    "PubMed", "No-PMID"
                                ),
                                "title": paper["title"],
                                "abstract": paper["abstract"],
                                "year": str(paper.get("year", "Unknown")),
                            }
                        )

                print(f"  … {len(all_papers)} citing papers so far")

                if "next" not in data:
                    break
                offset += 1000
                time.sleep(1)  # polite pause

            except Exception as exc:
                print(f"  Error fetching citation graph: {exc}")
                break

        print(
            f"  Citation graph complete: {len(all_papers)} papers cite "
            f"the {dataset.upper()} trial."
        )
        self._citation_graph_cache[anchor_pmid] = all_papers
        return all_papers

    # ------------------------------------------------------------------
    # Local filtering (replaces PubMed Boolean search)
    # ------------------------------------------------------------------

    def _get_dataset_feature_map(self, dataset: str) -> Dict[str, str]:
        """Return the feature→query-string map for a dataset.

        Kept in sync with the feature dicts inside construct_search_query so
        that _find_feature_query (inherited from parent) can resolve
        feature-name synonyms correctly.
        """
        _feature_maps: Dict[str, Dict[str, str]] = {
            "accord": {
                "hba1c":           'HbA1c OR "glycated hemoglobin" OR "glycemic control"',
                "sbp":             '"systolic blood pressure" OR hypertension',
                "dbp":             '"diastolic blood pressure" OR hypertension',
                "age":             "age OR elderly OR geriatric",
                "baseline_age":    "age OR elderly OR geriatric",
                "bmi":             'BMI OR obesity OR "body mass index"',
                "gfr":             'GFR OR eGFR OR "renal function"',
                "screat":          'creatinine OR "serum creatinine"',
                "uacr":            'UACR OR albuminuria OR "albumin creatinine ratio"',
                "umalcr":          'UACR OR albuminuria OR "albumin creatinine ratio"',
                "chol":            'cholesterol OR "total cholesterol"',
                "trig":            "triglyceride OR triglycerides",
                "trr":             "triglyceride OR triglycerides",
                "vldl":            "VLDL OR lipoprotein",
                "ldl":             'LDL OR "low density lipoprotein"',
                "hdl":             'HDL OR "high density lipoprotein"',
                "glur":            '"fasting plasma glucose" OR FPG OR glucose OR glycemia OR hyperglycemia',
                "hr":              '"heart rate" OR pulse OR tachycardia',
                "female":          'female OR sex OR gender',
                "race_black":      '"Continental Population Groups"[Mesh] OR "Black"[tiab] OR "White"[tiab] OR race[tiab]',
                "smoke_3cat":      "smoking OR smoker OR tobacco",
                "aspirin":         "aspirin OR antiplatelet",
                "statin":          "statin OR lipid-lowering",
                "cvd_hx_baseline": (
                    '"history of cardiovascular disease" OR '
                    '"prior cardiovascular disease" OR "prior MI"'
                ),
                "sub_cvd":         '"history of cardiovascular disease" OR "prior cardiovascular disease" OR "prior MI" OR "prior stroke"',
            },
            "sprint": {
                "age":    "age OR elderly OR geriatric",
                "sbp":    '"systolic blood pressure" OR "blood pressure"',
                "dbp":    '"diastolic blood pressure" OR "blood pressure"',
                "egfr":   'eGFR OR GFR OR "renal function"',
                "screat": 'creatinine OR "serum creatinine"',
                "bmi":    'BMI OR obesity OR "body mass index"',
                "sub_cvd": '"cardiovascular disease" OR CVD',
                "sub_ckd": '"chronic kidney disease" OR CKD',
            },
            "ist3": {
                "age":              "age OR elderly OR geriatric",
                "nihss":            'NIHSS OR "stroke severity"',
                "sbprand":          '"systolic blood pressure" OR hypertension',
                "dbprand":          '"diastolic blood pressure" OR hypertension',
                "glucose":          'glucose OR hyperglycemia OR "blood glucose"',
                "time_to_treatment":(
                    '"time to treatment" OR "onset to treatment" OR '
                    '"treatment delay"'
                ),
            },
            "crash_2": {
                "iage":        "age OR elderly OR geriatric",
                "isbp":        '"systolic blood pressure" OR hypotension',
                "ninjurytime": (
                    '"time from injury" OR "injury-to-treatment time" OR '
                    '"treatment delay"'
                ),
                "igcs":        'GCS OR "Glasgow Coma Scale"',
                "ihr":         '"heart rate" OR pulse OR tachycardia',
                "iinjurytype": (
                    '"injury type" OR "penetrating injury" OR "blunt trauma"'
                ),
            },
            "accord_glycemia": {
                "baseline_age":    "age OR elderly OR geriatric",
                "bmi":             'BMI OR obesity OR "body mass index"',
                "hba1c":           'HbA1c OR "glycated hemoglobin" OR "glycemic control"',
                "yrsdiab":         '"diabetes duration" OR "years of diabetes"',
                "sbp":             '"systolic blood pressure" OR hypertension',
                "dbp":             '"diastolic blood pressure" OR hypertension',
                "hr":              '"heart rate" OR pulse OR tachycardia',
                "fpg":             '"fasting plasma glucose" OR FPG OR glucose',
                "gfr":             'GFR OR eGFR OR "renal function"',
                "uacr":            'UACR OR albuminuria OR "albumin creatinine ratio"',
                "trig":            "triglyceride OR triglycerides",
                "ldl":             'LDL OR "low density lipoprotein"',
                "hdl":             'HDL OR "high density lipoprotein"',
                "insulin":         'insulin OR "insulin therapy"',
                "dm_med":          '"diabetes medication" OR "oral hypoglycemic"',
                "cvd_hx_baseline": (
                    '"history of cardiovascular disease" OR '
                    '"prior cardiovascular disease" OR "prior MI"'
                ),
            },
        }
        return _feature_maps.get(dataset.lower().replace("-", "_"), {})

    def _filter_citations_locally(
        self,
        citing_papers: List[Dict[str, str]],
        feature_name: str,
        mechanism: Dict[str, Any],
        dataset: str,
        require_interaction: bool,
    ) -> List[Dict[str, str]]:
        """Filter the citation graph locally using feature and mechanism keywords.

        Mirrors the tiered logic of construct_search_query / search_pubmed but
        without any network calls.
        """
        # Use the same feature→query map as the parent's construct_search_query
        dataset_features = self._get_dataset_feature_map(dataset)
        raw_feature_query = self._find_feature_query(feature_name, dataset_features)

        if raw_feature_query:
            # Strip PubMed field tags and quotes for plain-text matching
            feature_keywords = [
                kw.strip().strip('"').lower().replace("[tiab]", "").strip()
                for kw in raw_feature_query.split(" OR ")
                if kw.strip()
            ]
        else:
            feature_keywords = [self._normalize_text(feature_name)]

        # Mechanism biological keywords via the parent helper
        mech_raw = self._extract_mechanism_keywords(
            mechanism.get("description", "")
        )
        mech_keywords = [
            k.strip().strip('"').lower()
            for k in mech_raw.split(" OR ")
            if k.strip()
        ]

        interaction_terms = [
            "interaction", "effect modification", "heterogeneity", "subgroup",
            "differential", "predictive factor", "hte", "effect modifier",
            "treatment heterogeneity",
        ]

        filtered: List[Dict[str, str]] = []
        for paper in citing_papers:
            text = (
                paper.get("title", "") + " " + paper.get("abstract", "")
            ).lower()

            # Condition A: at least one feature keyword present
            if not any(kw in text for kw in feature_keywords):
                continue

            # Condition B: interaction terms (strict tier only)
            if require_interaction and not any(
                t in text for t in interaction_terms
            ):
                continue

            filtered.append(paper)

        return filtered

    # ------------------------------------------------------------------
    # LLM evaluation: classification + CEBM evidence level
    # ------------------------------------------------------------------

    def analyze_abstract_with_llm(
        self,
        abstract: Dict[str, str],
        mechanism: Dict[str, Any],
        feature_name: str,
        dataset: str,
    ) -> Dict[str, Any]:
        """Extend the parent LLM evaluation with an Oxford CEBM evidence level.

        Calls the parent to get classification / stance / reasoning, then makes
        a second lightweight LLM call to assign a CEBM level (1a–5) for the
        abstract.  The two results are merged and returned together.
        """
        # Task 1 — inherited classification + stance
        result = super().analyze_abstract_with_llm(
            abstract, mechanism, feature_name, dataset
        )

        # Task 2 — CEBM evidence level
        if not self.openai_client:
            result["cebm_level"] = "unassigned"
            result["cebm_reasoning"] = "No LLM available."
            return result

        title    = abstract.get("title", "")
        text     = abstract.get("abstract", "")
        year     = abstract.get("year", "unknown")

        cebm_prompt = f"""You are an expert in evidence-based medicine.

Assign the Oxford Centre for Evidence-Based Medicine (CEBM) Level of Evidence
(Howick et al., 2011) to the following paper based only on its title and abstract.

Levels:
  1a  Systematic review of RCTs
  1b  Individual RCT (with narrow confidence interval)
  1c  All-or-none study
  2a  Systematic review of cohort studies
  2b  Individual cohort study or low-quality RCT
  2c  Outcomes research
  3a  Systematic review of case-control studies
  3b  Individual case-control study
  4   Case series, poor-quality cohort or case-control
  5   Expert opinion, narrative review, bench research

Paper (year: {year}):
Title: {title}
Abstract: {text}

Return a valid JSON object with exactly two fields:
{{
  "cebm_level": "<one of: 1a 1b 1c 2a 2b 2c 3a 3b 4 5>",
  "cebm_reasoning": "<one sentence explaining why>"
}}"""

        try:
            request_kwargs = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an evidence-based medicine expert. "
                            "Output valid JSON only."
                        ),
                    },
                    {"role": "user", "content": cebm_prompt},
                ],
                "response_format": {"type": "json_object"},
            }
            model_lc = str(self.model).lower()
            if not model_lc.startswith("gpt-5"):
                request_kwargs["temperature"] = 0.0

            resp = self.openai_client.chat.completions.create(**request_kwargs)
            content = resp.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()

            cebm = json.loads(content)
            valid_levels = {
                "1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "4", "5",
            }
            result["cebm_level"] = (
                cebm.get("cebm_level", "5")
                if cebm.get("cebm_level", "") in valid_levels
                else "5"
            )
            result["cebm_reasoning"] = cebm.get(
                "cebm_reasoning", "No reasoning provided."
            )

            if self.llm_delay > 0:
                time.sleep(self.llm_delay)

        except Exception as exc:
            print(f"  CEBM assessment error: {exc}")
            result["cebm_level"] = "unassigned"
            result["cebm_reasoning"] = str(exc)

        return result


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic Scholar citation-graph mechanism validator"
    )
    parser.add_argument("--input",         required=True, help="Path to hypotheses JSON")
    parser.add_argument("--output",        default=None,  help="Output path for validation results")
    parser.add_argument("--dataset",       default="ist3")
    parser.add_argument("--email",         default="research@example.com")
    parser.add_argument("--api-key",       default=None,  help="OpenAI API key")
    parser.add_argument("--s2-api-key",    default=None,  help="Semantic Scholar API key")
    parser.add_argument("--max-abstracts", type=int, default=30)
    parser.add_argument("--model",         default="gpt-5-mini")
    parser.add_argument(
        "--api-provider", default="openai", choices=["openai", "openrouter"]
    )
    parser.add_argument("--api-base-url",  default=None)
    parser.add_argument(
        "--llm-delay", type=float, default=0.5,
        help="Seconds to sleep between LLM calls (0 = max speed)",
    )
    args = parser.parse_args()

    load_local_env()

    api_key = (
        args.api_key
        if args.api_provider == "openrouter"
        else (args.api_key or os.environ.get("OPENAI_API_KEY"))
    )

    validator = SemanticScholarMechanismValidator(
        s2_api_key=args.s2_api_key,
        email=args.email,
        api_key=api_key,
        max_abstracts=args.max_abstracts,
        model=args.model,
        api_provider=args.api_provider,
        api_base_url=args.api_base_url,
        llm_delay=args.llm_delay,
    )

    results = validator.validate_all_mechanisms(args.input, use_llm=True)

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.input)),
        "hypotheses_s2_validation.json",
    )
    validator.generate_report(results, output_path)


if __name__ == "__main__":
    main()
