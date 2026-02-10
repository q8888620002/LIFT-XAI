"""
PubMed Mechanism Validator
===========================
This script extracts PubMed abstracts and analyzes whether they support or 
conflict with proposed mechanisms from hypothesis files.

Usage:
    python pubmed_mechanism_validator.py --input hypotheses_with_shap_XLearner.json --output validation_results.json
    python pubmed_mechanism_validator.py --cohort ist3 --max-abstracts 50
"""

import json
import argparse
import time
import os
from typing import List, Dict, Any
from collections import defaultdict
import re

try:
    from Bio import Entrez
except ImportError:
    print("BioPython not found. Install with: pip install biopython")
    Entrez = None

try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    print("OpenAI not found. Install with: pip install openai")
    OpenAI = None
    openai_available = False


class PubMedMechanismValidator:
    """Validates mechanisms against PubMed literature."""
    
    def __init__(self, email: str = "research@example.com", api_key: str = None, max_abstracts: int = 30):
        """
        Initialize the validator.
        
        Args:
            email: Email for PubMed API (required by NCBI)
            api_key: OpenAI API key for LLM-based analysis
            max_abstracts: Maximum number of abstracts to retrieve per mechanism
        """
        if Entrez:
            Entrez.email = email
        self.api_key = api_key
        self.max_abstracts = max_abstracts
        
        # Initialize OpenAI client if API key provided
        self.openai_client = None
        if api_key and openai_available:
            self.openai_client = OpenAI(api_key=api_key)
    
    def load_hypotheses(self, filepath: str) -> Dict[str, Any]:
        """Load hypotheses from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def construct_search_query(self, feature_name: str, mechanism: Dict[str, Any], dataset: str) -> str:
        """
        Construct a highly specific PubMed search query using directional logic 
        and mechanistic 'bridge' terms to reduce neutral results.
        """
        description = mechanism.get('description', '').lower()
        mechanism_type = mechanism.get('mechanism_type', '').lower()
        # Extract effect direction from the hypothesis data if available
        effect_direction = mechanism.get('effect_direction', 'unknown')

        # Enhanced Dataset-specific configurations with mechanistic bridges
        dataset_config = {
            'ist3': {
                'context': ['stroke', 'alteplase', 'thrombolysis', '"ischemic stroke"'],
                'features': {
                    'nihss': 'NIHSS OR "stroke severity" OR "neurological deficit" OR "infarct volume"',
                    'dbprand': '"diastolic blood pressure" OR "blood pressure" OR "hypertension"',
                    'sbprand': '"systolic blood pressure" OR "blood pressure" OR "hypertension"',
                    'weight': 'weight OR BMI OR obesity OR "body mass" OR "pharmacokinetics"',
                    'antiplat_rand': 'antiplatelet OR aspirin OR clopidogrel OR "platelet inhibition"'
                }
            },
            'accord': {
                'context': ['diabetes', '"glycemic control"', 'HbA1c', 'cardiovascular'],
                'features': {
                    'hba1c': 'HbA1c OR "glycated hemoglobin" OR "glycemic control"',
                    'sbp': '"systolic blood pressure" OR hypertension',
                    'age': 'age OR elderly OR geriatric',
                    'duration': '"diabetes duration" OR "disease duration"'
                }
            },
            'crash_2': {
                'context': ['trauma', '"tranexamic acid"', 'TXA', 'bleeding', 'hemorrhage'],
                'features': {
                    'sbp': '"systolic blood pressure" OR "blood pressure" OR hypertension OR hypotension',
                    'isbp': '"systolic blood pressure" OR "blood pressure" OR "initial blood pressure" OR hypotension',
                    'gcs': 'GCS OR "Glasgow Coma Scale" OR "consciousness level" OR coma',
                    'igcs': 'GCS OR "Glasgow Coma Scale" OR "consciousness level" OR coma',
                    'age': 'age OR elderly OR geriatric',
                    'hr': '"heart rate" OR tachycardia OR bradycardia OR pulse',
                    'ihr': '"heart rate" OR "initial heart rate" OR tachycardia OR bradycardia OR pulse',
                    'rr': '"respiratory rate" OR breathing OR ventilation',
                    'time_to_treatment': '"time to treatment" OR "treatment delay" OR "early treatment"',
                    'ninjurytime': '"time to treatment" OR "treatment delay" OR "injury time" OR "time from injury"',
                    'iinjurytype': '"injury type" OR "penetrating injury" OR "blunt trauma" OR "mechanism of injury"',
                    'injury type': '"injury type" OR "penetrating injury" OR "blunt trauma" OR "mechanism of injury"',
                    'penetrating': '"penetrating injury" OR "penetrating trauma" OR gunshot OR stabbing',
                    'blunt': '"blunt trauma" OR "blunt injury" OR "closed trauma"',
                    'icc': '"clotting capacity" OR coagulopathy OR "coagulation" OR INR OR "prothrombin time" OR fibrinogen'
                }
            },
            'sprint': {
                'context': ['hypertension', '"blood pressure"', 'cardiovascular', '"intensive treatment"'],
                'features': {
                    'sbp': '"systolic blood pressure" OR "blood pressure" OR hypertension',
                    'age': 'age OR elderly OR geriatric',
                    'cvd': '"cardiovascular disease" OR CVD OR "heart disease"',
                    'ckd': '"chronic kidney disease" OR CKD OR "renal function"'
                }
            }
        }

        config = dataset_config.get(dataset, {'context': [], 'features': {}})
        context_terms = config['context']
        
        query_parts = []
        
        # 1. Add context (e.g., stroke AND alteplase)
        if context_terms:
            query_parts.append(f"({' OR '.join(context_terms)})")
        
        # 2. Add Feature Query
        # First try to find in config (for actual database feature names)
        feature_query = config['features'].get(feature_name)
        
        if not feature_query:
            # Try lowercase version
            feature_query = config['features'].get(feature_name.lower())
        
        if not feature_query and ':' in feature_name:
            # Handle categorical features like "Injury Type: Penetrating"
            parts = feature_name.split(':', 1)
            base_name = parts[0].strip().lower()
            value_name = parts[1].strip().lower()
            feature_query = config['features'].get(base_name) or config['features'].get(value_name)
            
        if feature_query:
            # Use configured mapping
            query_parts.append(f"({feature_query})")
        else:
            # For without_shap clinical concepts or unmapped features, use as-is
            # This handles cases like "Timing of Administration", "Severity of Bleeding", etc.
            query_parts.append(f'"{feature_name}"')

        # 3. Add Mechanistic "Bridge" Terms
        # These terms force PubMed to find 'how' or 'why' rather than just 'what'
        mech_bridges = {
            'biological': '("pathophysiology" OR "mechanism" OR "causal" OR "etiology" OR "interaction")',
            'physiological': '("physiology" OR "pathogenesis" OR "homeostasis")',
            'pharmacological': '("pharmacokinetics" OR "pharmacodynamics" OR "dose-response" OR "clearance")',
            'statistical': '("independent predictor" OR "confounding" OR "interaction effect")'
        }
        if mechanism_type in mech_bridges:
            query_parts.append(mech_bridges[mechanism_type])

        # 4. Add Directional Logic
        # Based on effect_direction, add terms that reflect the hypothesis stance
        if effect_direction == "negative":
            query_parts.append('("reduced efficacy" OR "worse outcomes" OR "resistance" OR "poor prognosis" OR "complication")')
        elif effect_direction == "positive":
            query_parts.append('("enhanced" OR "benefit" OR "synergy" OR "improved" OR "favorable")')

        # 5. Extract specific medical keywords from description dynamically
        # This makes the query more mechanism-specific
        excluded_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'can', 'may', 'could', 'would', 'should', 'which', 'that', 'this',
            'have', 'has', 'had', 'not', 'more', 'less', 'than', 'when', 'where',
            'who', 'what', 'how', 'why', 'their', 'they', 'them', 'these', 'those'
        }
        
        # Extract meaningful clinical terms from description (3+ chars)
        words = re.findall(r'\b[a-z]{3,}\b', description)
        mechanism_keywords = [
            word for word in words 
            if word not in excluded_words and len(word) >= 4
        ]
        
        # Get top 3-5 most distinctive terms (longer words tend to be more specific)
        mechanism_keywords = sorted(set(mechanism_keywords), key=len, reverse=True)[:5]
        
        if mechanism_keywords:
            # Add these mechanism-specific terms to narrow the search
            keyword_query = ' OR '.join([f'"{kw}"' for kw in mechanism_keywords[:3]])
            query_parts.append(f"({keyword_query})")

        # Combine with AND logic for high specificity
        if query_parts:
            query = ' AND '.join(query_parts)
            # Add publication type filters
            query += ' AND (Clinical Trial[PT] OR Review[PT] OR Meta-Analysis[PT])'
        else:
            # Fallback if no query parts were generated
            query = f'"{feature_name.replace("_", " ")}" AND (Clinical Trial[PT] OR Review[PT] OR Meta-Analysis[PT])'
        
        return query
    
    def search_pubmed(self, query: str, max_results: int = None) -> List[str]:
        """
        Search PubMed and return list of PMIDs.
        
        Args:
            query: PubMed search query
            max_results: Maximum number of results to return
            
        Returns:
            List of PubMed IDs
        """
        if not Entrez:
            print("Warning: BioPython not available. Skipping PubMed search.")
            return []
        
        max_results = max_results or self.max_abstracts
        
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
            record = Entrez.read(handle)
            handle.close()
            return record.get("IdList", [])
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []
    
    def fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, str]]:
        """
        Fetch abstracts for given PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of dictionaries with pmid, title, and abstract
        """
        if not Entrez or not pmids:
            return []
        
        abstracts = []
        
        try:
            # Fetch in batches to avoid overwhelming the server
            batch_size = 10
            for i in range(0, len(pmids), batch_size):
                batch_pmids = pmids[i:i+batch_size]
                handle = Entrez.efetch(db="pubmed", id=batch_pmids, rettype="abstract", retmode="xml")
                records = Entrez.read(handle)
                handle.close()
                
                for record in records.get('PubmedArticle', []):
                    try:
                        article = record['MedlineCitation']['Article']
                        pmid = str(record['MedlineCitation']['PMID'])
                        title = article.get('ArticleTitle', '')
                        
                        # Extract abstract text
                        abstract_text = ''
                        if 'Abstract' in article:
                            abstract_parts = article['Abstract'].get('AbstractText', [])
                            if isinstance(abstract_parts, list):
                                abstract_text = ' '.join([str(part) for part in abstract_parts])
                            else:
                                abstract_text = str(abstract_parts)
                        
                        abstracts.append({
                            'pmid': pmid,
                            'title': title,
                            'abstract': abstract_text
                        })
                    except Exception as e:
                        print(f"Error parsing article: {e}")
                        continue
                
                # Be nice to NCBI servers
                time.sleep(0.34)
                
        except Exception as e:
            print(f"Error fetching abstracts: {e}")
        
        return abstracts
    
    def analyze_abstract_with_llm(self, abstract: Dict[str, str], mechanism: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to analyze whether an abstract supports or conflicts with a mechanism.
        
        Args:
            abstract: Dictionary with abstract text
            mechanism: Mechanism dictionary
            
        Returns:
            Dictionary with analysis results
        """
        if not self.openai_client:
            return self.analyze_abstract_keyword(abstract, mechanism)
        
        prompt = f"""You are analyzing whether a scientific abstract provides evidence for or against a specific proposed mechanism.

                    PROPOSED MECHANISM:
                    Type: {mechanism.get('mechanism_type', 'unknown')}
                    Description: {mechanism.get('description', '')}
                    Effect Direction: {mechanism.get('effect_direction', 'unknown')}
                    Evidence Level: {mechanism.get('evidence_level', 'unknown')}

                    ABSTRACT TO ANALYZE:
                    Title: {abstract.get('title', '')}
                    Text: {abstract.get('abstract', '')}

                    CLASSIFICATION CRITERIA:
                    - SUPPORT: The abstract provides evidence that directly supports this specific mechanism (e.g., shows the same relationship, validates the pathway, demonstrates the effect)
                    - CONFLICT: The abstract provides evidence that contradicts this mechanism (e.g., shows opposite effect, refutes the pathway, shows no association where mechanism predicts one)
                    - NEUTRAL: The abstract is tangentially related but doesn't specifically validate or refute this mechanism, OR discusses the general topic without addressing the specific mechanistic claim

                    Be STRICT: Only classify as "support" or "conflict" if the abstract specifically addresses elements of this mechanism. Most abstracts will be neutral.

                    Respond with ONLY a JSON object:
                    {{
                        "stance": "support|conflict|neutral",
                        "confidence": "high|medium|low",
                        "reasoning": "Brief explanation focusing on mechanism-specific evidence",
                        "key_findings": "Specific findings from abstract relevant to this mechanism"
                    }}
                    """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5.1-2025-11-13",
                messages=[
                    {"role": "system", "content": "You are a medical research expert analyzing scientific literature. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                # temperature=0.3,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON if wrapped in markdown code blocks
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            result['pmid'] = abstract.get('pmid', '')
            result['title'] = abstract.get('title', '')
            result['analysis_method'] = 'llm'
            
            return result
            
        except Exception as e:
            print(f"Error with LLM analysis: {e}")
            return self.analyze_abstract_keyword(abstract, mechanism)
    
    def analyze_abstract_keyword(self, abstract: Dict[str, str], mechanism: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple keyword-based analysis as fallback.
        
        Args:
            abstract: Dictionary with abstract text
            mechanism: Mechanism dictionary
            
        Returns:
            Dictionary with analysis results
        """
        text = (abstract.get('title', '') + ' ' + abstract.get('abstract', '')).lower()
        description = mechanism.get('description', '').lower()
        
        # Extract key medical terms from mechanism description
        mechanism_terms = []
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                       'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
                       'can', 'may', 'could', 'would', 'should', 'which', 'that', 'this'}
        
        # Extract meaningful words from description (4+ characters for better specificity)
        words = re.findall(r'\b[a-z]{4,}\b', description)
        mechanism_terms = [w for w in words if w not in common_words][:15]
        
        # Count how many mechanism terms appear in abstract (with partial matching)
        term_matches = 0
        for term in mechanism_terms:
            if term in text:
                term_matches += 1
            # Also check for partial matches (e.g., "hemorrhagic" matches "hemorrhage")
            elif any(term in word or word in term for word in text.split() if len(word) > 5):
                term_matches += 0.5
        
        term_match_ratio = term_matches / len(mechanism_terms) if mechanism_terms else 0
        
        # Support keywords (expanded and more specific)
        support_keywords = [
            'confirm', 'support', 'consistent', 'demonstrate', 'show', 'evidence', 
            'associated with', 'correlation', 'related to', 'linked to',
            'increase', 'decrease', 'improve', 'reduce', 'enhance',
            'significant', 'significantly', 'positively', 'negatively',
            'found that', 'indicates', 'suggests', 'proves', 'validates'
        ]
        
        # Conflict keywords (expanded)
        conflict_keywords = [
            'contradict', 'conflict', 'contrary', 'however', 'nevertheless',
            'no association', 'not associated', 'no evidence', 'lack of',
            'no significant', 'not significant', 'fail', 'failed', 'unable',
            'did not', 'does not', 'no effect', 'no difference', 'no relationship',
            'unlikely', 'disputed', 'questioned', 'refute', 'challenge'
        ]
        
        # Count occurrences with phrase matching
        support_count = 0
        for keyword in support_keywords:
            support_count += text.count(keyword)
        
        conflict_count = 0
        for keyword in conflict_keywords:
            conflict_count += text.count(keyword)
        
        # Determine stance with improved logic
        # If abstract is topically relevant (term matches) and has more support/conflict signals
        if term_match_ratio > 0.15:  # At least 15% of mechanism terms present (lowered threshold)
            if support_count > conflict_count and support_count >= 1:
                stance = 'support'
                confidence = 'high' if support_count >= 5 and term_match_ratio > 0.3 else 'medium' if support_count >= 3 else 'low'
            elif conflict_count > support_count and conflict_count >= 1:
                stance = 'conflict'
                confidence = 'high' if conflict_count >= 5 and term_match_ratio > 0.3 else 'medium' if conflict_count >= 3 else 'low'
            else:
                stance = 'neutral'
                confidence = 'low'
        else:
            # Low topical relevance - be more conservative
            if support_count >= 5:
                stance = 'support'
                confidence = 'low'
            elif conflict_count >= 5:
                stance = 'conflict'
                confidence = 'low'
            else:
                stance = 'neutral'
                confidence = 'low'
        
        reasoning = f"Keyword analysis: {support_count} support signals, {conflict_count} conflict signals, {term_match_ratio:.1%} term match"
        
        return {
            'pmid': abstract.get('pmid', ''),
            'title': abstract.get('title', ''),
            'stance': stance,
            'confidence': confidence,
            'reasoning': reasoning,
            'key_findings': f"Matched {term_matches}/{len(mechanism_terms)} mechanism terms",
            'analysis_method': 'keyword'
        }
    
    def validate_mechanism(self, feature_name: str, mechanism: Dict[str, Any], 
                          dataset: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Validate a single mechanism against PubMed literature.
        
        Args:
            feature_name: Feature name
            mechanism: Mechanism dictionary
            dataset: Dataset name
            use_llm: Whether to use LLM for analysis
            
        Returns:
            Dictionary with validation results
        """
        print(f"\nValidating mechanism for {feature_name} ({mechanism.get('mechanism_type', 'unknown')})")
        
        # Construct search query
        query = self.construct_search_query(feature_name, mechanism, dataset)
        print(f"Search query: {query}")
        
        # Search PubMed
        pmids = self.search_pubmed(query)
        print(f"Found {len(pmids)} articles")
        
        if not pmids:
            return {
                'feature_name': feature_name,
                'mechanism': mechanism,
                'query': query,
                'total_abstracts': 0,
                'support_count': 0,
                'conflict_count': 0,
                'neutral_count': 0,
                'abstracts_analyzed': []
            }
        
        # Fetch abstracts
        abstracts = self.fetch_abstracts(pmids)
        print(f"Retrieved {len(abstracts)} abstracts")
        
        # Analyze each abstract
        analyses = []
        support_count = 0
        conflict_count = 0
        neutral_count = 0
        
        for abstract in abstracts:
            if use_llm and self.openai_client:
                analysis = self.analyze_abstract_with_llm(abstract, mechanism)
                time.sleep(1)  # Rate limiting for API
            else:
                analysis = self.analyze_abstract_keyword(abstract, mechanism)
            
            analyses.append(analysis)
            
            # Count stances
            stance = analysis.get('stance', 'neutral')
            if stance == 'support':
                support_count += 1
            elif stance == 'conflict':
                conflict_count += 1
            else:
                neutral_count += 1
        
        # Calculate support ratio S/(S+C+N)
        support_ratio = None
        total_count = support_count + conflict_count + neutral_count
        if total_count > 0:
            support_ratio = support_count / total_count
        
        return {
            'feature_name': feature_name,
            'mechanism': mechanism,
            'query': query,
            'total_abstracts': len(abstracts),
            'support_count': support_count,
            'conflict_count': conflict_count,
            'neutral_count': neutral_count,
            'support_percentage': (support_count / len(abstracts) * 100) if abstracts else 0,
            'conflict_percentage': (conflict_count / len(abstracts) * 100) if abstracts else 0,
            'support_ratio': support_ratio,
            'abstracts_analyzed': analyses
        }
    
    def validate_all_mechanisms(self, hypotheses_file: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        Validate all mechanisms in a hypotheses file.
        
        Args:
            hypotheses_file: Path to hypotheses JSON file
            use_llm: Whether to use LLM for analysis
            
        Returns:
            Dictionary with all validation results
        """
        # Load hypotheses
        hypotheses = self.load_hypotheses(hypotheses_file)
        dataset = hypotheses.get('dataset', 'unknown')
        
        print(f"Validating mechanisms for dataset: {dataset}")
        print(f"Analysis method: {'LLM' if use_llm and self.openai_client else 'Keyword-based'}")
        
        # Process each feature and its mechanisms
        all_results = []
        
        for feature_hypothesis in hypotheses.get('feature_hypotheses', []):
            feature_name = feature_hypothesis.get('feature_name', 'unknown')
            mechanisms = feature_hypothesis.get('mechanisms', [])
            
            for mechanism in mechanisms:
                result = self.validate_mechanism(feature_name, mechanism, dataset, use_llm)
                all_results.append(result)
        
        # Create summary
        summary = {
            'dataset': dataset,
            'total_mechanisms_analyzed': len(all_results),
            'total_abstracts_retrieved': sum(r['total_abstracts'] for r in all_results),
            'overall_support_count': sum(r['support_count'] for r in all_results),
            'overall_conflict_count': sum(r['conflict_count'] for r in all_results),
            'overall_neutral_count': sum(r['neutral_count'] for r in all_results),
            'mechanism_results': all_results
        }
        
        return summary
    
    def generate_report(self, results: Dict[str, Any], output_file: str = None):
        """
        Generate a report from validation results.
        
        Args:
            results: Validation results dictionary
            output_file: Optional output file path
        """
        print("\n" + "="*80)
        print(f"PUBMED VALIDATION REPORT: {results['dataset'].upper()}")
        print("="*80)
        
        print(f"\nTotal Mechanisms Analyzed: {results['total_mechanisms_analyzed']}")
        print(f"Total Abstracts Retrieved: {results['total_abstracts_retrieved']}")
        print(f"\nOverall Stance Distribution:")
        print(f"  Supporting: {results['overall_support_count']}")
        print(f"  Conflicting: {results['overall_conflict_count']}")
        print(f"  Neutral: {results['overall_neutral_count']}")
        
        # Calculate average rates across all mechanisms
        total_abstracts = results['total_abstracts_retrieved']
        if total_abstracts > 0:
            avg_support_rate = (results['overall_support_count'] / total_abstracts) * 100
            avg_conflict_rate = (results['overall_conflict_count'] / total_abstracts) * 100
            avg_neutral_rate = (results['overall_neutral_count'] / total_abstracts) * 100
            print(f"\nAverage Rates Across All Mechanisms:")
            print(f"  Support Rate: {avg_support_rate:.2f}%")
            print(f"  Conflict Rate: {avg_conflict_rate:.2f}%")
            print(f"  Neutral Rate: {avg_neutral_rate:.2f}%")
            
            # Calculate average support ratio
            avg_support_ratio = results['overall_support_count'] / total_abstracts
            print(f"  Average Support Ratio S/(S+C+N): {avg_support_ratio:.3f}")
        
        # Calculate average rates per feature
        feature_stats = {}
        for result in results['mechanism_results']:
            feature_name = result['feature_name']
            if feature_name not in feature_stats:
                feature_stats[feature_name] = {
                    'support': 0, 'conflict': 0, 'neutral': 0, 'total': 0
                }
            feature_stats[feature_name]['support'] += result['support_count']
            feature_stats[feature_name]['conflict'] += result['conflict_count']
            feature_stats[feature_name]['neutral'] += result['neutral_count']
            feature_stats[feature_name]['total'] += result['total_abstracts']
        
        # Calculate average across features
        if feature_stats:
            feature_support_rates = []
            feature_conflict_rates = []
            feature_neutral_rates = []
            feature_support_ratios = []
            
            for feature_name, stats in feature_stats.items():
                if stats['total'] > 0:
                    feature_support_rates.append((stats['support'] / stats['total']) * 100)
                    feature_conflict_rates.append((stats['conflict'] / stats['total']) * 100)
                    feature_neutral_rates.append((stats['neutral'] / stats['total']) * 100)
                    feature_support_ratios.append(stats['support'] / stats['total'])
            
            if feature_support_rates:
                print(f"\nAverage Rates Across Features (n={len(feature_support_rates)}):")
                print(f"  Avg Support Rate per Feature: {sum(feature_support_rates)/len(feature_support_rates):.2f}%")
                print(f"  Avg Conflict Rate per Feature: {sum(feature_conflict_rates)/len(feature_conflict_rates):.2f}%")
                print(f"  Avg Neutral Rate per Feature: {sum(feature_neutral_rates)/len(feature_neutral_rates):.2f}%")
                print(f"  Avg Support Ratio per Feature: {sum(feature_support_ratios)/len(feature_support_ratios):.3f}")
        
        print("\n" + "-"*80)
        print("MECHANISM-SPECIFIC RESULTS")
        print("-"*80)
        
        for result in results['mechanism_results']:
            print(f"\nFeature: {result['feature_name']}")
            print(f"Mechanism Type: {result['mechanism']['mechanism_type']}")
            print(f"Description: {result['mechanism']['description'][:100]}...")
            print(f"Abstracts Found: {result['total_abstracts']}")
            if result['total_abstracts'] > 0:
                print(f"  Support: {result['support_count']} ({result['support_percentage']:.1f}%)")
                print(f"  Conflict: {result['conflict_count']} ({result['conflict_percentage']:.1f}%)")
                print(f"  Neutral: {result['neutral_count']}")
                if result['support_ratio'] is not None:
                    print(f"  Support Ratio S/(S+C): {result['support_ratio']:.3f}")
        
        # Save detailed results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate mechanisms against PubMed literature')
    parser.add_argument('--input', type=str, help='Path to hypotheses JSON file')
    parser.add_argument('--cohort', type=str, choices=['ist3', 'accord', 'crash_2', 'sprint'],
                       help='Cohort name (alternative to --input)')
    parser.add_argument('--output', type=str, help='Output file path for results')
    parser.add_argument('--email', type=str, default='research@example.com',
                       help='Email for PubMed API')
    parser.add_argument('--api-key', type=str, help='OpenAI API key for LLM analysis')
    parser.add_argument('--max-abstracts', type=int, default=30,
                       help='Maximum abstracts per mechanism')
    parser.add_argument('--no-llm', action='store_true',
                       help='Use keyword-based analysis instead of LLM')
    
    args = parser.parse_args()
    
    # Determine input file
    if args.input:
        input_file = args.input
    elif args.cohort:
        input_file = f'/homes/gws/mingyulu/shap_IPW/docs/agent/{args.cohort}/hypotheses_with_shap_XLearner.json'
    else:
        print("Error: Must specify either --input or --cohort")
        return
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.dirname(input_file)
        output_file = os.path.join(output_dir, f"{base_name}_pubmed_validation.json")
    
    # Create validator
    validator = PubMedMechanismValidator(
        email=args.email,
        api_key=api_key,
        max_abstracts=args.max_abstracts
    )
    
    # Run validation
    use_llm = not args.no_llm
    results = validator.validate_all_mechanisms(input_file, use_llm=use_llm)
    
    # Generate report
    validator.generate_report(results, output_file)


if __name__ == '__main__':
    main()
