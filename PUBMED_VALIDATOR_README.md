# PubMed Mechanism Validator

This tool validates proposed clinical mechanisms against existing PubMed literature by:
1. Searching PubMed for relevant abstracts
2. Analyzing whether abstracts support or conflict with each mechanism
3. Generating summary reports

## Installation

```bash
pip install -r pubmed_requirements.txt
```

## Usage

### Basic Usage with Cohort Name

```bash
# Validate IST3 mechanisms
python pubmed_mechanism_validator.py --cohort ist3

# Validate ACCORD mechanisms
python pubmed_mechanism_validator.py --cohort accord

# Validate CRASH-2 mechanisms
python pubmed_mechanism_validator.py --cohort crash_2

# Validate SPRINT mechanisms
python pubmed_mechanism_validator.py --cohort sprint
```

### Advanced Usage

```bash
# Use custom input file
python pubmed_mechanism_validator.py --input docs/agent/ist3/hypotheses_with_shap_XLearner.json

# Specify output file
python pubmed_mechanism_validator.py --cohort ist3 --output my_validation.json

# Use LLM analysis (requires OpenAI API key)
export OPENAI_API_KEY="your-api-key-here"
python pubmed_mechanism_validator.py --cohort ist3

# Or provide API key directly
python pubmed_mechanism_validator.py --cohort ist3 --api-key "your-api-key-here"

# Use keyword-based analysis (no API key needed)
python pubmed_mechanism_validator.py --cohort ist3 --no-llm

# Adjust number of abstracts per mechanism
python pubmed_mechanism_validator.py --cohort ist3 --max-abstracts 50

# Provide your email for PubMed API
python pubmed_mechanism_validator.py --cohort ist3 --email "your-email@domain.com"
```

## Analysis Methods

### 1. LLM-Based Analysis (Recommended)
- Requires OpenAI API key
- Uses GPT-4 to analyze abstract relevance
- Provides detailed reasoning for each classification
- More accurate but requires API costs

### 2. Keyword-Based Analysis (Fallback)
- No API key required
- Uses simple keyword matching
- Free but less accurate
- Good for initial screening

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "dataset": "ist3",
  "total_mechanisms_analyzed": 15,
  "total_abstracts_retrieved": 245,
  "overall_support_count": 150,
  "overall_conflict_count": 30,
  "overall_neutral_count": 65,
  "mechanism_results": [
    {
      "feature_name": "nihss",
      "mechanism": {...},
      "query": "PubMed search query used",
      "total_abstracts": 20,
      "support_count": 15,
      "conflict_count": 2,
      "neutral_count": 3,
      "support_percentage": 75.0,
      "conflict_percentage": 10.0,
      "abstracts_analyzed": [
        {
          "pmid": "12345678",
          "title": "Article title",
          "stance": "support",
          "confidence": "high",
          "reasoning": "Detailed explanation...",
          "key_findings": "Key findings from abstract",
          "analysis_method": "llm"
        }
      ]
    }
  ]
}
```

## Features

- **Automatic Query Construction**: Builds appropriate PubMed queries based on feature names and mechanism descriptions
- **Dataset-Specific Context**: Tailors searches to specific clinical contexts (stroke, diabetes, trauma, hypertension)
- **Quality Filtering**: Focuses on high-quality evidence (clinical trials, meta-analyses, reviews, RCTs)
- **Batch Processing**: Efficiently processes multiple mechanisms and features
- **Rate Limiting**: Respects NCBI API rate limits
- **Comprehensive Reporting**: Generates both console and JSON output

## Example Output

```
================================================================================
PUBMED VALIDATION REPORT: IST3
================================================================================

Total Mechanisms Analyzed: 15
Total Abstracts Retrieved: 245

Overall Stance Distribution:
  Supporting: 150
  Conflicting: 30
  Neutral: 65

--------------------------------------------------------------------------------
MECHANISM-SPECIFIC RESULTS
--------------------------------------------------------------------------------

Feature: nihss
Mechanism Type: biological
Description: Higher NIHSS scores indicate more severe cerebral injury, which can limit the efficacy...
Abstracts Found: 20
  Support: 15 (75.0%)
  Conflict: 2 (10.0%)
  Neutral: 3

Feature: dbprand
Mechanism Type: physiological
Description: High diastolic pressure may perpetuate cerebral edema, reducing alteplase efficacy...
Abstracts Found: 18
  Support: 12 (66.7%)
  Conflict: 3 (16.7%)
  Neutral: 3
```

## Best Practices

1. **Use LLM Analysis for Final Reports**: More accurate and provides detailed reasoning
2. **Start with Keyword Analysis**: Quick and free for initial screening
3. **Adjust Max Abstracts**: Increase for comprehensive analysis, decrease for quick checks
4. **Review Query Construction**: Check generated queries to ensure they match your intent
5. **Rate Limiting**: Be patient - the script respects NCBI API limits

## Limitations

- PubMed API has rate limits (3 requests/second without API key)
- LLM analysis requires API costs
- Keyword analysis is less nuanced than expert review
- Abstract availability varies by publication date and journal

## Troubleshooting

**No abstracts found:**
- Check that BioPython is installed
- Verify internet connection
- Try broader search terms

**LLM analysis not working:**
- Verify OpenAI API key is set
- Check API credit balance
- Try keyword analysis as fallback

**Too many/few results:**
- Adjust `--max-abstracts` parameter
- Modify query construction in code for specific needs

## Citation

If you use this tool in your research, please cite appropriately and acknowledge the use of PubMed/NCBI resources.
