# Clinical Hypothesis Evaluation Interface

A simple web interface for clinicians to review and rate mechanistic hypotheses for treatment effect heterogeneity.

## Setup for GitHub Pages

1. **Enable GitHub Pages:**
   - Go to your repository settings
   - Navigate to "Pages" section
   - Set source to "Deploy from a branch"
   - Select the `main` branch and `/docs` folder
   - Click Save

2. **Access the site:**
   - Your site will be available at: `https://<your-username>.github.io/<repository-name>/`
   - It may take a few minutes for the site to become available

## File Structure

```
docs/
├── index.html       # Main interface
├── style.css        # Styling
├── script.js        # Interactive functionality
└── README.md        # This file
```

## Usage

1. **Select Trial Cohort**: Choose from CRASH-2, IST-3, SPRINT, or ACCORD
2. **Select Condition**: 
   - WITH SHAP: Hypotheses generated with data-driven SHAP guidance
   - WITHOUT SHAP: Hypotheses generated from clinical literature only
3. **Enter Your ID**: Your initials or name for tracking
4. **Load Hypotheses**: Click to load the hypotheses
5. **Rate Each Hypothesis**: Use the 1-10 sliders for each criterion:
   - Mechanism Plausibility
   - Clinical Interpretation
   - Evidence Alignment
   - Subgroup Implications
   - Validation Plan Quality
   - Caveat Awareness
6. **Add Comments**: Optional additional feedback
7. **Export Ratings**: Download your ratings as a JSON file

## Rating Criteria

**1-10 Scale Guidelines:**
- **9-10**: Excellent, strong evidence, well-executed
- **7-8**: Good, solid approach with minor limitations
- **5-6**: Adequate, moderate quality, notable gaps
- **3-4**: Poor, significant issues
- **1-2**: Very poor, fundamental flaws

## Data Collection

Exported ratings are saved as JSON files with the format:
```
ratings_<cohort>_<condition>_<rater_id>_<timestamp>.json
```

Submit these files for analysis.

## Requirements

The interface expects hypothesis JSON files to be located at:
```
docs/agent/<cohort>/hypotheses_<condition>_XLearner.json
```

Make sure these files are in the `docs/agent/` directory and committed to your repository.

## Troubleshooting

**Hypotheses not loading?**
- Check that JSON files are in the correct location
- Ensure files are committed to the repository
- Check browser console for errors (F12)

**GitHub Pages not working?**
- Wait a few minutes after enabling Pages
- Check that the `docs` folder is in the `main` branch
- Verify Pages settings point to `/docs` folder
