// Condition mapping (blinded for raters)
// Randomized per cohort to prevent systematic bias
// DO NOT SHARE THIS MAPPING WITH RATERS
const conditionMapping = {
    crash_2: {
        condition_a: 'with_shap',           // A = SHAP
        condition_b: 'without_shap_baseline' // B = Literature
    },
    ist3: {
        condition_a: 'without_shap_baseline', // A = Literature
        condition_b: 'with_shap'              // B = SHAP
    },
    sprint: {
        condition_a: 'with_shap',           // A = SHAP
        condition_b: 'without_shap_baseline' // B = Literature
    },
    accord: {
        condition_a: 'without_shap_baseline', // A = Literature
        condition_b: 'with_shap'              // B = SHAP
    }
};

// Trial metadata
const trialInfo = {
    crash_2: {
        treatment: "Tranexamic acid (TXA)",
        outcome: "All-cause mortality at 28 days or in-hospital death",
        population: "Trauma patients with significant bleeding or at risk of significant hemorrhage"
    },
    ist3: {
        treatment: "IV alteplase (recombinant tissue plasminogen activator)",
        outcome: "Alive and independent (Oxford Handicap Score 0-2) at 6 months",
        population: "Acute ischemic stroke patients within 6 hours of symptom onset"
    },
    sprint: {
        treatment: "Intensive blood pressure control (systolic BP target <120 mmHg)",
        outcome: "Composite of major cardiovascular events (MI, stroke, heart failure, cardiovascular death)",
        population: "Non-diabetic adults aged ≥50 with hypertension and increased cardiovascular risk"
    },
    accord: {
        treatment: "Intensive glucose control (HbA1c target <6.0%)",
        outcome: "Major cardiovascular events (nonfatal MI, nonfatal stroke, cardiovascular death)",
        population: "Adults with type 2 diabetes and high cardiovascular risk"
    }
};

// Rating criteria with descriptions
const ratingCriteria = [
    {
        id: 'mechanism_plausibility',
        label: 'Mechanism Plausibility',
        description: 'Biological/clinical coherence, consistency with pathophysiology, specificity'
    },
    {
        id: 'evidence_alignment',
        label: 'Evidence Alignment',
        description: 'Grounding in published trials, systematic reviews, established literature'
    },
    {
        id: 'subgroup_implications',
        label: 'Subgroup Implications',
        description: 'Clarity, actionability, feasibility, clinical utility'
    },
    {
        id: 'caveat_awareness',
        label: 'Caveat Awareness',
        description: 'Acknowledgment of limitations, confounding, epistemic humility'
    }
];

let currentHypotheses = [];
let ratings = {};

// Load hypotheses when button is clicked
document.getElementById('load-btn').addEventListener('click', loadHypotheses);

async function loadHypotheses() {
    const cohort = document.getElementById('cohort-select').value;
    const conditionBlind = document.getElementById('condition-select').value;
    const expertise = document.getElementById('expertise-select').value;
    const specialty = document.getElementById('specialty-input').value.trim();

    if (!cohort) {
        alert('Please select a trial cohort');
        return;
    }

    if (!expertise) {
        alert('Please select your clinical expertise level');
        return;
    }

    if (!specialty) {
        alert('Please enter your specialty');
        return;
    }

    // Map blinded condition to actual file path based on cohort
    const condition = conditionMapping[cohort][conditionBlind];

    // WITHOUT SHAP files don't include learner name in filename
    // crash_2 and ist3 use DRLearner, others use XLearner
    const learner = (cohort === 'crash_2' || cohort === 'ist3') ? 'DRLearner' : 'XLearner';
    const fileName = condition === 'without_shap_baseline'
        ? `hypotheses_${condition}.json`
        : `hypotheses_${condition}_${learner}.json`;

    const filePath = `agent/${cohort}/${fileName}`;

    try {
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        displayTrialInfo(cohort);
        displayHypotheses(data.feature_hypotheses, cohort, conditionBlind, expertise, specialty);

    } catch (error) {
        const container = document.getElementById('hypotheses-container');
        container.innerHTML = `
            <div class="error">
                <strong>Error loading hypotheses:</strong> ${error.message}<br>
                <small>Make sure the JSON files are in the correct location: docs/agent/${cohort}/</small>
            </div>
        `;
    }
}

function displayTrialInfo(cohort) {
    const info = trialInfo[cohort];
    document.getElementById('trial-treatment').textContent = info.treatment;
    document.getElementById('trial-outcome').textContent = info.outcome;
    document.getElementById('trial-population').textContent = info.population;
    document.getElementById('trial-info').style.display = 'block';
}

function displayHypotheses(hypotheses, cohort, condition, expertise, specialty) {
    currentHypotheses = hypotheses;
    ratings = {
        expertise: expertise,
        specialty: specialty,
        cohort: cohort,
        condition: condition,
        timestamp: new Date().toISOString(),
        ratings: []
    };

    const container = document.getElementById('hypotheses-container');
    container.innerHTML = '';

    hypotheses.forEach((hyp, index) => {
        const card = createHypothesisCard(hyp, index);
        container.appendChild(card);
    });

    document.getElementById('summary-section').style.display = 'block';
}

function createHypothesisCard(hypothesis, index) {
    const card = document.createElement('div');
    card.className = 'hypothesis-card';
    card.id = `hyp-${index}`;

    card.innerHTML = `
        <div class="hypothesis-header">
            <div class="hypothesis-title">Feature ${index + 1}</div>
            <div>
                <span class="feature-badge">Rank: ${hypothesis.importance_rank || index + 1}</span>
            </div>
        </div>

        <div class="hypothesis-content">
            <div class="content-section">
                <h4>Clinical Interpretation</h4>
                <p>${hypothesis.clinical_interpretation}</p>
            </div>

            <div class="content-section">
                <h4>Why Important for Treatment Heterogeneity</h4>
                <p>${hypothesis.why_important}</p>
            </div>

            <div class="content-section">
                <h4>Proposed Mechanisms</h4>
                ${hypothesis.mechanisms.map(m => `
                    <div class="mechanism-item">
                        <strong>${m.mechanism_type}:</strong> ${m.description}
                        <br><small><em>Evidence level: ${m.evidence_level}</em></small>
                    </div>
                `).join('')}
            </div>

            <div class="content-section">
                <h4>Subgroup Implications</h4>
                <p>${hypothesis.subgroup_implications}</p>
            </div>

            <div class="content-section">
                <h4>Validation Suggestions</h4>
                <ul>
                    ${hypothesis.validation_suggestions.map(v => `<li>${v}</li>`).join('')}
                </ul>
            </div>

            <div class="content-section">
                <h4>Caveats</h4>
                <ul>
                    ${hypothesis.caveats.map(c => `<li>${c}</li>`).join('')}
                </ul>
            </div>
        </div>

        <div class="rating-section">
            <h4>Your Ratings (1-5 scale)</h4>
            ${createRatingInputs(index)}

            <div class="rating-group">
                <label class="rating-label">Additional Comments (optional)</label>
                <textarea id="comments-${index}" placeholder="Any additional thoughts, concerns, or suggestions..."></textarea>
            </div>
        </div>
    `;

    return card;
}

function createRatingInputs(hypIndex) {
    return ratingCriteria.map(criterion => `
        <div class="rating-group">
            <label class="rating-label">${criterion.label}</label>
            <div class="rating-description">${criterion.description}</div>
            <div class="rating-input">
                <input
                    type="range"
                    id="${criterion.id}-${hypIndex}"
                    min="1"
                    max="5"
                    value="5"
                    oninput="updateRatingValue('${criterion.id}', ${hypIndex})"
                >
                <span class="rating-value" id="${criterion.id}-${hypIndex}-value">5</span>
            </div>
        </div>
    `).join('');
}

function updateRatingValue(criterionId, hypIndex) {
    const input = document.getElementById(`${criterionId}-${hypIndex}`);
    const display = document.getElementById(`${criterionId}-${hypIndex}-value`);
    display.textContent = input.value;
}

// Export ratings
document.getElementById('export-btn').addEventListener('click', exportRatings);

function exportRatings() {
    const expertise = document.getElementById('expertise-select').value;
    const specialty = document.getElementById('specialty-input').value.trim();

    if (!expertise) {
        alert('Please select your clinical expertise level');
        return;
    }

    if (!specialty) {
        alert('Please enter your specialty');
        return;
    }

    // Collect all ratings
    ratings.ratings = currentHypotheses.map((hyp, index) => {
        const featureRating = {
            feature_name: hyp.feature_name,
            feature_index: index,
        };

        // Collect scores for each criterion
        ratingCriteria.forEach(criterion => {
            const input = document.getElementById(`${criterion.id}-${index}`);
            featureRating[criterion.id] = parseInt(input.value);
        });

        // Collect comments
        const comments = document.getElementById(`comments-${index}`).value.trim();
        if (comments) {
            featureRating.comments = comments;
        }

        return featureRating;
    });

    // Create download
    const dataStr = JSON.stringify(ratings, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `ratings_${ratings.cohort}_${ratings.condition}_${Date.now()}.json`;
    link.click();

    URL.revokeObjectURL(url);

    alert('Ratings exported successfully! Please submit the downloaded JSON file.');
}
