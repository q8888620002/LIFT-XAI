// Blinded method mapping (randomized per cohort to prevent systematic bias)
// DO NOT SHARE THIS MAPPING WITH RATERS
const methodMapping = {
    crash_2: {
        method_a: 'with_shap_drlearner',
        method_b: 'hypogenic',
        method_c: 'cot',
        method_d: 'researchagent'
    },
    ist3: {
        method_a: 'cot',
        method_b: 'with_shap_drlearner',
        method_c: 'researchagent',
        method_d: 'hypogenic'
    },
    sprint: {
        method_a: 'researchagent',
        method_b: 'cot',
        method_c: 'hypogenic',
        method_d: 'with_shap_drlearner'
    },
    accord: {
        method_a: 'hypogenic',
        method_b: 'researchagent',
        method_c: 'with_shap_drlearner',
        method_d: 'cot'
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

// Feature name mapping for clean display
const featureNameMap = {
    // IST-3 features
    'nihss': 'NIHSS Score',
    'age': 'Age',
    'weight': 'Weight',
    'glucose': 'Blood Glucose',
    'gcs_score_rand': 'Glasgow Coma Scale Score',
    'sbprand': 'Systolic Blood Pressure',
    'dbprand': 'Diastolic Blood Pressure',
    'gender': 'Female Gender',
    'antiplat_rand': 'Antiplatelet Usage',
    'atrialfib_rand': 'Atrial Fibrillation History',
    'infarct': 'Prior Infarct',
    'stroketype_1': 'Stroke Type: TACI',
    'stroketype_2': 'Stroke Type: PACI',
    'stroketype_3': 'Stroke Type: POCI',
    'stroketype_4': 'Stroke Type: LACI',
    'Stroke Type: TACI (Total Anterior Circulation Infarct)': 'Stroke Type: TACI',

    // CRASH-2 features
    'iage': 'Age',
    'isbp': 'Systolic Blood Pressure',
    'irr': 'Respiratory Rate',
    'icc': 'Central Capillary Refill Time',
    'ihr': 'Heart Rate',
    'igcs': 'Glasgow Coma Scale',
    'ninjurytime': 'Time from Injury to Treatment',
    'isex': 'Male Gender',
    'iinjurytype': 'Injury Type',
    'iinjurytype_1': 'Injury Type: Blunt',
    'iinjurytype_2': 'Injury Type: Penetrating',
    'Injury Type: Penetrating': 'Injury Type: Penetrating',
    'Injury Type: Blunt': 'Injury Type: Blunt',

    // SPRINT features
    'sbp': 'Systolic Blood Pressure',
    'dbp': 'Diastolic Blood Pressure',
    'n_agents': 'Number of Antihypertensive Agents',
    'egfr': 'Estimated Glomerular Filtration Rate',
    'screat': 'Serum Creatinine',
    'chr': 'Total Cholesterol/HDL Ratio',
    'glur': 'Fasting Glucose',
    'hdl': 'HDL Cholesterol',
    'trr': 'Triglycerides',
    'umalcr': 'Urine Albumin-Creatinine Ratio',
    'bmi': 'Body Mass Index',
    'female': 'Female Gender',
    'race_black': 'Black Race',
    'smoke_3cat': 'Current Smoker',
    'aspirin': 'Aspirin Use',
    'statin': 'Statin Use',
    'sub_cvd': 'History of Cardiovascular Disease',
    'sub_ckd': 'Chronic Kidney Disease',

    // ACCORD features
    'baseline_age': 'Age',
    'hr': 'Heart Rate',
    'fpg': 'Fasting Plasma Glucose',
    'alt': 'Alanine Aminotransferase',
    'cpk': 'Creatine Phosphokinase',
    'potassium': 'Serum Potassium',
    'gfr': 'Glomerular Filtration Rate',
    'uacr': 'Urine Albumin-Creatinine Ratio',
    'chol': 'Total Cholesterol',
    'trig': 'Triglycerides',
    'vldl': 'VLDL Cholesterol',
    'ldl': 'LDL Cholesterol',
    'bp_med': 'Number of Blood Pressure Medications',
    'raceclass': 'Black Race',
    'cvd_hx_baseline': 'History of Cardiovascular Disease',
    'antiarrhythmic': 'Antiarrhythmic Medication Use',
    'anti_coag': 'Anticoagulant Use',
    'x4smoke': 'Current Smoker',
};

function getDisplayFeatureName(featureName) {
    // Return mapped name if exists, otherwise return original name
    return featureNameMap[featureName] || featureName;
}

// Rating criteria: 4 robustness gates + novelty bonus
// Aligned with judge_evaluation.py gate logic
const ratingGates = [
    {
        id: 'is_biologically_coherent',
        label: 'Gate 1: Logical Coherence',
        description: 'Is the proposed mechanism logically coherent? Does it provide a plausible explanation (biological, pharmacological, physiological, or clinical) that mechanistically connects the feature to differential treatment effect? FALSE if only a statistical/epidemiological claim, circular reasoning, or logically inconsistent.'
    },
    {
        id: 'is_causally_plausible',
        label: 'Gate 2: Genuine HTE vs Statistical Artifact',
        description: 'Is this a true treatment effect modifier — the drug works differently in this subgroup — rather than a statistical artifact? FALSE if the sole argument is absolute-risk amplification (higher baseline risk × constant RRR), post-treatment variable, reverse causality, or trivial severity proxy.'
    },
    {
        id: 'is_clinically_actionable',
        label: 'Gate 3: Practical Utility',
        description: 'Does this propose clear, operationalisable patient subgroups with distinct treatment recommendations usable in clinical practice?'
    },
    {
        id: 'is_literature_backed',
        label: 'Gate 4: External Evidence',
        description: 'Is this specific feature × treatment interaction supported by published clinical literature (ideally RCT subgroup analyses or meta-analyses)?'
    }
];

const noveltyBonus = {
    id: 'is_novel',
    label: 'Novelty Bonus',
    description: 'Does this hypothesis identify an underexplored mechanism or subgroup not already well-covered in existing clinical guidelines or major reviews? (Does not affect overall score.)'
};



let currentHypotheses = [];
let ratings = {};

// Load hypotheses when button is clicked
document.getElementById('load-btn').addEventListener('click', loadHypotheses);

async function loadHypotheses() {
    const cohort = document.getElementById('cohort-select').value;
    const methodBlind = document.getElementById('method-select').value;
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

    // Map blinded label to actual method based on cohort
    const method = methodMapping[cohort][methodBlind];
    const filePath = `agent/${cohort}/gpt-5-mini/${method}/seed_0/hypotheses.json`;

    try {
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        // Normalize different JSON formats into unified hypothesis list
        const hypotheses = normalizeHypotheses(data, method);

        displayTrialInfo(cohort);
        displayHypotheses(hypotheses, cohort, methodBlind, expertise, specialty);

    } catch (error) {
        const container = document.getElementById('hypotheses-container');
        container.innerHTML = `
            <div class="error">
                <strong>Error loading hypotheses:</strong> ${error.message}<br>
                <small>Expected path: ${filePath}</small>
            </div>
        `;
    }
}

// Normalize different method JSON formats into a common structure
function normalizeHypotheses(data, method) {
    if (method === 'hypogenic') {
        // HypoGeniC uses a dict keyed by hypothesis text
        return Object.values(data.hypotheses || {}).map((h, i) => ({
            feature_name: h.subgroup_rule?.feature || `Hypothesis ${i + 1}`,
            hypothesis_text: h.hypothesis,
            mechanisms: [{ description: h.hypothesis }],
            subgroup_rule: h.subgroup_rule,
            recommendation: h.recommendation,
            importance_rank: i + 1
        }));
    }
    // ALEX, CoT, ResearchAgent all use feature_hypotheses array
    return (data.feature_hypotheses || []).map((h, i) => ({
        feature_name: h.feature_name,
        mechanisms: (h.mechanisms || []).map(m => ({ description: m.description })),
        importance_rank: h.importance_rank || i + 1
    }));
}

function displayTrialInfo(cohort) {
    const info = trialInfo[cohort];
    document.getElementById('trial-treatment').textContent = info.treatment;
    document.getElementById('trial-outcome').textContent = info.outcome;
    document.getElementById('trial-population').textContent = info.population;
    document.getElementById('trial-info').style.display = 'block';
}

function displayHypotheses(hypotheses, cohort, method, expertise, specialty) {
    currentHypotheses = hypotheses;
    ratings = {
        expertise: expertise,
        specialty: specialty,
        cohort: cohort,
        method: method,
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

    // Build subgroup rule display for HypoGeniC
    const subgroupHTML = hypothesis.subgroup_rule ? `
            <div class="content-section">
                <h4>Subgroup Rule</h4>
                <p><code>${hypothesis.subgroup_rule.feature} ${hypothesis.subgroup_rule.operator} ${hypothesis.subgroup_rule.threshold}</code>
                — ${hypothesis.subgroup_rule.description || ''}
                (Recommendation: <strong>${hypothesis.recommendation || 'N/A'}</strong>)</p>
            </div>` : '';

    card.innerHTML = `
        <div class="hypothesis-header">
            <div class="hypothesis-title">${getDisplayFeatureName(hypothesis.feature_name)}</div>
            <div>
                <span class="feature-badge">Rank: ${hypothesis.importance_rank || index + 1}</span>
            </div>
        </div>

        <div class="hypothesis-content">
            <div class="content-section">
                <h4>Hypotheses</h4>
                ${hypothesis.mechanisms.map(m => `
                    <div class="mechanism-item">
                        ${m.description}
                    </div>
                `).join('')}
            </div>
            ${subgroupHTML}
        </div>

        <div class="rating-section">
            <h4>Gate Evaluation</h4>
            <p class="gate-instructions">For each gate, select TRUE or FALSE. Gates are evaluated independently.</p>
            ${createGateInputs(index)}

            <div class="rating-group">
                <label class="rating-label">Justification / Comments (optional)</label>
                <textarea id="comments-${index}" placeholder="Brief reasoning for your gate decisions, or any additional thoughts..."></textarea>
            </div>
        </div>
    `;

    return card;
}

function createGateInputs(hypIndex) {
    const gateHTML = ratingGates.map(gate => `
        <div class="rating-group gate-group">
            <label class="rating-label">${gate.label}</label>
            <div class="rating-description">${gate.description}</div>
            <div class="gate-toggle">
                <button type="button" class="gate-btn gate-btn-true" id="${gate.id}-${hypIndex}-true"
                    onclick="setGate('${gate.id}', ${hypIndex}, true)">
                    TRUE
                </button>
                <button type="button" class="gate-btn gate-btn-false" id="${gate.id}-${hypIndex}-false"
                    onclick="setGate('${gate.id}', ${hypIndex}, false)">
                    FALSE
                </button>
                <span class="gate-status" id="${gate.id}-${hypIndex}-status">Not rated</span>
            </div>
        </div>
    `).join('');

    const noveltyHTML = `
        <div class="rating-group gate-group novelty-group">
            <label class="rating-label">${noveltyBonus.label}</label>
            <div class="rating-description">${noveltyBonus.description}</div>
            <div class="gate-toggle">
                <button type="button" class="gate-btn gate-btn-true" id="${noveltyBonus.id}-${hypIndex}-true"
                    onclick="setGate('${noveltyBonus.id}', ${hypIndex}, true)">
                    TRUE
                </button>
                <button type="button" class="gate-btn gate-btn-false" id="${noveltyBonus.id}-${hypIndex}-false"
                    onclick="setGate('${noveltyBonus.id}', ${hypIndex}, false)">
                    FALSE
                </button>
                <span class="gate-status" id="${noveltyBonus.id}-${hypIndex}-status">Not rated</span>
            </div>
        </div>
    `;

    return gateHTML + noveltyHTML;
}

function setGate(gateId, hypIndex, value) {
    const trueBtn = document.getElementById(`${gateId}-${hypIndex}-true`);
    const falseBtn = document.getElementById(`${gateId}-${hypIndex}-false`);
    const status = document.getElementById(`${gateId}-${hypIndex}-status`);

    // Clear both
    trueBtn.classList.remove('active');
    falseBtn.classList.remove('active');

    if (value) {
        trueBtn.classList.add('active');
        status.textContent = 'TRUE';
        status.className = 'gate-status gate-true';
    } else {
        falseBtn.classList.add('active');
        status.textContent = 'FALSE';
        status.className = 'gate-status gate-false';
    }

    // Store the value
    trueBtn.dataset.value = value ? 'true' : '';
    falseBtn.dataset.value = value ? '' : 'false';
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

    // Collect all gate ratings
    ratings.ratings = currentHypotheses.map((hyp, index) => {
        const featureRating = {
            feature_name: hyp.feature_name,
            feature_index: index,
        };

        // Collect gate values
        let allRated = true;
        ratingGates.forEach(gate => {
            const trueBtn = document.getElementById(`${gate.id}-${index}-true`);
            if (trueBtn.classList.contains('active')) {
                featureRating[gate.id] = true;
            } else {
                const falseBtn = document.getElementById(`${gate.id}-${index}-false`);
                if (falseBtn.classList.contains('active')) {
                    featureRating[gate.id] = false;
                } else {
                    featureRating[gate.id] = null;
                    allRated = false;
                }
            }
        });

        // Collect novelty bonus
        const noveltyTrue = document.getElementById(`${noveltyBonus.id}-${index}-true`);
        const noveltyFalse = document.getElementById(`${noveltyBonus.id}-${index}-false`);
        if (noveltyTrue.classList.contains('active')) {
            featureRating[noveltyBonus.id] = true;
        } else if (noveltyFalse.classList.contains('active')) {
            featureRating[noveltyBonus.id] = false;
        } else {
            featureRating[noveltyBonus.id] = null;
        }

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
    link.download = `ratings_${ratings.cohort}_${ratings.method}_${Date.now()}.json`;
    link.click();

    URL.revokeObjectURL(url);

    alert('Ratings exported successfully! Please submit the downloaded JSON file.');
}
