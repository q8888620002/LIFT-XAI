const ALEX_METHOD = 'with_shap_drlearner';

// Trial metadata
const trialInfo = {
    crash_2: {
        treatment: "Tranexamic acid (TXA)",
        outcome: "All-cause mortality at 28 days or in-hospital death",
        population: "Trauma patients with significant bleeding or at risk of significant hemorrhage",
        description: "CRASH-2 was a large international randomised placebo-controlled trial (N=20,211) evaluating the effect of early administration of tranexamic acid on death, vascular occlusive events, and blood transfusion in adult trauma patients with or at risk of significant bleeding, conducted across 274 hospitals in 40 countries.",
        link: "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(10)60835-5/fulltext"
    },
    ist3: {
        treatment: "IV alteplase (recombinant tissue plasminogen activator)",
        outcome: "Alive and independent (Oxford Handicap Score 0-2) at 6 months",
        population: "Acute ischemic stroke patients within 6 hours of symptom onset",
        description: "IST-3 was an international randomised open-label trial (N=3,035) testing whether IV alteplase (0.9 mg/kg) given within 6 hours of acute ischaemic stroke improved functional outcome at 6 months, enrolling patients across 156 hospitals in 12 countries, including those over 80 years of age.",
        link: "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(12)60768-5/fulltext"
    },
    sprint: {
        treatment: "Intensive blood pressure control (systolic BP target <120 mmHg)",
        outcome: "Composite of major cardiovascular events (MI, stroke, heart failure, cardiovascular death)",
        population: "Non-diabetic adults aged ≥50 with hypertension and increased cardiovascular risk",
        description: "SPRINT was a multicentre open-label randomised trial (N=9,361) comparing intensive systolic BP target (<120 mmHg) to standard target (<140 mmHg) in non-diabetic adults aged ≥50 with hypertension and at least one additional cardiovascular risk factor, conducted at 102 clinical sites in the United States.",
        link: "https://www.nejm.org/doi/full/10.1056/NEJMoa1511939"
    },
    accord: {
        treatment: "Intensive blood pressure control (systolic BP target <120 mmHg)",
        outcome: "Major cardiovascular events (nonfatal MI, nonfatal stroke, cardiovascular death)",
        population: "Adults with type 2 diabetes and high cardiovascular risk",
        description: "ACCORD-BP was a randomised trial (N=4,733) embedded within the ACCORD study, comparing intensive systolic BP target (<120 mmHg) to standard target (<140 mmHg) in adults with type 2 diabetes and high cardiovascular risk, conducted at 77 clinical sites across the United States and Canada.",
        link: "https://www.nejm.org/doi/full/10.1056/NEJMoa1001286"
    },
    accord_glycemia: {
        treatment: "Intensive glycemic control (HbA1c target <6.0%)",
        outcome: "First major cardiovascular event composite (nonfatal MI, nonfatal stroke, or cardiovascular death)",
        population: "Adults with type 2 diabetes at high cardiovascular risk",
        description: "ACCORD Glycemia was a randomized trial comparing intensive glucose lowering (HbA1c target <6.0%) versus standard control (target 7.0-7.9%) in adults with type 2 diabetes at high cardiovascular risk. The primary composite cardiovascular outcome was nonfatal MI, nonfatal stroke, or cardiovascular death; the intensive glycemia strategy was stopped early because of increased all-cause mortality.",
        link: "https://www.nejm.org/doi/full/10.1056/NEJMoa0802743"
    }
};

const specialtyToCohort = {
    'Emergency': 'crash_2',
    'Surgery': 'crash_2',
    'Neurology': 'ist3',
    'Endocrinology and Metabolism': 'sprint',
    'Cardiology': 'accord',
    'Internal Medicine': 'accord_glycemia',
};

function getCohortForSpecialty(specialty) {
    return specialtyToCohort[specialty] || '';
}

// Feature name mapping for clean display
const featureNameMap = {
    // IST-3 features
    'nihss': 'NIHSS Score',
    'age': 'Age',
    'weight': 'Weight',
    'glucose': 'Blood Glucose',
    'gcs_score_rand': 'GCS',
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
    'icc': 'Injury Classification Code',
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
    // Check direct mapping first
    if (featureNameMap[featureName]) return featureNameMap[featureName];

    // Strip parenthesized raw variable names, e.g. "Injury classification code (icc)" -> "Injury classification code"
    const stripped = featureName.replace(/\s*\([^)]*\)\s*$/, '').trim();
    if (featureNameMap[stripped]) return featureNameMap[stripped];

    // Return the cleaned name (without raw variable in parentheses)
    return stripped;
}

// Rating criteria: 4 robustness gates + novelty bonus
// Aligned with judge_evaluation.py gate logic
const ratingGates = [
    {
        id: 'is_biologically_coherent',
        label: 'Q1: Logical Coherence',
        description: 'Is the proposed mechanism logically coherent? Does it provide a plausible explanation (biological, pharmacological, physiological, or clinical) that mechanistically connects the feature to differential treatment effect? DISAGREE if only a statistical/epidemiological claim, circular reasoning, or logically inconsistent.'
    },
    {
        id: 'is_causally_plausible',
        label: 'Q2: Causal Plausibility',
        description: 'Is the proposed mechanism causally plausible? AGREE example: "Patients with renal impairment clear the drug more slowly, leading to higher effective exposure and greater benefit." DISAGREE example: "Older patients benefit more" when the real reason is simply that older patients have higher baseline event rates (absolute-risk amplification with constant relative risk reduction). Also DISAGREE for post-treatment variables, reverse causality, or trivial severity proxies.'
    },
    {
        id: 'is_clinically_actionable',
        label: 'Q3: Clinical Actionability',
        description: 'Is the explanation clinically actionable? Does it propose clear, operationalisable patient subgroups with distinct treatment recommendations usable in clinical practice?'
    },
    {
        id: 'is_literature_backed',
        label: 'Q4: External Evidence',
        description: 'Based on your knowledge, is this specific feature × treatment interaction supported by existing evidence? For example, has it been reported in published RCT subgroup analyses, meta-analyses, clinical guidelines, or well-known clinical observations? AGREE if you are aware of supporting evidence; DISAGREE if you have never encountered this interaction in the literature or clinical practice.'
    }
];

const noveltyBonus = {
    id: 'is_novel',
    label: 'Novelty Bonus',
    description: 'Does this explanation identify an underexplored mechanism or subgroup not already well-covered in existing clinical guidelines or major reviews?'
};

const API_BASE_URL = window.RATINGS_API_BASE_URL || 'http://localhost:8000';



let currentHypotheses = [];
let ratings = {};

// Load explanations when button is clicked
document.getElementById('load-btn').addEventListener('click', loadHypotheses);

async function loadHypotheses() {
    const methodLabel = 'alex';
    const expertise = document.getElementById('expertise-select').value;
    const specialty = document.getElementById('specialty-input').value.trim();
    const cohort = getCohortForSpecialty(specialty);
    const raterId = document.getElementById('rater-id-input').value.trim();
    const raterIdPattern = /^[a-zA-Z0-9_-]{3,64}$/;

    if (!cohort) {
        alert('Please select a specialty with a mapped trial cohort');
        return;
    }

    if (!expertise) {
        alert('Please select your clinical expertise level');
        return;
    }

    if (!specialty) {
        alert('Please select your specialty');
        return;
    }

    if (!raterId || !raterIdPattern.test(raterId)) {
        alert('Please enter a valid anonymous ID (3-64 chars; letters, numbers, _ or -)');
        return;
    }

    const filePath = `agent/${cohort}/gpt-5-mini/${ALEX_METHOD}/seed_0/hypotheses.json`;

    try {
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        // Normalize different JSON formats into unified hypothesis list
        const hypotheses = normalizeHypotheses(data, ALEX_METHOD);

        displayTrialInfo(cohort);
        displayHypotheses(hypotheses, cohort, methodLabel, expertise, specialty, raterId);

    } catch (error) {
        const container = document.getElementById('hypotheses-container');
        container.innerHTML = `
            <div class="error">
                <strong>Error loading explanations:</strong> ${error.message}<br>
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
    let hypotheses = (data.feature_hypotheses || []).map((h, i) => ({
        feature_name: h.feature_name,
        mechanisms: (h.mechanisms || []).map(m => ({ description: m.description })),
        importance_rank: h.importance_rank || i + 1
    }));

    // For ResearchAgent, split each mechanism into its own card, deduplicate, then sample 5
    if (method === 'researchagent') {
        const seen = new Set();
        const split = [];
        for (const h of hypotheses) {
            for (const m of h.mechanisms) {
                const desc = m.description.trim();
                if (!seen.has(desc)) {
                    seen.add(desc);
                    split.push({
                        feature_name: h.feature_name,
                        mechanisms: [{ description: desc }],
                        importance_rank: 0
                    });
                }
            }
        }
        // Shuffle and pick 5
        for (let i = split.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [split[i], split[j]] = [split[j], split[i]];
        }
        hypotheses = split.slice(0, 5);
        hypotheses.forEach((h, i) => h.importance_rank = i + 1);
    }

    return hypotheses;
}

function displayTrialInfo(cohort) {
    const info = trialInfo[cohort];
    document.getElementById('trial-treatment').textContent = info.treatment;
    document.getElementById('trial-outcome').textContent = info.outcome;
    document.getElementById('trial-population').textContent = info.population;
    document.getElementById('trial-description').textContent = info.description;
    const linkEl = document.getElementById('trial-link');
    linkEl.href = info.link;
    linkEl.textContent = 'View Publication';
    document.getElementById('trial-info').style.display = 'block';
}

function displayHypotheses(hypotheses, cohort, method, expertise, specialty, raterId) {
    currentHypotheses = hypotheses;
    ratings = {
        expertise: expertise,
        specialty: specialty,
        rater_id: raterId,
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

    card.innerHTML = `
        <div class="hypothesis-header">
            <div class="hypothesis-title">${getDisplayFeatureName(hypothesis.feature_name)}</div>
            <div>
                <span class="feature-badge">Rank: ${hypothesis.importance_rank || index + 1}</span>
            </div>
        </div>

        <div class="hypothesis-content">
            <div class="content-section">
                <h4>Explanations</h4>
                ${hypothesis.mechanisms.map(m => `
                    <div class="mechanism-item">
                        ${m.description}
                    </div>
                `).join('')}
            </div>
        </div>

        <div class="rating-section">
            <h4>Your Assessment</h4>
            <p class="gate-instructions">For each criterion, select AGREE or DISAGREE.</p>
            ${createGateInputs(index)}

            <div class="rating-group">
                <label class="rating-label">Additional comments (optional)</label>
                <textarea id="comments-${index}" placeholder="Any additional thoughts not captured above..."></textarea>
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
                    AGREE
                </button>
                <button type="button" class="gate-btn gate-btn-false" id="${gate.id}-${hypIndex}-false"
                    onclick="setGate('${gate.id}', ${hypIndex}, false)">
                    DISAGREE
                </button>
                <span class="gate-status" id="${gate.id}-${hypIndex}-status">Not rated</span>
            </div>
            ${['is_causally_plausible', 'is_clinically_actionable', 'is_literature_backed'].includes(gate.id) ? `
                <div class="rating-group gate-comment-group">
                    <label class="rating-label">Additional comments for ${gate.label} (optional)</label>
                    <textarea id="${gate.id}-comments-${hypIndex}" placeholder="Optional explanation for your ${gate.label} rating..."></textarea>
                </div>
            ` : ''}
        </div>
    `).join('');

    const noveltyHTML = `
        <div class="rating-group gate-group novelty-group">
            <label class="rating-label">${noveltyBonus.label}</label>
            <div class="rating-description">${noveltyBonus.description}</div>
            <div class="gate-toggle">
                <button type="button" class="gate-btn gate-btn-true" id="${noveltyBonus.id}-${hypIndex}-true"
                    onclick="setGate('${noveltyBonus.id}', ${hypIndex}, true)">
                    AGREE
                </button>
                <button type="button" class="gate-btn gate-btn-false" id="${noveltyBonus.id}-${hypIndex}-false"
                    onclick="setGate('${noveltyBonus.id}', ${hypIndex}, false)">
                    DISAGREE
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
        status.textContent = 'AGREE';
        status.className = 'gate-status gate-true';
    } else {
        falseBtn.classList.add('active');
        status.textContent = 'DISAGREE';
        status.className = 'gate-status gate-false';
    }

    // Store the value
    trueBtn.dataset.value = value ? 'true' : '';
    falseBtn.dataset.value = value ? '' : 'false';
}

// Submit ratings
document.getElementById('submit-btn').addEventListener('click', submitRatings);

function collectRatingsPayload() {
    const expertise = document.getElementById('expertise-select').value;
    const specialty = document.getElementById('specialty-input').value.trim();
    const cohort = getCohortForSpecialty(specialty);
    const raterId = document.getElementById('rater-id-input').value.trim();
    const raterIdPattern = /^[a-zA-Z0-9_-]{3,64}$/;

    if (!expertise) {
        alert('Please select your clinical expertise level');
        return null;
    }

    if (!specialty) {
        alert('Please select your specialty');
        return null;
    }

    if (!cohort) {
        alert('Selected specialty is not mapped to a trial cohort');
        return null;
    }

    if (!raterId || !raterIdPattern.test(raterId)) {
        alert('Please enter a valid anonymous ID (3-64 chars; letters, numbers, _ or -)');
        return null;
    }

    ratings.rater_id = raterId;
    ratings.specialty = specialty;
    ratings.cohort = cohort;

    const missingQuestions = [];

    // Collect all gate ratings
    ratings.ratings = currentHypotheses.map((hyp, index) => {
        const featureRating = {
            feature_name: hyp.feature_name,
            feature_index: index,
        };

        // Collect gate values
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
                    missingQuestions.push({
                        index,
                        gateId: gate.id,
                        label: gate.label,
                        featureName: getDisplayFeatureName(hyp.feature_name),
                    });
                }
            }

            if (['is_causally_plausible', 'is_clinically_actionable', 'is_literature_backed'].includes(gate.id)) {
                const gateCommentEl = document.getElementById(`${gate.id}-comments-${index}`);
                const gateComment = gateCommentEl ? gateCommentEl.value.trim() : '';
                if (gateComment) {
                    featureRating[`${gate.id}_comments`] = gateComment;
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
            missingQuestions.push({
                index,
                gateId: noveltyBonus.id,
                label: noveltyBonus.label,
                featureName: getDisplayFeatureName(hyp.feature_name),
            });
        }

        // Collect comments
        const comments = document.getElementById(`comments-${index}`).value.trim();
        if (comments) {
            featureRating.comments = comments;
        }

        return featureRating;
    });

    if (missingQuestions.length > 0) {
        const firstMissing = missingQuestions[0];
        const targetEl = document.getElementById(`${firstMissing.gateId}-${firstMissing.index}-status`);
        if (targetEl) {
            targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        alert(
            `Please answer all questions before submitting. ` +
            `First missing: ${firstMissing.label} for ${firstMissing.featureName}.`
        );
        return null;
    }

    return ratings;
}

async function submitRatings() {
    const payload = collectRatingsPayload();
    if (!payload) {
        return;
    }

    const submitBtn = document.getElementById('submit-btn');
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Submitting...';

    try {
        const response = await fetch(`${API_BASE_URL}/api/ratings`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const body = await response.text();
            throw new Error(`Submit failed (${response.status}): ${body}`);
        }

        const result = await response.json();
        alert(`Ratings submitted successfully. Submission ID: ${result.submission_id}`);
    } catch (error) {
        alert(
            `Could not submit ratings to backend at ${API_BASE_URL}. ` +
            `Please make sure the ratings server is running.\n\nError: ${error.message}`
        );
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
    }
}
