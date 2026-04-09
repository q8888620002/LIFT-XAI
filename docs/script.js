const ALEX_METHOD = 'with_shap_drlearner';

// Trial metadata
const trialInfo = {
    crash_2: {
        abstract: "Background: Tranexamic acid can reduce bleeding in patients undergoing elective surgery. We assessed the effects of early administration of a short course of tranexamic acid on death, vascular occlusive events, and the receipt of blood transfusion in trauma patients.\n\nMethods: This randomised controlled trial was undertaken in 274 hospitals in 40 countries. 20 211 adult trauma patients with, or at risk of, significant bleeding were randomly assigned within 8 h of injury to either tranexamic acid (loading dose 1 g over 10 min then infusion of 1 g over 8 h) or matching placebo. Randomisation was balanced by centre, with an allocation sequence based on a block size of eight, generated with a computer random number generator. Both participants and study staff (site investigators and trial coordinating centre staff) were masked to treatment allocation. The primary outcome was death in hospital within 4 weeks of injury, and was described with the following categories: bleeding, vascular occlusion (myocardial infarction, stroke and pulmonary embolism), multiorgan failure, head injury, and other. All analyses were by intention to treat. \n\nFindings: 10 096 patients were allocated to tranexamic acid and 10 115 to placebo, of whom 10 060 and 10 067, respectively, were analysed. All-cause mortality was significantly reduced with tranexamic acid (1463 [14.5%] tranexamic acid group vs 1613 [16.0%] placebo group; relative risk 0.91, 95% CI 0.85-0.97; p=0.0035). The risk of death due to bleeding was significantly reduced (489 [4.9%] vs 574 [5.7%]; relative risk 0.85, 95% CI 0.76-0.96; p=0.0077).",
        interpretation: "Early tranexamic acid appears to reduce all-cause mortality and death due to bleeding in trauma patients when given soon after injury.",
        treatment: "Tranexamic acid (TXA)",
        outcome: "All-cause mortality at 28 days or in-hospital death",
        population: "Trauma patients with significant bleeding or at risk of significant hemorrhage",
        description: "CRASH-2 was a large international randomised placebo-controlled trial (N=20,211) evaluating the effect of early administration of tranexamic acid on death, vascular occlusive events, and blood transfusion in adult trauma patients with or at risk of significant bleeding, conducted across 274 hospitals in 40 countries.",
        link: "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(10)60835-5/fulltext"
    },
    ist3: {
        background: "Thrombolysis is of net benefit in patients with acute ischaemic stroke, who are younger than 80 years of age and are treated within 4.5 h of onset. The third International Stroke Trial (IST-3) sought to determine whether a wider range of patients might benefit up to 6 h from stroke onset.",
        methods: "In this international, multicentre, randomised, open-treatment trial, patients were allocated to 0.9 mg/kg intravenous recombinant tissue plasminogen activator (rt-PA) or to control. The primary analysis was of the proportion of patients alive and independent, as defined by an Oxford Handicap Score (OHS) of 0-2 at 6 months. The study is registered, ISRCTN25765518.",
        findings: "3035 patients were enrolled by 156 hospitals in 12 countries. All of these patients were included in the analyses (1515 in the rt-PA group vs 1520 in the control group), of whom 1617 (53%) were older than 80 years of age. At 6 months, 554 (37%) patients in the rt-PA group versus 534 (35%) in the control group were alive and independent (OHS 0-2; adjusted odds ratio [OR] 1.13, 95% CI 0.95-1.35, p=0.181; a non-significant absolute increase of 14/1000, 95% CI -20 to 48). An ordinal analysis showed a significant shift in OHS scores; common OR 1.27 (95% CI 1.10-1.47, p=0.001). Fatal or non-fatal symptomatic intracranial haemorrhage within 7 days occurred in 104 (7%) patients in the rt-PA group versus 16 (1%) in the control group (adjusted OR 6.94, 95% CI 4.07-11.8; absolute excess 58/1000, 95% CI 44-72). More deaths occurred within 7 days in the rt-PA group (163 [11%]) than in the control group (107 [7%], adjusted OR 1.60, 95% CI 1.22-2.08, p=0.001; absolute increase 37/1000, 95% CI 17-57), but between 7 days and 6 months there were fewer deaths in the rt-PA group than in the control group, so that by 6 months, similar numbers, in total, had died (408 [27%] in the rt-PA group vs 407 [27%] in the control group).",
        interpretation: "For the types of patient recruited in IST-3, despite the early hazards, thrombolysis within 6 h improved functional outcome. Benefit did not seem to be diminished in elderly patients.",
        treatment: "IV alteplase (recombinant tissue plasminogen activator)",
        outcome: "Alive and independent (Oxford Handicap Score 0-2) at 6 months",
        population: "Acute ischemic stroke patients within 6 hours of symptom onset",
        description: "IST-3 was an international randomised open-label trial (N=3,035) testing whether IV alteplase (0.9 mg/kg) given within 6 hours of acute ischaemic stroke improved functional outcome at 6 months, enrolling patients across 156 hospitals in 12 countries, including those over 80 years of age.",
        link: "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(12)60768-5/fulltext"
    },
    sprint: {
        background: "The most appropriate targets for systolic blood pressure to reduce cardiovascular morbidity and mortality among persons without diabetes remain uncertain.",
        methods: "We randomly assigned 9361 persons with a systolic blood pressure of 130 mm Hg or higher and an increased cardiovascular risk, but without diabetes, to a systolic blood-pressure target of less than 120 mm Hg (intensive treatment) or a target of less than 140 mm Hg (standard treatment). The primary composite outcome was myocardial infarction, other acute coronary syndromes, stroke, heart failure, or death from cardiovascular causes.",
        findings: "At 1 year, the mean systolic blood pressure was 121.4 mm Hg in the intensive-treatment group and 136.2 mm Hg in the standard-treatment group. The intervention was stopped early after a median follow-up of 3.26 years owing to a significantly lower rate of the primary composite outcome in the intensive-treatment group than in the standard-treatment group (1.65% per year vs. 2.19% per year; hazard ratio with intensive treatment, 0.75; 95% confidence interval [CI], 0.64 to 0.89; P<0.001). All-cause mortality was also significantly lower in the intensive-treatment group (hazard ratio, 0.73; 95% CI, 0.60 to 0.90; P=0.003). Rates of serious adverse events of hypotension, syncope, electrolyte abnormalities, and acute kidney injury or failure, but not of injurious falls, were higher in the intensive-treatment group than in the standard-treatment group.",
        interpretation: "Among patients at high risk for cardiovascular events but without diabetes, targeting a systolic blood pressure of less than 120 mm Hg, as compared with less than 140 mm Hg, resulted in lower rates of fatal and nonfatal major cardiovascular events and death from any cause, although significantly higher rates of some adverse events were observed in the intensive-treatment group.",
        treatment: "Intensive blood pressure control (systolic BP target <120 mmHg)",
        outcome: "Composite of major cardiovascular events (MI, stroke, heart failure, cardiovascular death)",
        population: "Non-diabetic adults aged ≥50 with hypertension and increased cardiovascular risk",
        description: "SPRINT was a multicentre open-label randomised trial (N=9,361) comparing intensive systolic BP target (<120 mmHg) to standard target (<140 mmHg) in non-diabetic adults aged ≥50 with hypertension and at least one additional cardiovascular risk factor, conducted at 102 clinical sites in the United States.",
        link: "https://www.nejm.org/doi/full/10.1056/NEJMoa1511939"
    },
    accord: {
        background: "There is no evidence from randomized trials to support a strategy of lowering systolic blood pressure below 135 to 140 mm Hg in persons with type 2 diabetes mellitus. We investigated whether therapy targeting normal systolic pressure (i.e., <120 mm Hg) reduces major cardiovascular events in participants with type 2 diabetes at high risk for cardiovascular events.",
        methods: "A total of 4733 participants with type 2 diabetes were randomly assigned to intensive therapy, targeting a systolic pressure of less than 120 mm Hg, or standard therapy, targeting a systolic pressure of less than 140 mm Hg. The primary composite outcome was nonfatal myocardial infarction, nonfatal stroke, or death from cardiovascular causes. The mean follow-up was 4.7 years.",
        findings: "After 1 year, the mean systolic blood pressure was 119.3 mm Hg in the intensive-therapy group and 133.5 mm Hg in the standard-therapy group. The annual rate of the primary outcome was 1.87% in the intensive-therapy group and 2.09% in the standard-therapy group (hazard ratio with intensive therapy, 0.88; 95% confidence interval [CI], 0.73 to 1.06; P=0.20). The annual rates of death from any cause were 1.28% and 1.19% in the two groups, respectively (hazard ratio, 1.07; 95% CI, 0.85 to 1.35; P=0.55). The annual rates of stroke, a prespecified secondary outcome, were 0.32% and 0.53% in the two groups, respectively (hazard ratio, 0.59; 95% CI, 0.39 to 0.89; P=0.01). Serious adverse events attributed to antihypertensive treatment occurred in 77 of the 2362 participants in the intensive-therapy group (3.3%) and 30 of the 2371 participants in the standard-therapy group (1.3%) (P<0.001).",
        interpretation: "In patients with type 2 diabetes at high risk for cardiovascular events, targeting a systolic blood pressure of less than 120 mm Hg, as compared with less than 140 mm Hg, did not reduce the rate of a composite outcome of fatal and nonfatal major cardiovascular events.",
        treatment: "Intensive blood pressure control (systolic BP target <120 mmHg)",
        outcome: "Major cardiovascular events (nonfatal MI, nonfatal stroke, cardiovascular death)",
        population: "Adults with type 2 diabetes and high cardiovascular risk",
        description: "ACCORD-BP was a randomised trial (N=4,733) embedded within the ACCORD study, comparing intensive systolic BP target (<120 mmHg) to standard target (<140 mmHg) in adults with type 2 diabetes and high cardiovascular risk, conducted at 77 clinical sites across the United States and Canada.",
        link: "https://www.nejm.org/doi/full/10.1056/NEJMoa1001286"
    },
    accord_glycemia: {
        background: "Epidemiologic studies have shown a relationship between glycated hemoglobin levels and cardiovascular events in patients with type 2 diabetes. We investigated whether intensive therapy to target normal glycated hemoglobin levels would reduce cardiovascular events in patients with type 2 diabetes who had either established cardiovascular disease or additional cardiovascular risk factors.",
        methods: "In this randomized study, 10,251 patients (mean age, 62.2 years) with a median glycated hemoglobin level of 8.1% were assigned to receive intensive therapy (targeting a glycated hemoglobin level below 6.0%) or standard therapy (targeting a level from 7.0 to 7.9%). Of these patients, 38% were women, and 35% had had a previous cardiovascular event. The primary outcome was a composite of nonfatal myocardial infarction, nonfatal stroke, or death from cardiovascular causes. The finding of higher mortality in the intensive-therapy group led to a discontinuation of intensive therapy after a mean of 3.5 years of follow-up.",
        findings: "At 1 year, stable median glycated hemoglobin levels of 6.4% and 7.5% were achieved in the intensive-therapy group and the standard-therapy group, respectively. During follow-up, the primary outcome occurred in 352 patients in the intensive-therapy group, as compared with 371 in the standard-therapy group (hazard ratio, 0.90; 95% confidence interval [CI], 0.78 to 1.04; P=0.16). At the same time, 257 patients in the intensive-therapy group died, as compared with 203 patients in the standard-therapy group (hazard ratio, 1.22; 95% CI, 1.01 to 1.46; P=0.04). Hypoglycemia requiring assistance and weight gain of more than 10 kg were more frequent in the intensive-therapy group (P<0.001).",
        interpretation: "As compared with standard therapy, the use of intensive therapy to target normal glycated hemoglobin levels for 3.5 years increased mortality and did not significantly reduce major cardiovascular events. These findings identify a previously unrecognized harm of intensive glucose lowering in high-risk patients with type 2 diabetes.",
        treatment: "Intensive glycemic control (HbA1c target <6.0%)",
        outcome: "First major cardiovascular event composite (nonfatal MI, nonfatal stroke, or cardiovascular death)",
        population: "Adults with type 2 diabetes at high cardiovascular risk",
        description: "ACCORD Glycemia was a randomized trial comparing intensive glucose lowering (HbA1c target <6.0%) versus standard control (target 7.0-7.9%) in adults with type 2 diabetes at high cardiovascular risk. The primary composite cardiovascular outcome was nonfatal MI, nonfatal stroke, or cardiovascular death; the intensive glycemia strategy was stopped early because of increased all-cause mortality.",
        link: "https://www.nejm.org/doi/full/10.1056/NEJMoa0802743"
    }
};

const specialtyToCohort = {
    emergency: 'crash_2',
    surgery: 'crash_2',
    neurology: 'ist3',
    'endocrinology and metabolism': 'sprint',
    cardiology: 'accord',
    'internal medicine': 'accord_glycemia',
};

function normalizeSpecialty(specialty) {
    return (specialty || '').trim().toLowerCase().replace(/\s+/g, ' ');
}

function getCohortForSpecialty(specialty) {
    return specialtyToCohort[normalizeSpecialty(specialty)] || '';
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
        label: 'Q1: Is the proposed mechanism logically coherent?',
        description: 'Logic coherence: Does it provide a plausible explanation (biological, pharmacological, physiological, or clinical) that mechanistically connects the feature to differential treatment effect? DISAGREE if only a statistical/epidemiological claim, circular reasoning, or logically inconsistent.'
    },
    {
        id: 'is_causally_plausible',
        label: 'Q2: Is the proposed mechanism causally plausible?',
        description: 'Causal Plausibility: AGREE example: "Patients with renal impairment clear the drug more slowly, leading to higher effective exposure and greater benefit." DISAGREE example: "Older patients benefit more" when the real reason is simply that older patients have higher baseline event rates (absolute-risk amplification with constant relative risk reduction). Also DISAGREE for post-treatment variables, reverse causality, or trivial severity proxies.'
    },
    {
        id: 'is_clinically_actionable',
        label: 'Q3: Is the explanation clinically actionable?',
        description: 'Clinical Actionability: Does it propose clear, operationalisable patient subgroups with distinct treatment recommendations usable in clinical practice?'
    },
    {
        id: 'is_literature_backed',
        label: 'Q4: Is there any evidence base supporting this explanation?',
        description: 'External Evidence: Based on your knowledge, is this specific explanation supported by existing evidence? For example, has it been reported in published RCT subgroup analyses, meta-analyses, clinical guidelines, or well-known clinical observations? AGREE if you are aware of supporting evidence; DISAGREE if you have never encountered this interaction in the literature or clinical practice.'
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

function setLoadStatus(message, type = 'info') {
    const statusEl = document.getElementById('load-status');
    if (!statusEl) return;

    if (!message) {
        statusEl.style.display = 'none';
        statusEl.textContent = '';
        statusEl.style.color = '';
        return;
    }

    statusEl.style.display = 'block';
    statusEl.textContent = message;
    if (type === 'error') {
        statusEl.style.color = '#b00020';
    } else if (type === 'success') {
        statusEl.style.color = '#0a7a2f';
    } else {
        statusEl.style.color = '';
    }
}

// Load explanations when button is clicked
document.getElementById('load-btn').addEventListener('click', loadHypotheses);

async function loadHypotheses() {
    const methodLabel = 'alex';
    const expertise = document.getElementById('expertise-select').value;
    const specialty = document.getElementById('specialty-input').value;
    const raterId = document.getElementById('rater-id-input').value.trim();
    const raterIdPattern = /^[a-zA-Z0-9_-]{3,64}$/;

    if (!specialty) {
        setLoadStatus('Please select your specialty.', 'error');
        alert('Please select your specialty');
        return;
    }

    const cohort = getCohortForSpecialty(specialty);

    if (!cohort) {
        setLoadStatus('Please select a specialty mapped to a trial cohort.', 'error');
        alert('Please select a specialty with a mapped trial cohort');
        return;
    }

    if (!expertise) {
        setLoadStatus('Please select your clinical expertise level.', 'error');
        alert('Please select your clinical expertise level');
        return;
    }

    if (!raterId || !raterIdPattern.test(raterId)) {
        setLoadStatus('Please enter a valid anonymous ID (3-64 chars; letters, numbers, _ or -).', 'error');
        alert('Please enter a valid anonymous ID (3-64 chars; letters, numbers, _ or -)');
        return;
    }

    const filePath = `agent/${cohort}/gpt-5-mini/${ALEX_METHOD}/seed_0/hypotheses.json`;
    setLoadStatus('Loading explanations...');

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
        setLoadStatus(`Loaded ${hypotheses.length} explanations.`, 'success');

    } catch (error) {
        setLoadStatus(`Error loading explanations from ${filePath}: ${error.message}`, 'error');
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

    const trialDisplayNames = {
        crash_2: 'CRASH-2',
        ist3: 'IST-3',
        sprint: 'SPRINT',
        accord: 'ACCORD-BP',
        accord_glycemia: 'ACCORD Glycemia',
    };

    const extractSection = (abstractText, sectionName) => {
        if (!abstractText) return '';
        const regex = new RegExp(`${sectionName}:\\s*([\\s\\S]*?)(?=\\n\\n(?:Background|Methods|Findings|Interpretation):|$)`, 'i');
        const match = abstractText.match(regex);
        return match ? match[1].trim() : '';
    };

    const background = info.background
        || extractSection(info.abstract, 'Background')
        || info.description
        || '';
    const methods = info.methods
        || extractSection(info.abstract, 'Methods')
        || `Trial population: ${info.population}.`;
    const findings = info.findings
        || extractSection(info.abstract, 'Findings')
        || `Primary outcome: ${info.outcome}.`;
    const interpretation = info.interpretation
        || extractSection(info.abstract, 'Interpretation')
        || `Clinical interpretation: ${info.treatment} evaluated in ${cohort.replace('_', ' ').toUpperCase()} for outcome improvement.`;

    document.getElementById('trial-name').textContent = trialDisplayNames[cohort] || cohort;
    document.getElementById('trial-background').textContent = background;
    document.getElementById('trial-methods').textContent = methods;
    document.getElementById('trial-findings').textContent = findings;
    document.getElementById('trial-interpretation').textContent = interpretation;

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
    const specialty = document.getElementById('specialty-input').value;
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

    const cohort = getCohortForSpecialty(specialty);

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
