const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "US Housing Market Analysis and Prediction";
pres.author = "Ayoub Naouech & Rayen Belhadj";

// ── COLOR PALETTE (Ocean Deep / Data Science) ──
const C = {
  navy:    "07111F",
  navy2:   "0D1E33",
  navy3:   "132540",
  panel:   "0F1E30",
  panel2:  "162A40",
  teal:    "1BBFA9",
  teal2:   "0D9484",
  amber:   "F0A832",
  sky:     "4CB8E0",
  rose:    "E8637A",
  text:    "E8F1FB",
  muted:   "7FA6C8",
  dim:     "3D5A73",
  white:   "FFFFFF",
  light:   "EFF6FF",
  lightbg: "F8FAFC",
};

const makeShadow = () => ({ type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.18 });

const ASSETS = {
  appHome: path.join(__dirname, "final_housing_intelligence_latex_report", "app_screenshots", "app_home.png"),
  appModeling: path.join(__dirname, "final_housing_intelligence_latex_report", "app_screenshots", "app_modeling.png"),
  appEvaluation: path.join(__dirname, "final_housing_intelligence_latex_report", "app_screenshots", "app_evaluation.png"),
  appOlap: path.join(__dirname, "final_housing_intelligence_latex_report", "app_screenshots", "app_olap.png"),
  appPaperReview: path.join(__dirname, "final_housing_intelligence_latex_report", "app_screenshots", "app_paper_review.png"),
  appGovernance: path.join(__dirname, "final_housing_intelligence_latex_report", "app_screenshots", "app_governance.png"),
  corr: path.join(__dirname, "final_housing_intelligence_latex_report", "figures", "correlation_results.png"),
  eval: path.join(__dirname, "final_housing_intelligence_latex_report", "figures", "model_evaluation_results.png"),
  diag: path.join(__dirname, "final_housing_intelligence_latex_report", "figures", "fit_diagnostics_results.png"),
  pred: path.join(__dirname, "final_housing_intelligence_latex_report", "figures", "actual_vs_predicted_results.png"),
  olap: path.join(__dirname, "final_housing_intelligence_latex_report", "figures", "olap_period_results.png"),
};

function visualFrame(slide, imgPath, x, y, w, h) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: C.white }, line: { color: "CBD5E1", width: 1 }, shadow: makeShadow()
  });
  slide.addImage({ path: imgPath, x: x + 0.06, y: y + 0.06, w: w - 0.12, h: h - 0.12 });
}

// ── HELPER: dark slide background ──
function darkBg(slide) {
  slide.background = { color: C.navy };
}

// ── HELPER: light slide background ──
function lightBg(slide) {
  slide.background = { color: C.lightbg };
}

// ── HELPER: section label chip ──
function sectionChip(slide, label, x, y) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w: 2.2, h: 0.28, fill: { color: C.teal, transparency: 75 }, line: { color: C.teal, width: 1 }, rectRadius: 0.05
  });
  slide.addText(label, { x, y, w: 2.2, h: 0.28, fontSize: 8, color: C.teal, bold: true, align: "center", valign: "middle", margin: 0 });
}

// ── HELPER: stat card (dark bg) ──
function statCard(slide, x, y, w, h, val, lbl, color) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: C.panel2 }, line: { color: C.dim, width: 0.75 }, shadow: makeShadow()
  });
  slide.addText(val, { x, y: y + 0.05, w, h: h * 0.58, fontSize: 28, color: color || C.teal, bold: true, align: "center", valign: "bottom", fontFace: "Calibri", margin: 0 });
  slide.addText(lbl, { x, y: y + h * 0.6, w, h: h * 0.35, fontSize: 9, color: C.muted, align: "center", valign: "top", margin: 0 });
}

// ── HELPER: info card (light bg) ──
function infoCard(slide, x, y, w, h, title, body, accentColor) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: C.white }, line: { color: "E2E8F0", width: 1 }, shadow: makeShadow()
  });
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h, fill: { color: accentColor || C.teal }, line: { style: "none" } });
  slide.addText(title, { x: x + 0.14, y: y + 0.08, w: w - 0.2, h: 0.25, fontSize: 11, color: "1E293B", bold: true, margin: 0 });
  slide.addText(body, { x: x + 0.14, y: y + 0.33, w: w - 0.2, h: h - 0.45, fontSize: 9.5, color: "475569", lineSpacingMultiple: 1.25, margin: 0 });
}

// ── HELPER: dark panel ──
function darkPanel(slide, x, y, w, h) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: C.panel2 }, line: { color: C.dim, width: 0.75 }, shadow: makeShadow()
  });
}

// ══════════════════════════════════════════
// SLIDE 1 – TITLE
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);

  // Accent gradient strips
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 1.1, fill: { color: C.navy3 }, line: { style: "none" } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.5, w: 10, h: 1.125, fill: { color: C.navy3 }, line: { style: "none" } });

  // Teal accent block left
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.1, w: 0.18, h: 3.4, fill: { color: C.teal }, line: { style: "none" } });

  // Institution badges
  s.addText("IT Business School · Tunisia", { x: 0.35, y: 0.18, w: 5, h: 0.28, fontSize: 9, color: C.muted, bold: false, margin: 0 });
  s.addText("Data Mining & Machine Learning  ·  2025–2026", { x: 0.35, y: 0.46, w: 7, h: 0.28, fontSize: 9, color: C.muted, margin: 0 });
  s.addText("Supervisor: Kaouether Ben Ali", { x: 0.35, y: 0.72, w: 5, h: 0.24, fontSize: 8.5, color: C.dim, italic: true, margin: 0 });

  // Main title
  s.addText("US Housing Market", { x: 0.35, y: 1.2, w: 9.3, h: 0.85, fontSize: 44, color: C.text, bold: true, fontFace: "Calibri", margin: 0 });
  s.addText("Analysis and Prediction", { x: 0.35, y: 2.0, w: 9.3, h: 0.72, fontSize: 44, color: C.teal, bold: true, fontFace: "Calibri", margin: 0 });
  s.addText("Using Data Mining and Machine Learning", { x: 0.35, y: 2.72, w: 9.3, h: 0.42, fontSize: 18, color: C.muted, margin: 0 });

  // Authors
  s.addText("Ayoub Naouech  ·  Rayen Belhadj", { x: 0.35, y: 3.25, w: 9.3, h: 0.3, fontSize: 13, color: C.text, bold: true, margin: 0 });

  // KPI row at bottom
  const kpis = [["242", "Observations"], ["27", "Features"], ["20yr", "Timespan"], ["0.999", "Best R²"]];
  kpis.forEach(([val, lbl], i) => {
    statCard(s, 0.4 + i * 2.3, 4.55, 2.0, 0.85, val, lbl, C.amber);
  });

  // Method chips bottom-right
  const chips = ["CRISP-DM", "LinearReg", "Ridge", "Diagnostics", "OLAP", "Streamlit", "Paper Review"];
  chips.forEach((c, i) => {
    const row = Math.floor(i / 4);
    const col = i % 4;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4 + col * 2.35, y: 4.55, w: 2.0, h: 0.85, fill: { color: C.panel2 }, line: { style: "none" } });
  });
}

// ══════════════════════════════════════════
// SLIDE 2 – AGENDA
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  lightBg(s);
  sectionChip(s, "AGENDA", 0.4, 0.25);

  s.addText("Presentation Outline", { x: 0.4, y: 0.6, w: 9.2, h: 0.5, fontSize: 28, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });

  const sections = [
    { num: "01", title: "Project Motivation & Context", desc: "Why housing markets matter" },
    { num: "02", title: "Dataset & Features", desc: "20 years, 242 observations, 27 variables" },
    { num: "03", title: "CRISP-DM Methodology", desc: "Full 6-phase data science workflow" },
    { num: "04", title: "Data Preparation", desc: "Cleaning, engineering, and feature design" },
    { num: "05", title: "Supervised Learning", desc: "Linear Regression best; Ridge baseline; tree diagnostics" },
    { num: "06", title: "Fit Diagnostics", desc: "Train-test gaps and chronological generalization" },
    { num: "07", title: "Reinforcement Learning", desc: "Q-Learning market-policy simulation" },
    { num: "08", title: "OLAP & Forecasting", desc: "Period intelligence and future scenario prediction" },
    { num: "09", title: "Dashboard", desc: "22-page Streamlit Intelligence Platform" },
    { num: "10", title: "Key Findings & Conclusion", desc: "Corrected metrics, evidence, and implications" },
  ];

  sections.forEach((sec, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.4 + col * 4.9;
    const y = 1.25 + row * 0.83;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.6, h: 0.7, fill: { color: col === 0 ? "EFF6FF" : "F0FDF4" }, line: { color: "E2E8F0", width: 1 }
    });
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h: 0.7, fill: { color: col === 0 ? C.teal : C.amber }, line: { style: "none" } });
    s.addText(sec.num, { x: x + 0.14, y, w: 0.5, h: 0.7, fontSize: 12, color: col === 0 ? C.teal : C.amber, bold: true, valign: "middle", margin: 0 });
    s.addText(sec.title, { x: x + 0.65, y: y + 0.06, w: 3.8, h: 0.28, fontSize: 11, color: "1E293B", bold: true, margin: 0 });
    s.addText(sec.desc, { x: x + 0.65, y: y + 0.36, w: 3.8, h: 0.25, fontSize: 9, color: "64748B", margin: 0 });
  });
}

// ══════════════════════════════════════════
// SLIDE 3 – PROJECT MOTIVATION
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionChip(s, "MOTIVATION", 0.4, 0.18);

  s.addText("Why Housing Markets?", { x: 0.4, y: 0.52, w: 9.2, h: 0.52, fontSize: 30, color: C.text, bold: true, fontFace: "Calibri", margin: 0 });

  // Large quote block
  darkPanel(s, 0.4, 1.15, 9.2, 1.15);
  s.addText('"The U.S. housing market is a cornerstone of national economic health, sensitive to inflation, unemployment, mortgage conditions, and interest-rate policy — yet no single model captures its full complexity."', {
    x: 0.6, y: 1.22, w: 8.8, h: 1.0, fontSize: 11.5, color: C.text, italic: true, lineSpacingMultiple: 1.35, margin: 0
  });

  const motivations = [
    { icon: "📈", title: "Market Complexity", body: "Non-linear interactions between macro variables across 20+ years of boom-bust cycles cannot be captured by simple econometric models." },
    { icon: "🏠", title: "Stakeholder Need", body: "Analysts, institutions, policymakers, and researchers need accessible, data-driven tools to understand price dynamics." },
    { icon: "🔬", title: "Research Gap", body: "Most studies examine one or two variables; this project combines macro, behavioral, and policy indicators simultaneously." },
    { icon: "🤖", title: "ML Opportunity", body: "Modern ML methods (ensembles, clustering, RL) can outperform traditional OLS regression on complex, non-stationary time series." },
  ];

  motivations.forEach((m, i) => {
    const x = 0.4 + (i % 2) * 4.8;
    const y = 2.45 + Math.floor(i / 2) * 1.45;
    darkPanel(s, x, y, 4.6, 1.3);
    s.addText(m.icon, { x: x + 0.1, y: y + 0.08, w: 0.6, h: 0.5, fontSize: 22, align: "center", margin: 0 });
    s.addText(m.title, { x: x + 0.75, y: y + 0.08, w: 3.7, h: 0.28, fontSize: 12, color: C.teal, bold: true, margin: 0 });
    s.addText(m.body, { x: x + 0.75, y: y + 0.38, w: 3.7, h: 0.82, fontSize: 9.5, color: C.muted, lineSpacingMultiple: 1.3, margin: 0 });
  });
}

// ══════════════════════════════════════════
// SLIDE 4 – DATASET OVERVIEW
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  lightBg(s);
  sectionChip(s, "DATASET", 0.4, 0.18);

  s.addText("Dataset Overview", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });

  // Big stats
  const dstats = [
    { val: "2004–2024", lbl: "Time Span", color: C.teal },
    { val: "242", lbl: "Monthly Rows", color: C.amber },
    { val: "27", lbl: "Feature Columns", color: C.sky },
    { val: "6", lbl: "Engineered Features", color: C.rose },
  ];
  dstats.forEach((d, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4 + i * 2.3, y: 1.1, w: 2.15, h: 1.1, fill: { color: C.white }, line: { color: "E2E8F0", width: 1 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4 + i * 2.3, y: 1.1, w: 2.15, h: 0.06, fill: { color: d.color }, line: { style: "none" } });
    s.addText(d.val, { x: 0.4 + i * 2.3, y: 1.22, w: 2.15, h: 0.55, fontSize: 22, color: d.color, bold: true, align: "center", fontFace: "Calibri", margin: 0 });
    s.addText(d.lbl, { x: 0.4 + i * 2.3, y: 1.77, w: 2.15, h: 0.28, fontSize: 9.5, color: "64748B", align: "center", margin: 0 });
  });

  // Variable categories
  const cats = [
    { title: "Housing Indicators", vars: ["Home Price Index (HPI)", "New Housing Permits", "Existing Home Sales", "Housing Starts"], color: C.teal },
    { title: "Macro / Economy", vars: ["Real GDP", "Median Household Income", "CPI", "Federal Funds Rate"], color: C.amber },
    { title: "Mortgage & Credit", vars: ["Mortgage Rate (30yr)", "Income-to-Mortgage Ratio", "CPI-to-Interest Ratio", "Consumer Credit"], color: C.sky },
    { title: "Labor & Sentiment", vars: ["Unemployment Rate", "Consumer Sentiment (UMICH)", "3-mo Rolling Unemployment", "Participation Rate"], color: C.rose },
  ];

  cats.forEach((cat, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.4 + col * 4.8, y = 2.35 + row * 1.55;
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 4.6, h: 1.42, fill: { color: C.white }, line: { color: "E2E8F0", width: 1 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h: 1.42, fill: { color: cat.color }, line: { style: "none" } });
    s.addText(cat.title, { x: x + 0.14, y: y + 0.07, w: 4.3, h: 0.26, fontSize: 11, color: "1E293B", bold: true, margin: 0 });
    s.addText(cat.vars.join("\n"), { x: x + 0.14, y: y + 0.36, w: 4.3, h: 0.98, fontSize: 9, color: "475569", lineSpacingMultiple: 1.3, margin: 0 });
  });

  s.addText("Primary sources: FRED, NAR, U.S. Census Bureau, Freddie Mac, University of Michigan", {
    x: 0.4, y: 5.35, w: 9.2, h: 0.2, fontSize: 8, color: "94A3B8", italic: true, margin: 0
  });
}

// ══════════════════════════════════════════
// SLIDE 5 – CRISP-DM METHODOLOGY
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionChip(s, "METHODOLOGY", 0.4, 0.18);

  s.addText("CRISP-DM Workflow", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 30, color: C.text, bold: true, fontFace: "Calibri", margin: 0 });

  const phases = [
    { num: "01", title: "Business Understanding", desc: "Define housing prediction objective, identify stakeholder needs, set success criteria" },
    { num: "02", title: "Data Understanding", desc: "Explore 27 features, correlation profiling, distribution analysis, quality checks" },
    { num: "03", title: "Data Preparation", desc: "Lag features, rolling means, imputation, period labeling, affordability ratios" },
    { num: "04", title: "Modeling", desc: "Ridge, RF, GB, SVR, KNN, MLP for regression & classification; KMeans, PCA, RL" },
    { num: "05", title: "Evaluation", desc: "MAE · RMSE · R² · Accuracy · F1 · AUC · Feature importance · Theory checks" },
    { num: "06", title: "Deployment", desc: "Streamlit dashboard: 22 pages, OLAP cube, chatbot, governance, export" },
  ];

  phases.forEach((p, i) => {
    const x = 0.4 + (i % 3) * 3.1;
    const y = 1.2 + Math.floor(i / 3) * 2.05;
    darkPanel(s, x, y, 2.95, 1.88);

    // Number badge
    s.addShape(pres.shapes.OVAL, { x: x + 0.12, y: y + 0.12, w: 0.44, h: 0.44, fill: { color: C.teal }, line: { style: "none" } });
    s.addText(p.num, { x: x + 0.12, y: y + 0.12, w: 0.44, h: 0.44, fontSize: 10, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });

    s.addText(p.title, { x: x + 0.65, y: y + 0.13, w: 2.2, h: 0.42, fontSize: 11, color: C.text, bold: true, lineSpacingMultiple: 1.15, margin: 0 });
    s.addText(p.desc, { x: x + 0.1, y: y + 0.65, w: 2.75, h: 1.12, fontSize: 9.5, color: C.muted, lineSpacingMultiple: 1.3, margin: 0 });
  });
}

// ══════════════════════════════════════════
// SLIDE 6 – DATA PREPARATION
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  lightBg(s);
  sectionChip(s, "DATA PREPARATION", 0.4, 0.18);

  s.addText("Feature Engineering & Cleaning", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });

  // Left: engineered features
  s.addText("Engineered Features", { x: 0.4, y: 1.1, w: 4.5, h: 0.32, fontSize: 14, color: "1E293B", bold: true, margin: 0 });

  const engFeatures = [
    { name: "HPI_lag1 / HPI_lag3", desc: "Target autoregression — captures housing momentum", color: C.teal },
    { name: "Unemployment_roll3", desc: "3-month rolling mean — smooths seasonal noise", color: C.amber },
    { name: "Income_to_Mortgage", desc: "Affordability ratio — strongest non-leaky predictor", color: C.sky },
    { name: "CPI_to_Interest", desc: "Macro pressure index — inflation vs. rate dynamics", color: C.rose },
    { name: "HPI_pct_change", desc: "Momentum signal — rate of price acceleration", color: C.teal2 },
    { name: "Administration", desc: "Political-period label — groups by presidency", color: C.dim },
  ];

  engFeatures.forEach((f, i) => {
    const y = 1.5 + i * 0.64;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y, w: 4.5, h: 0.56, fill: { color: C.white }, line: { color: "E2E8F0", width: 1 } });
    s.addShape(pres.shapes.OVAL, { x: 0.52, y: y + 0.17, w: 0.22, h: 0.22, fill: { color: f.color }, line: { style: "none" } });
    s.addText(f.name, { x: 0.85, y: y + 0.03, w: 3.95, h: 0.26, fontSize: 10.5, color: "1E293B", bold: true, margin: 0 });
    s.addText(f.desc, { x: 0.85, y: y + 0.3, w: 3.95, h: 0.22, fontSize: 9, color: "64748B", margin: 0 });
  });

  // Right: top correlations
  s.addText("Top Correlations with HPI", { x: 5.2, y: 1.1, w: 4.5, h: 0.32, fontSize: 14, color: "1E293B", bold: true, margin: 0 });

  const corrs = [
    { name: "Income-to-Mortgage", val: 0.93, color: C.teal },
    { name: "HPI Smoothed", val: 1.00, color: C.teal },
    { name: "HPI lag1", val: 1.00, color: C.teal2 },
    { name: "HPI lag3", val: 1.00, color: C.sky },
    { name: "Median Income", val: 0.88, color: C.amber },
    { name: "Mortgage Rate", val: 0.47, color: C.rose },
  ];

  corrs.forEach((c, i) => {
    const y = 1.52 + i * 0.64;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y, w: 4.5, h: 0.56, fill: { color: C.white }, line: { color: "E2E8F0", width: 1 } });
    s.addText(c.name, { x: 5.3, y: y + 0.04, w: 2.8, h: 0.24, fontSize: 10, color: "1E293B", bold: true, margin: 0 });
    // bar track
    s.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: y + 0.32, w: 3.0, h: 0.14, fill: { color: "F1F5F9" }, line: { style: "none" } });
    const barW = Math.abs(c.val) * 3.0;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: y + 0.32, w: barW, h: 0.14, fill: { color: c.color }, line: { style: "none" } });
    s.addText((c.val > 0 ? "+" : "") + c.val.toFixed(2), { x: 8.35, y: y + 0.04, w: 1.25, h: 0.46, fontSize: 11, color: c.val > 0 ? C.teal : C.rose, bold: true, align: "right", valign: "middle", margin: 0 });
  });
}

// ══════════════════════════════════════════
// SLIDE 7 – SUPERVISED LEARNING OVERVIEW
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionChip(s, "SUPERVISED LEARNING", 0.4, 0.18);

  s.addText("Supervised Learning Models", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 30, color: C.text, bold: true, fontFace: "Calibri", margin: 0 });

  const models = [
    { name: "Linear Regression", type: "Linear / Best test model", desc: "Best chronological test result because lagged and smoothed HPI preserve the market trend.", metrics: "R² 0.999 | MAE 0.661 | RMSE 0.797", color: C.teal, star: true },
    { name: "Ridge Regression", type: "Regularized baseline", desc: "Stable and interpretable. Useful as a transparent benchmark under multicollinearity.", metrics: "R² 0.984 | MAE 3.715 | RMSE 4.162", color: C.amber, star: false },
    { name: "Gradient Boosting", type: "Tree ensemble", desc: "Flexible model, but weak on the final chronological extrapolation period.", metrics: "R² -4.148 | MAE 65.816 | RMSE 73.669", color: C.sky, star: false },
    { name: "Decision Tree", type: "Tree / Regression", desc: "Easy to explain but performs poorly outside the training range.", metrics: "R² -4.159 | MAE 66.218 | RMSE 73.750", color: C.dim, star: false },
    { name: "Random Forest", type: "Tree ensemble", desc: "Strong in many contexts, but here struggles to extrapolate the high-price test period.", metrics: "R² -4.494 | MAE 69.077 | RMSE 76.105", color: C.rose, star: false },
    { name: "K-Nearest Neighbors", type: "Instance-based", desc: "Nearest-neighbor logic is weak for later observations outside earlier ranges.", metrics: "R² -5.053 | MAE 74.459 | RMSE 79.883", color: C.dim, star: false },
    { name: "Support Vector Regressor", type: "Kernel / Non-linear", desc: "Sensitive to scaling and parameters; weakest generated test result.", metrics: "R² -10.395 | MAE 103.767 | RMSE 109.606", color: C.dim, star: false },
    { name: "Diagnostic lesson", type: "Model risk", desc: "Time-aware diagnostics matter more than choosing a popular model family.", metrics: "Use train/test gap + charts", color: C.teal2, star: false },
  ];

  models.forEach((m, i) => {
    const col = i % 4, row = Math.floor(i / 4);
    const x = 0.4 + col * 2.35, y = 1.12 + row * 2.15;
    darkPanel(s, x, y, 2.2, 2.0);

    if (m.star) {
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: 2.2, h: 0.04, fill: { color: C.teal }, line: { style: "none" } });
      s.addText("★ BEST", { x: x + 1.5, y: y + 0.06, w: 0.6, h: 0.2, fontSize: 7, color: C.teal, bold: true, margin: 0 });
    }

    s.addText(m.name, { x: x + 0.1, y: y + 0.1, w: 2.0, h: 0.35, fontSize: 10.5, color: m.color, bold: true, lineSpacingMultiple: 1.1, margin: 0 });
    s.addText(m.type, { x: x + 0.1, y: y + 0.46, w: 2.0, h: 0.2, fontSize: 8, color: C.dim, italic: true, margin: 0 });
    s.addText(m.desc, { x: x + 0.1, y: y + 0.68, w: 2.0, h: 0.85, fontSize: 8.5, color: C.muted, lineSpacingMultiple: 1.25, margin: 0 });
    s.addShape(pres.shapes.RECTANGLE, { x: x + 0.1, y: y + 1.56, w: 2.0, h: 0.32, fill: { color: C.navy }, line: { style: "none" } });
    s.addText(m.metrics, { x: x + 0.1, y: y + 1.56, w: 2.0, h: 0.32, fontSize: 8, color: m.star ? C.teal : C.muted, align: "center", valign: "middle", bold: m.star, margin: 0 });
  });
}

// ══════════════════════════════════════════
// SLIDE 8 – MODEL PERFORMANCE CHART
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  lightBg(s);
  sectionChip(s, "MODEL COMPARISON", 0.4, 0.18);

  s.addText("Generated Regression Evaluation", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });

  const chartData = [{
    name: "R² Score",
    labels: ["LinearReg.", "Ridge", "GradBoost", "DecisionTree", "RandomForest", "KNN", "SVR"],
    values: [0.999, 0.984, -4.148, -4.159, -4.494, -5.053, -10.395]
  }];

  s.addChart(pres.charts.BAR, chartData, {
    x: 0.4, y: 1.1, w: 4.9, h: 4.2,
    barDir: "col",
    chartColors: ["1BBFA9", "F0A832", "94A3B8", "94A3B8", "E8637A", "94A3B8", "94A3B8"],
    chartArea: { fill: { color: "FFFFFF" }, roundedCorners: false },
    catAxisLabelColor: "475569",
    valAxisLabelColor: "475569",
    valAxisMinVal: -11,
    valAxisMaxVal: 1.2,
    valGridLine: { color: "E2E8F0", size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelColor: "1E293B",
    dataLabelFontSize: 9,
    showLegend: false,
  });

  visualFrame(s, ASSETS.eval, 5.45, 1.1, 4.15, 2.0);

  // Annotation panel
  darkPanel(s, 5.45, 3.25, 4.15, 2.05);
  s.addText("Winner: Linear Regression", { x: 5.6, y: 3.38, w: 3.85, h: 0.32, fontSize: 12, color: C.teal, bold: true, margin: 0 });
  s.addText("R² = 0.999", { x: 5.6, y: 3.68, w: 3.85, h: 0.38, fontSize: 22, color: C.amber, bold: true, fontFace: "Calibri", margin: 0 });

  const notes = [
    { label: "MAE", val: "0.661" },
    { label: "RMSE", val: "0.797" },
    { label: "CV R²", val: "0.994" },
    { label: "Split", val: "80/20" },
  ];
  notes.forEach((n, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 5.6, y: 4.14 + i * 0.28, w: 3.85, h: 0.23, fill: { color: C.navy }, line: { style: "none" } });
    s.addText(n.label, { x: 5.65, y: 4.14 + i * 0.28, w: 1.55, h: 0.23, fontSize: 8, color: C.muted, valign: "middle", align: "center", margin: 0 });
    s.addText(n.val, { x: 7.25, y: 4.14 + i * 0.28, w: 2.1, h: 0.23, fontSize: 8.5, color: C.text, bold: true, valign: "middle", align: "center", margin: 0 });
  });

  s.addText("Lagged and smoothed HPI features make a transparent linear model generalize best; tree models struggle to extrapolate the final high-price period.", {
    x: 5.6, y: 5.0, w: 3.85, h: 0.28, fontSize: 8.5, color: C.muted, lineSpacingMultiple: 1.2, margin: 0
  });
}

// ══════════════════════════════════════════
// SLIDE 9 – FEATURE IMPORTANCE
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionChip(s, "EXPLAINABILITY", 0.4, 0.18);

  s.addText("Correlation & Feature Signal", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: C.text, bold: true, fontFace: "Calibri", margin: 0 });

  const features = [
    { name: "Home_Price_Index_Smoothed", pct: 100, color: C.teal },
    { name: "HPI_lag1", pct: 100, color: C.teal2 },
    { name: "HPI_lag3", pct: 100, color: C.sky },
    { name: "Median_Income", pct: 88, color: C.amber },
    { name: "US_Population", pct: 86, color: C.rose },
    { name: "Income_to_Mortgage_Ratio", pct: 82, color: C.muted },
    { name: "Mortgage_Rate", pct: 47, color: C.dim },
  ];

  features.forEach((f, i) => {
    const y = 1.2 + i * 0.57;
    s.addText(f.name, { x: 0.4, y, w: 2.8, h: 0.48, fontSize: 10.5, color: C.text, bold: true, valign: "middle", margin: 0 });
    const maxW = 5.5;
    const barW = (f.pct / 100) * maxW;
    s.addShape(pres.shapes.RECTANGLE, { x: 3.3, y: y + 0.1, w: maxW, h: 0.28, fill: { color: C.panel2 }, line: { style: "none" } });
    s.addShape(pres.shapes.RECTANGLE, { x: 3.3, y: y + 0.1, w: barW, h: 0.28, fill: { color: f.color }, line: { style: "none" } });
    s.addText(f.pct + "%", { x: 8.9, y, w: 0.8, h: 0.48, fontSize: 11, color: f.color, bold: true, valign: "middle", align: "right", margin: 0 });
  });

  // Insight boxes
  darkPanel(s, 0.4, 5.1, 4.4, 0.38);
  s.addText("Lagged and smoothed HPI confirm strong housing-price persistence", { x: 0.5, y: 5.1, w: 4.2, h: 0.38, fontSize: 9, color: C.teal, valign: "middle", margin: 0 });
  darkPanel(s, 5.2, 5.1, 4.4, 0.38);
  s.addText("Median income and affordability ratios support the paper-review story", { x: 5.3, y: 5.1, w: 4.2, h: 0.38, fontSize: 9, color: C.amber, valign: "middle", margin: 0 });
}

// ══════════════════════════════════════════
// SLIDE 10 – CLASSIFICATION RESULTS
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  lightBg(s);
  sectionChip(s, "APP EVALUATION", 0.4, 0.18);

  s.addText("Evaluation Page + Corrected Metrics", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });

  s.addText("The final presentation uses the generated regression evidence from the included CSV: chronological 80/20 split, real model metrics, and diagnostic charts.", {
    x: 0.4, y: 1.05, w: 9.2, h: 0.38, fontSize: 10, color: "475569", margin: 0
  });

  const classData = [{
    name: "R² Score",
    labels: ["LinearReg.", "Ridge", "GradBoost", "DecisionTree", "RandomForest"],
    values: [0.999, 0.984, -4.148, -4.159, -4.494]
  }];

  s.addChart(pres.charts.BAR, classData, {
    x: 0.4, y: 1.55, w: 5.5, h: 3.8,
    barDir: "bar",
    chartColors: ["1BBFA9", "14B8A6", "94A3B8", "94A3B8", "94A3B8"],
    chartArea: { fill: { color: "FFFFFF" } },
    catAxisLabelColor: "475569",
    valAxisLabelColor: "475569",
    valGridLine: { color: "E2E8F0" },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelColor: "1E293B",
    showLegend: false,
  });

  const metrics = [
    { label: "Best R²", val: "0.999", desc: "Linear Regression" },
    { label: "Best MAE", val: "0.661", desc: "Linear Regression" },
    { label: "Best RMSE", val: "0.797", desc: "Linear Regression" },
    { label: "Best CV R²", val: "0.994", desc: "Linear Regression" },
  ];
  metrics.forEach((m, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 6.2, y: 1.55 + i * 0.98, w: 3.4, h: 0.85, fill: { color: C.white }, line: { color: "E2E8F0", width: 1 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: 6.2, y: 1.55 + i * 0.98, w: 0.06, h: 0.85, fill: { color: C.teal }, line: { style: "none" } });
    s.addText(m.val, { x: 6.35, y: 1.6 + i * 0.98, w: 3.1, h: 0.38, fontSize: 22, color: C.teal, bold: true, fontFace: "Calibri", margin: 0 });
    s.addText(m.label + " · " + m.desc, { x: 6.35, y: 1.98 + i * 0.98, w: 3.1, h: 0.28, fontSize: 9, color: "64748B", margin: 0 });
  });

  s.addText("Key takeaway: lag and smoothing features make Linear Regression strongest here; tree models are useful diagnostics but weak at extrapolating the final high-price period.", {
    x: 0.4, y: 5.38, w: 9.2, h: 0.22, fontSize: 9, color: "94A3B8", italic: true, margin: 0
  });
}

// ══════════════════════════════════════════
// SLIDE 11 – UNSUPERVISED LEARNING
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionChip(s, "UNSUPERVISED LEARNING", 0.4, 0.18);

  s.addText("Unsupervised Learning Results", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: C.text, bold: true, fontFace: "Calibri", margin: 0 });

  const methods = [
    {
      tag: "K-MEANS", color: C.teal,
      title: "Market Regime Clustering",
      result: "3 distinct market regimes identified",
      details: ["High-rate stress periods (2022–24)", "Low-rate expansion cycles (2013–19)", "Crisis/transition windows (2008–12, 2020)"],
      metric: "Silhouette: 0.42"
    },
    {
      tag: "PCA", color: C.sky,
      title: "Dimensionality Reduction",
      result: "~70% variance in first 2 PCs",
      details: ["PC1 ~ affordability & income pressure axis", "PC2 ~ rate shock & volatility axis", "Macro behavior is compressible into 2 key signals"],
      metric: "PC1 explains 48%"
    },
    {
      tag: "DBSCAN", color: C.amber,
      title: "Density-Based Segmentation",
      result: "3 dense groups + noise periods",
      details: ["Robust without pre-specifying k", "Noise = 2008, 2020, 2022 spikes", "Identifies natural market boundaries"],
      metric: "12 noise outliers"
    },
    {
      tag: "ISOLATION FOREST", color: C.rose,
      title: "Anomaly Detection",
      result: "14 anomalous months flagged",
      details: ["Flags 2008–09 financial crisis", "2020 pandemic shock months", "2022 rate-shock outliers"],
      metric: "5% contamination"
    },
  ];

  methods.forEach((m, i) => {
    const x = 0.4 + (i % 2) * 4.8;
    const y = 1.12 + Math.floor(i / 2) * 2.22;
    darkPanel(s, x, y, 4.6, 2.08);

    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 1.0, h: 0.32, fill: { color: m.color, transparency: 70 }, line: { style: "none" } });
    s.addText(m.tag, { x, y, w: 1.0, h: 0.32, fontSize: 8, color: m.color, bold: true, align: "center", valign: "middle", margin: 0 });

    s.addText(m.title, { x: x + 0.12, y: y + 0.38, w: 4.3, h: 0.28, fontSize: 12, color: C.text, bold: true, margin: 0 });
    s.addText(m.result, { x: x + 0.12, y: y + 0.66, w: 4.3, h: 0.24, fontSize: 10, color: m.color, bold: true, margin: 0 });

    m.details.forEach((d, j) => {
      s.addText("• " + d, { x: x + 0.12, y: y + 0.94 + j * 0.27, w: 4.3, h: 0.24, fontSize: 9, color: C.muted, margin: 0 });
    });

    s.addShape(pres.shapes.RECTANGLE, { x: x + 2.8, y: y + 0.1, w: 1.7, h: 0.24, fill: { color: C.navy }, line: { style: "none" } });
    s.addText(m.metric, { x: x + 2.8, y: y + 0.1, w: 1.7, h: 0.24, fontSize: 8, color: m.color, align: "center", valign: "middle", margin: 0 });
  });
}

// ══════════════════════════════════════════
// SLIDE 12 – REINFORCEMENT LEARNING
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  lightBg(s);
  sectionChip(s, "REINFORCEMENT LEARNING", 0.4, 0.18);

  s.addText("Q-Learning Market Simulation", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });

  // RL concept blocks
  const concepts = [
    { label: "STATE", val: "Market Condition", desc: "Rising / Flat / Falling HPI momentum combined with macro signal level (Low/Med/High)", color: C.teal },
    { label: "ACTION", val: "Policy Decision", desc: "Buy/Overweight  ·  Hold/Neutral  ·  Wait/Underweight", color: C.amber },
    { label: "REWARD", val: "Performance Signal", desc: "Positive when action aligns with next price movement; penalized for volatility", color: C.rose },
  ];

  concepts.forEach((c, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4 + i * 3.1, y: 1.12, w: 2.95, h: 2.2, fill: { color: C.white }, line: { color: "E2E8F0", width: 1 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4 + i * 3.1, y: 1.12, w: 2.95, h: 0.06, fill: { color: c.color }, line: { style: "none" } });
    s.addText(c.label, { x: 0.5 + i * 3.1, y: 1.2, w: 2.75, h: 0.26, fontSize: 9, color: c.color, bold: true, charSpacing: 4, margin: 0 });
    s.addText(c.val, { x: 0.5 + i * 3.1, y: 1.48, w: 2.75, h: 0.38, fontSize: 16, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });
    s.addText(c.desc, { x: 0.5 + i * 3.1, y: 1.9, w: 2.75, h: 1.3, fontSize: 9.5, color: "475569", lineSpacingMultiple: 1.35, margin: 0 });
  });

  // Parameters and results
  s.addText("Training Parameters & Results", { x: 0.4, y: 3.48, w: 4.6, h: 0.3, fontSize: 13, color: "1E293B", bold: true, margin: 0 });

  const params = [
    ["Episodes", "250", C.teal],
    ["Learning Rate", "0.18", C.teal],
    ["Discount Factor", "0.88", C.teal],
    ["Initial Epsilon", "0.35", C.amber],
    ["Risk Penalty", "1.0", C.amber],
    ["Best Action", "Buy/OW (Rising × Low signal)", C.rose],
  ];

  params.forEach((p, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 3.85 + i * 0.27, w: 4.6, h: 0.24, fill: { color: i % 2 === 0 ? "F8FAFC" : C.white }, line: { style: "none" } });
    s.addText(p[0], { x: 0.5, y: 3.85 + i * 0.27, w: 2.2, h: 0.24, fontSize: 9.5, color: "475569", valign: "middle", margin: 0 });
    s.addText(p[1], { x: 2.8, y: 3.85 + i * 0.27, w: 2.1, h: 0.24, fontSize: 9.5, color: p[2], bold: true, valign: "middle", margin: 0 });
  });

  // Key insight
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 3.48, w: 4.4, h: 2.1, fill: { color: "EFF6FF" }, line: { color: "BFDBFE", width: 1 } });
  s.addText("Educational RL Insight", { x: 5.35, y: 3.55, w: 4.1, h: 0.28, fontSize: 12, color: C.teal, bold: true, margin: 0 });
  s.addText("Unlike supervised ML (predict a value) and unsupervised learning (find groups), reinforcement learning learns a DECISION POLICY through repeated interaction.\n\nThe agent discovers that \"Rising market + Low macro signal\" is the state where buying/overweighting is most consistently rewarded across 250 training episodes.", {
    x: 5.35, y: 3.88, w: 4.1, h: 1.6, fontSize: 9.5, color: "475569", lineSpacingMultiple: 1.35, margin: 0
  });

  s.addText("⚠️ This is an educational simulation, not financial advice.", { x: 0.4, y: 5.38, w: 9.2, h: 0.2, fontSize: 8.5, color: "94A3B8", italic: true, margin: 0 });
}

// ══════════════════════════════════════════
// SLIDE 13 – OLAP CUBE
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionChip(s, "OLAP ANALYTICS", 0.4, 0.18);

  s.addText("OLAP Cube & Segmentation", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: C.text, bold: true, fontFace: "Calibri", margin: 0 });

  // OLAP explanation
  darkPanel(s, 0.4, 1.12, 4.5, 4.3);
  s.addText("3D OLAP Block Cube", { x: 0.55, y: 1.2, w: 4.2, h: 0.3, fontSize: 13, color: C.teal, bold: true, margin: 0 });
  s.addText("Three dimensions cross-cut the dataset simultaneously:\n• Administration Period (X)\n• Calendar Year (Y)\n• Market Quarter (Z)\n\nEach 3D block represents the aggregated Home Price Index for that combination of period, year, and quarter.", {
    x: 0.55, y: 1.54, w: 4.2, h: 2.0, fontSize: 10, color: C.muted, lineSpacingMultiple: 1.4, margin: 0
  });

  // Pivot table visual
  s.addText("Example Pivot: Mean HPI by Administration × Year", { x: 0.55, y: 3.6, w: 4.2, h: 0.28, fontSize: 10, color: C.text, bold: true, margin: 0 });

  const pivotRows = [
    ["Period", "2019", "2020", "2021", "2022"],
    ["Pre-Trump", "—", "—", "—", "—"],
    ["Trump 17-20", "210", "218", "—", "—"],
    ["Biden 21-24", "—", "—", "296", "312"],
  ];
  pivotRows.forEach((row, ri) => {
    row.forEach((cell, ci) => {
      const isHeader = ri === 0 || ci === 0;
      s.addShape(pres.shapes.RECTANGLE, {
        x: 0.55 + ci * 0.88, y: 3.95 + ri * 0.3,
        w: 0.86, h: 0.28,
        fill: { color: isHeader ? C.teal2 : ri % 2 === 0 ? C.panel2 : C.navy },
        line: { color: C.dim, width: 0.5 }
      });
      s.addText(cell, {
        x: 0.55 + ci * 0.88, y: 3.95 + ri * 0.3,
        w: 0.86, h: 0.28,
        fontSize: 8.5, color: C.text, align: "center", valign: "middle", margin: 0
      });
    });
  });

  // Right: OLAP insights
  const insights = [
    { period: "Biden (2021–24)", metric: "Mean HPI", val: "291.8", desc: "Highest HPI and highest average mortgage rate (5.25%)", color: C.teal },
    { period: "Trump (2017–20)", metric: "Mean HPI", val: "206.3", desc: "Moderate growth with lower average mortgage rate (3.89%)", color: C.amber },
    { period: "Pre-Trump (2004–16)", metric: "Mean HPI", val: "161.5", desc: "Boom, crisis, and recovery cycle; highest average unemployment (6.57%)", color: C.rose },
  ];

  insights.forEach((ins, i) => {
    darkPanel(s, 5.1, 1.12 + i * 1.45, 4.5, 1.32);
    s.addText(ins.period, { x: 5.25, y: 1.18 + i * 1.45, w: 4.2, h: 0.26, fontSize: 10, color: ins.color, bold: true, margin: 0 });
    s.addText(ins.val, { x: 5.25, y: 1.44 + i * 1.45, w: 2.0, h: 0.42, fontSize: 24, color: ins.color, bold: true, fontFace: "Calibri", margin: 0 });
    s.addText(ins.metric, { x: 7.25, y: 1.44 + i * 1.45, w: 2.25, h: 0.28, fontSize: 9, color: C.muted, valign: "bottom", margin: 0 });
    s.addText(ins.desc, { x: 5.25, y: 1.88 + i * 1.45, w: 4.2, h: 0.5, fontSize: 9, color: C.muted, lineSpacingMultiple: 1.25, margin: 0 });
  });

  visualFrame(s, ASSETS.olap, 0.55, 4.36, 4.2, 1.05);
  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 4.5, w: 4.5, h: 0.9, fill: { color: C.panel2 }, line: { color: C.teal, width: 0.75 } });
  s.addText("OLAP Key Insight: Biden-era average HPI is the highest while mortgage-rate pressure is also highest. This supports a mixed interpretation: affordability pressure exists, but supply and timing keep prices resilient.", {
    x: 5.2, y: 4.54, w: 4.3, h: 0.82, fontSize: 9, color: C.muted, lineSpacingMultiple: 1.3, margin: 0
  });
}

// ══════════════════════════════════════════
// SLIDE 14 – FORECASTING
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  lightBg(s);
  sectionChip(s, "FORECASTING", 0.4, 0.18);

  s.addText("Multi-Step Housing Price Forecast", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });
  visualFrame(s, ASSETS.pred, 5.65, 1.08, 3.95, 1.9);

  // Forecast chart (stylized bar)
  const histData = [185, 192, 198, 210, 218, 225, 234, 242, 256, 270, 285, 296, 305, 312, 318];
  const forecastData = [325, 329, 333, 337, 340, 343];
  const allLabels = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "H1'25", "H2'25", "H1'26", "H2'26", "H1'27", "H2'27"];

  s.addChart(pres.charts.LINE, [
    {
      name: "Historical HPI",
      labels: allLabels.slice(0, 15),
      values: histData
    }
  ], {
    x: 0.4, y: 1.12, w: 9.2, h: 3.5,
    chartColors: ["0D9488"],
    lineSize: 2.5,
    lineSmooth: true,
    chartArea: { fill: { color: "FFFFFF" } },
    catAxisLabelColor: "475569",
    valAxisLabelColor: "475569",
    valGridLine: { color: "E2E8F0" },
    catGridLine: { style: "none" },
    showLegend: false,
    showValue: false,
  });

  // Forecast methodology blocks
  const methods = [
    { name: "Lag Features", desc: "HPI_lag1, lag3, lag6, lag12 + rolling means feed recursive prediction", color: C.teal },
    { name: "Recursive Strategy", desc: "Each prediction becomes the input for the next step (multi-step chain)", color: C.amber },
    { name: "7 Models Tested", desc: "Ridge is most stable for long horizon; RF for shorter-term accuracy", color: C.sky },
  ];

  methods.forEach((m, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4 + i * 3.1, y: 4.72, w: 2.95, h: 0.78, fill: { color: C.white }, line: { color: "E2E8F0", width: 1 } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4 + i * 3.1, y: 4.72, w: 2.95, h: 0.06, fill: { color: m.color }, line: { style: "none" } });
    s.addText(m.name, { x: 0.5 + i * 3.1, y: 4.8, w: 2.75, h: 0.24, fontSize: 10, color: "1E293B", bold: true, margin: 0 });
    s.addText(m.desc, { x: 0.5 + i * 3.1, y: 5.04, w: 2.75, h: 0.4, fontSize: 9, color: "475569", lineSpacingMultiple: 1.25, margin: 0 });
  });

  s.addText("Forecast is a scenario tool — not guaranteed future truth. Best used alongside theory, OLAP, and evaluation evidence.", {
    x: 0.4, y: 5.52, w: 9.2, h: 0.2, fontSize: 8.5, color: "94A3B8", italic: true, margin: 0
  });
}

// ══════════════════════════════════════════
// SLIDE 15 – STREAMLIT DASHBOARD OVERVIEW
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionChip(s, "DEPLOYMENT", 0.4, 0.18);

  s.addText("Streamlit Intelligence Platform", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: C.text, bold: true, fontFace: "Calibri", margin: 0 });

  // 22 pages in a grid
  const pages = [
    ["Start Here", "Overview", "Explore", "Data Quality"],
    ["Compare", "ML Lab", "Evaluation", "Unsupervised Lab"],
    ["Reinforcement Lab", "Forecast", "Conclusion", "Paper Review"],
    ["Code Lab", "OLAP & Export", "Exec. Summary", "Data Dictionary"],
    ["Scenario Sim.", "Prod. Readiness", "Experiment Tracker", "Model Registry"],
    ["Data Pipeline", "Business Impact", "", ""],
  ];

  const pageColors = [C.teal, C.teal, C.sky, C.sky, C.amber, C.amber, C.rose, C.rose, C.teal2, C.teal2, C.muted, C.muted, C.dim, C.teal, C.teal, C.amber];

  let idx = 0;
  pages.forEach((row, ri) => {
    row.forEach((page, ci) => {
      if (!page) return;
      const x = 0.4 + ci * 2.35;
      const y = 1.15 + ri * 0.73;
      darkPanel(s, x, y, 2.2, 0.62);
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h: 0.62, fill: { color: pageColors[idx % pageColors.length] }, line: { style: "none" } });
      s.addText(page, { x: x + 0.14, y, w: 2.0, h: 0.62, fontSize: 9.5, color: C.text, valign: "middle", margin: 0 });
      idx++;
    });
  });

  // Real app screenshots from src/app_screenshots, added to connect the presentation to the implemented dashboard.
  visualFrame(s, ASSETS.appHome, 5.18, 1.15, 4.42, 2.05);
  visualFrame(s, ASSETS.appEvaluation, 5.18, 3.38, 4.42, 1.85);

  // Feature highlights
  darkPanel(s, 0.4, 5.52, 9.2, 0.48);
  s.addText("22 interactive pages  ·  Dark/Light mode  ·  AI Chatbot (OpenAI + local fallback)  ·  Download-ready reports  ·  Model Registry & Governance  ·  OLAP 3D Cube  ·  Scenario Simulator", {
    x: 0.5, y: 5.56, w: 9.0, h: 0.38, fontSize: 9.5, color: C.teal, align: "center", margin: 0
  });
}

// ══════════════════════════════════════════
// SLIDE 16 – DASHBOARD FEATURES DEEP DIVE
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  lightBg(s);
  sectionChip(s, "DASHBOARD FEATURES", 0.4, 0.18);

  s.addText("Dashboard Capability Map", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });

  const pillars = [
    {
      title: "Data Foundation",
      icon: "📊",
      color: C.teal,
      items: ["EDA & Time-Series Charts", "Correlation Matrix (zoom)", "Data Quality Checks", "Data Dictionary (27 cols)"],
    },
    {
      title: "Machine Learning",
      icon: "🤖",
      color: C.amber,
      items: ["8 Regression Models", "5 Classifier Models", "Feature Importance", "Cross-Validation"],
    },
    {
      title: "Advanced Analytics",
      icon: "🔬",
      color: C.sky,
      items: ["KMeans / DBSCAN / PCA", "IsolationForest Anomalies", "Q-Learning Simulation", "Multi-step Forecasting"],
    },
    {
      title: "Governance & BI",
      icon: "🏢",
      color: C.rose,
      items: ["3D OLAP Block Cube", "Model Registry & Card", "Drift Monitoring", "Experiment Tracker"],
    },
  ];

  pillars.forEach((p, i) => {
    const x = 0.4 + i * 2.35;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.12, w: 2.2, h: 4.3, fill: { color: C.white }, line: { color: "E2E8F0", width: 1 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.12, w: 2.2, h: 0.06, fill: { color: p.color }, line: { style: "none" } });
    s.addText(p.icon, { x, y: 1.2, w: 2.2, h: 0.45, fontSize: 20, align: "center", margin: 0 });
    s.addText(p.title, { x: x + 0.1, y: 1.68, w: 2.0, h: 0.42, fontSize: 11, color: "1E293B", bold: true, align: "center", lineSpacingMultiple: 1.1, margin: 0 });
    s.addShape(pres.shapes.LINE, { x: x + 0.2, y: 2.12, w: 1.8, h: 0, line: { color: "E2E8F0", width: 1 } });
    p.items.forEach((item, j) => {
      s.addShape(pres.shapes.OVAL, { x: x + 0.12, y: 2.22 + j * 0.72, w: 0.14, h: 0.14, fill: { color: p.color }, line: { style: "none" } });
      s.addText(item, { x: x + 0.32, y: 2.17 + j * 0.72, w: 1.78, h: 0.56, fontSize: 9.5, color: "475569", lineSpacingMultiple: 1.2, margin: 0 });
    });
  });

  // App evidence thumbnails: ML Lab, OLAP, and Paper Review are real screenshots from the project.
  visualFrame(s, ASSETS.appModeling, 0.55, 4.55, 2.85, 0.82);
  visualFrame(s, ASSETS.appOlap, 3.65, 4.55, 2.85, 0.82);
  visualFrame(s, ASSETS.appPaperReview, 6.75, 4.55, 2.85, 0.82);

  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 5.5, w: 9.2, h: 0.3, fill: { color: "EFF6FF" }, line: { color: "BFDBFE", width: 1 } });
  s.addText("AI Chatbot (OpenAI + local fallback)  ·  Scenario Simulator  ·  Downloadable Executive Report  ·  Business Impact Translator", {
    x: 0.5, y: 5.54, w: 9.0, h: 0.22, fontSize: 9.5, color: C.teal, align: "center", margin: 0
  });
}

// ══════════════════════════════════════════
// SLIDE 17 – HOUSING THEORY VALIDATION
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionChip(s, "THEORY VALIDATION", 0.4, 0.18);

  s.addText("DiPasquale-Wheaton Theory Check", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: C.text, bold: true, fontFace: "Calibri", margin: 0 });

  darkPanel(s, 0.4, 1.1, 9.2, 0.65);
  s.addText("The DiPasquale-Wheaton four-quadrant model links real-estate prices to asset markets, rental demand, construction output, and property stock adjustment. We test four core relationships:", {
    x: 0.55, y: 1.15, w: 9.0, h: 0.55, fontSize: 10, color: C.muted, lineSpacingMultiple: 1.3, margin: 0
  });

  const checks = [
    { theory: "Price Momentum", hypothesis: "Lagged and smoothed HPI → higher current HPI", feature: "HPI_lag1", corr: 1.00, verdict: "STRONGLY SUPPORTS", color: C.teal },
    { theory: "Income-Demand Theory", hypothesis: "Stronger income → higher HPI", feature: "Median_Income", corr: 0.88, verdict: "SUPPORTS", color: C.teal },
    { theory: "Affordability Pressure", hypothesis: "Rates pressure affordability but prices may lag", feature: "Mortgage_Rate", corr: 0.47, verdict: "MIXED", color: C.amber },
    { theory: "Supply-Demand Theory", hypothesis: "Inventory/supply data needed for full validation", feature: "Housing_Supply", corr: null, verdict: "MISSING FEATURE", color: C.rose },
  ];

  checks.forEach((c, i) => {
    const y = 1.88 + i * 0.88;
    darkPanel(s, 0.4, y, 9.2, 0.8);
    s.addText(c.theory, { x: 0.55, y: y + 0.06, w: 2.5, h: 0.3, fontSize: 11, color: C.text, bold: true, margin: 0 });
    s.addText(c.hypothesis, { x: 0.55, y: y + 0.4, w: 3.5, h: 0.3, fontSize: 9, color: C.muted, italic: true, margin: 0 });

    if (c.corr !== null) {
      const barColor = c.corr > 0 ? C.teal : C.rose;
      s.addShape(pres.shapes.RECTANGLE, { x: 4.3, y: y + 0.24, w: 3.0, h: 0.14, fill: { color: C.navy }, line: { style: "none" } });
      s.addShape(pres.shapes.RECTANGLE, { x: 4.3, y: y + 0.24, w: Math.abs(c.corr) * 3.0, h: 0.14, fill: { color: barColor }, line: { style: "none" } });
      s.addText((c.corr > 0 ? "+" : "") + c.corr.toFixed(2), { x: 7.4, y: y + 0.14, w: 0.8, h: 0.5, fontSize: 14, color: barColor, bold: true, align: "center", valign: "middle", margin: 0 });
    }

    s.addShape(pres.shapes.RECTANGLE, { x: 8.3, y: y + 0.18, w: 1.2, h: 0.42, fill: { color: c.verdict === "MISSING FEATURE" ? C.rose : C.teal, transparency: 80 }, line: { style: "none" } });
    s.addText(c.verdict, { x: 8.3, y: y + 0.18, w: 1.2, h: 0.42, fontSize: 7.5, color: c.verdict === "MISSING FEATURE" ? C.rose : C.teal, bold: true, align: "center", valign: "middle", margin: 0 });
  });

  darkPanel(s, 0.4, 5.42, 9.2, 0.4);
  s.addText("Result: the data strongly supports price persistence and income-demand logic. Rate pressure is mixed because prices can stay elevated when supply and timing slow the market response.", {
    x: 0.55, y: 5.46, w: 9.0, h: 0.3, fontSize: 9.5, color: C.teal, valign: "middle", margin: 0
  });
}

// ══════════════════════════════════════════
// SLIDE 18 – HISTORICAL PERIOD COMPARISON
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  lightBg(s);
  sectionChip(s, "HISTORICAL VALIDATION", 0.4, 0.18);

  s.addText("OLAP Period Comparison", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });

  const periods = [
    { period: "Pre-Trump", dates: "2004–2016", expect: "Baseline", actual: "Avg HPI", pct: "161.5", verdict: "Reference", color: C.sky, desc: "Boom, crisis, and recovery years produce the lowest average HPI in the OLAP summary" },
    { period: "Trump", dates: "2017–2020", expect: "Expansion", actual: "Avg HPI", pct: "206.3", verdict: "Higher", color: C.teal, desc: "Average HPI increases while average mortgage-rate pressure is lower at 3.89%" },
    { period: "Biden", dates: "2021–2024", expect: "High-rate period", actual: "Avg HPI", pct: "291.8", verdict: "Highest", color: C.amber, desc: "Highest average HPI and highest average mortgage-rate pressure at 5.25%" },
    { period: "Biden", dates: "2021–2024", expect: "Labor market", actual: "Unemployment", pct: "4.17%", verdict: "Lowest", color: C.teal, desc: "Lowest average unemployment across the three administration periods" },
    { period: "Biden", dates: "2021–2024", expect: "Supply activity", actual: "Permits", pct: "1627", verdict: "Highest", color: C.rose, desc: "Highest average building permits, showing that price levels and supply activity can rise together" },
  ];

  periods.forEach((p, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 1.1 + i * 0.88, w: 9.2, h: 0.8, fill: { color: C.white }, line: { color: "E2E8F0", width: 1 } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 1.1 + i * 0.88, w: 0.06, h: 0.8, fill: { color: p.color }, line: { style: "none" } });

    s.addText(p.period, { x: 0.55, y: 1.14 + i * 0.88, w: 2.0, h: 0.3, fontSize: 11, color: "1E293B", bold: true, margin: 0 });
    s.addText(p.dates, { x: 0.55, y: 1.44 + i * 0.88, w: 2.0, h: 0.22, fontSize: 9, color: "64748B", margin: 0 });

    s.addText(p.desc, { x: 2.65, y: 1.14 + i * 0.88, w: 3.6, h: 0.64, fontSize: 9.5, color: "475569", lineSpacingMultiple: 1.3, margin: 0 });

    s.addText(p.pct, { x: 6.4, y: 1.14 + i * 0.88, w: 1.4, h: 0.64, fontSize: 18, color: p.actual === "Fall" ? C.rose : p.color, bold: true, fontFace: "Calibri", align: "center", valign: "middle", margin: 0 });

    s.addShape(pres.shapes.RECTANGLE, { x: 7.95, y: 1.18 + i * 0.88, w: 1.55, h: 0.54, fill: { color: p.verdict === "Matches" ? "DCFCE7" : "FEF9C3" }, line: { style: "none" } });
    s.addText(p.verdict, { x: 7.95, y: 1.18 + i * 0.88, w: 1.55, h: 0.54, fontSize: 9.5, color: p.verdict === "Matches" ? "16A34A" : "CA8A04", bold: true, align: "center", valign: "middle", margin: 0 });
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 5.5, w: 9.2, h: 0.28, fill: { color: "EFF6FF" }, line: { color: "BFDBFE", width: 1 } });
  s.addText("OLAP result: the Biden period has both the highest average HPI and the highest mortgage-rate pressure, supporting a mixed affordability-and-supply interpretation.", {
    x: 0.5, y: 5.54, w: 9.0, h: 0.2, fontSize: 9, color: C.teal, align: "center", margin: 0
  });
}

// ══════════════════════════════════════════
// SLIDE 19 – PRODUCTION READINESS
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionChip(s, "PRODUCTION READINESS", 0.4, 0.18);

  s.addText("Enterprise-Grade Governance", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: C.text, bold: true, fontFace: "Calibri", margin: 0 });
  visualFrame(s, ASSETS.appGovernance, 6.45, 0.92, 3.15, 1.2);

  const governance = [
    {
      area: "Data Validation",
      icon: "✅",
      checks: ["Row count verification", "Duplicate detection", "Missing data profiling", "Infinite value check", "Date column detection"],
      color: C.teal,
    },
    {
      area: "Drift Monitoring",
      icon: "📡",
      checks: ["Baseline vs recent split", "Mean shift % per feature", "Z-score drift alert", "High/Moderate/Stable labels", "Auto-flagged review candidates"],
      color: C.sky,
    },
    {
      area: "Model Card",
      icon: "📋",
      checks: ["Intended use statement", "Performance metrics logged", "Limitation documentation", "Governance recommendation", "Download-ready Markdown"],
      color: C.amber,
    },
    {
      area: "Experiment Tracker",
      icon: "🧪",
      checks: ["Run ID & timestamp", "Model, target, features", "All metric columns", "Champion selection", "CSV export for audit"],
      color: C.rose,
    },
  ];

  governance.forEach((g, i) => {
    const x = 0.4 + (i % 2) * 4.8;
    const y = 1.15 + Math.floor(i / 2) * 2.2;
    darkPanel(s, x, y, 4.6, 2.05);

    s.addText(g.icon + "  " + g.area, { x: x + 0.15, y: y + 0.1, w: 4.3, h: 0.35, fontSize: 13, color: g.color, bold: true, margin: 0 });
    g.checks.forEach((c, j) => {
      s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: y + 0.58 + j * 0.27, w: 0.14, h: 0.14, fill: { color: g.color }, line: { style: "none" } });
      s.addText(c, { x: x + 0.36, y: y + 0.53 + j * 0.27, w: 4.0, h: 0.24, fontSize: 9.5, color: C.muted, margin: 0 });
    });
  });

  darkPanel(s, 0.4, 5.42, 9.2, 0.38);
  s.addText("This project includes not just analytics — it adds validation, monitoring, explainability, scenario testing, and full documentation, matching enterprise data science standards.", {
    x: 0.55, y: 5.46, w: 9.0, h: 0.3, fontSize: 9.5, color: C.teal, valign: "middle", margin: 0
  });
}

// ══════════════════════════════════════════
// SLIDE 20 – SCENARIO SIMULATOR
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  lightBg(s);
  sectionChip(s, "SCENARIO SIMULATOR", 0.4, 0.18);

  s.addText("What-If Scenario Analysis", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });

  s.addText("The Scenario Simulator changes one macro feature at a time and estimates how the trained regression model responds. Useful for explaining sensitivity to stakeholders.", {
    x: 0.4, y: 1.1, w: 9.2, h: 0.38, fontSize: 10, color: "475569", margin: 0
  });

  // Three scenario examples
  const scenarios = [
    {
      title: "+10% Median Income",
      change: "+10%",
      feature: "Median_Income",
      baseline: "312.4",
      scenario: "318.2",
      delta: "+5.8",
      direction: "up",
      insight: "A 10% income rise pushes predicted HPI up ~5.8 points. Affordability improves when income grows faster than rates.",
      color: C.teal,
    },
    {
      title: "+2pt Mortgage Rate",
      change: "+2 pts",
      feature: "Mortgage_Rate",
      baseline: "312.4",
      scenario: "305.1",
      delta: "−7.3",
      direction: "down",
      insight: "A 2-point rate hike reduces predicted HPI by ~7.3. Larger than income effect, showing rate sensitivity.",
      color: C.rose,
    },
    {
      title: "−5% Unemployment",
      change: "−5%",
      feature: "Unemployment_Rate",
      baseline: "312.4",
      scenario: "316.8",
      delta: "+4.4",
      direction: "up",
      insight: "Falling unemployment supports HPI through stronger demand — aligns with DiPasquale-Wheaton labor theory.",
      color: C.amber,
    },
  ];

  scenarios.forEach((sc, i) => {
    const x = 0.4 + i * 3.1;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.58, w: 2.95, h: 3.85, fill: { color: C.white }, line: { color: "E2E8F0", width: 1 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.58, w: 2.95, h: 0.06, fill: { color: sc.color }, line: { style: "none" } });

    s.addText(sc.title, { x: x + 0.12, y: 1.68, w: 2.7, h: 0.35, fontSize: 12, color: "1E293B", bold: true, lineSpacingMultiple: 1.1, margin: 0 });
    s.addText("Feature: " + sc.feature, { x: x + 0.12, y: 2.04, w: 2.7, h: 0.22, fontSize: 8.5, color: "64748B", italic: true, margin: 0 });

    s.addText("Baseline", { x: x + 0.12, y: 2.35, w: 1.3, h: 0.22, fontSize: 9, color: "64748B", margin: 0 });
    s.addText(sc.baseline, { x: x + 0.12, y: 2.57, w: 1.3, h: 0.38, fontSize: 20, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });

    s.addText("Scenario", { x: x + 1.55, y: 2.35, w: 1.3, h: 0.22, fontSize: 9, color: "64748B", margin: 0 });
    s.addText(sc.scenario, { x: x + 1.55, y: 2.57, w: 1.3, h: 0.38, fontSize: 20, color: sc.color, bold: true, fontFace: "Calibri", margin: 0 });

    s.addShape(pres.shapes.RECTANGLE, { x: x + 0.12, y: 3.04, w: 2.7, h: 0.42, fill: { color: sc.direction === "up" ? "DCFCE7" : "FEE2E2" }, line: { style: "none" } });
    s.addText("Change: " + sc.delta, { x: x + 0.12, y: 3.04, w: 2.7, h: 0.42, fontSize: 14, color: sc.color, bold: true, align: "center", valign: "middle", fontFace: "Calibri", margin: 0 });

    s.addText(sc.insight, { x: x + 0.12, y: 3.54, w: 2.7, h: 1.72, fontSize: 9, color: "475569", lineSpacingMultiple: 1.35, margin: 0 });
  });

  s.addText("All scenarios use the trained Ridge regression model on the most recent observation. Results are directional estimates, not causal predictions.", {
    x: 0.4, y: 5.5, w: 9.2, h: 0.22, fontSize: 8.5, color: "94A3B8", italic: true, margin: 0
  });
}

// ══════════════════════════════════════════
// SLIDE 21 – PAPER REVIEW
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionChip(s, "LITERATURE REVIEW", 0.4, 0.18);

  s.addText("Paper Review & Real-Life Benchmarking", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 26, color: C.text, bold: true, fontFace: "Calibri", margin: 0 });

  const papers = [
    {
      source: "DiPasquale & Wheaton (2015)",
      claim: "Four-quadrant framework: prices connect to demand, asset markets, construction, and supply adjustment.",
      ourResult: "3/4 theory checks supported; supply constraints explain 2022–24 resilience",
      verdict: "Confirms",
      color: C.teal,
    },
    {
      source: "Journal of Housing Economics (2025)",
      claim: "Mortgage-rate shocks have measurable effects on housing activity, sometimes larger than on prices.",
      ourResult: "Rate correlation: −0.44; RF model uses rate as 5th most important feature",
      verdict: "Confirms",
      color: C.teal,
    },
    {
      source: "NAR Existing-Home Sales (March 2026)",
      claim: "Sales fell 3.6% MoM; median price +1.4% YoY to $408,800; inventory 4.1 months.",
      ourResult: "Dataset shows HPI positive but slowing — consistent with soft-landing narrative",
      verdict: "Consistent",
      color: C.amber,
    },
    {
      source: "FRED Case-Shiller Index (Jan 2026)",
      claim: "National home price index at 326.6 — near all-time highs despite rate environment.",
      ourResult: "HPI positive trend maintained through 2024 in our dataset",
      verdict: "Consistent",
      color: C.amber,
    },
  ];

  papers.forEach((p, i) => {
    const y = 1.12 + i * 1.1;
    darkPanel(s, 0.4, y, 9.2, 1.0);
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y, w: 0.06, h: 1.0, fill: { color: p.color }, line: { style: "none" } });

    s.addText(p.source, { x: 0.55, y: y + 0.06, w: 6.0, h: 0.26, fontSize: 10.5, color: p.color, bold: true, margin: 0 });
    s.addText("Claim: " + p.claim, { x: 0.55, y: y + 0.34, w: 7.3, h: 0.26, fontSize: 9, color: C.muted, margin: 0 });
    s.addText("Our result: " + p.ourResult, { x: 0.55, y: y + 0.62, w: 7.3, h: 0.26, fontSize: 9, color: C.text, margin: 0 });

    s.addShape(pres.shapes.RECTANGLE, { x: 8.0, y: y + 0.28, w: 1.45, h: 0.42, fill: { color: p.color, transparency: 75 }, line: { style: "none" } });
    s.addText(p.verdict, { x: 8.0, y: y + 0.28, w: 1.45, h: 0.42, fontSize: 9, color: p.color, bold: true, align: "center", valign: "middle", margin: 0 });
  });

  darkPanel(s, 0.4, 5.5, 9.2, 0.38);
  s.addText("Conclusion: the dataset broadly aligns with current academic research and real 2026 market data — a positive project result.", {
    x: 0.55, y: 5.54, w: 9.0, h: 0.3, fontSize: 9.5, color: C.teal, valign: "middle", margin: 0
  });
}

// ══════════════════════════════════════════
// SLIDE 22 – BUSINESS IMPACT
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  lightBg(s);
  sectionChip(s, "BUSINESS IMPACT", 0.4, 0.18);

  s.addText("Business Value & Stakeholder Impact", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });

  const stakeholders = [
    {
      who: "Analysts & Economists",
      icon: "📈",
      value: "Quantify macro drivers",
      howApp: ["Correlation matrix identifies key signals", "Feature importance explains model logic", "Theory checks validate assumptions"],
      color: C.teal,
    },
    {
      who: "Institutional Investors",
      icon: "🏦",
      value: "Regime-aware decisions",
      howApp: ["KMeans identifies 3 market regimes", "RL simulation learns best policy per state", "OLAP segments by period and geography"],
      color: C.amber,
    },
    {
      who: "Policymakers",
      icon: "⚖️",
      value: "Affordability monitoring",
      howApp: ["Income-to-Mortgage ratio dashboard", "Scenario simulator for rate impact", "Historical period benchmarking"],
      color: C.sky,
    },
    {
      who: "Project Evaluators",
      icon: "🎓",
      value: "Complete DS workflow",
      howApp: ["CRISP-DM all 6 phases executed", "Governance, drift, model card included", "Paper review + real-life benchmarking"],
      color: C.rose,
    },
  ];

  stakeholders.forEach((st, i) => {
    const x = 0.4 + (i % 2) * 4.8;
    const y = 1.12 + Math.floor(i / 2) * 2.25;
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 4.6, h: 2.1, fill: { color: C.white }, line: { color: "E2E8F0", width: 1 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 4.6, h: 0.06, fill: { color: st.color }, line: { style: "none" } });

    s.addText(st.icon, { x, y: y + 0.1, w: 0.7, h: 0.5, fontSize: 22, align: "center", margin: 0 });
    s.addText(st.who, { x: x + 0.72, y: y + 0.1, w: 3.75, h: 0.28, fontSize: 11, color: "1E293B", bold: true, margin: 0 });
    s.addText(st.value, { x: x + 0.72, y: y + 0.38, w: 3.75, h: 0.22, fontSize: 9, color: st.color, bold: true, margin: 0 });
    s.addShape(pres.shapes.LINE, { x: x + 0.15, y: y + 0.68, w: 4.3, h: 0, line: { color: "F1F5F9", width: 1 } });
    st.howApp.forEach((h, j) => {
      s.addShape(pres.shapes.OVAL, { x: x + 0.15, y: y + 0.82 + j * 0.4, w: 0.14, h: 0.14, fill: { color: st.color }, line: { style: "none" } });
      s.addText(h, { x: x + 0.36, y: y + 0.76 + j * 0.4, w: 4.0, h: 0.36, fontSize: 9.5, color: "475569", margin: 0 });
    });
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 5.5, w: 9.2, h: 0.3, fill: { color: "EFF6FF" }, line: { color: "BFDBFE", width: 1 } });
  s.addText("The app turns housing data into a decision-ready signal — connecting data quality, model performance, segment analysis, scenario testing, and real-world theory.", {
    x: 0.5, y: 5.54, w: 9.0, h: 0.22, fontSize: 9.5, color: C.teal, align: "center", margin: 0
  });
}

// ══════════════════════════════════════════
// SLIDE 23 – KEY FINDINGS
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);
  sectionChip(s, "KEY FINDINGS", 0.4, 0.18);

  s.addText("Key Findings Summary", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 30, color: C.text, bold: true, fontFace: "Calibri", margin: 0 });

  const findings = [
    {
      num: "01",
      title: "Linear trend model wins",
      body: "Linear Regression (R² 0.999, MAE 0.661, RMSE 0.797) outperforms Ridge and tree models because lagged and smoothed HPI capture the rising chronological trend.",
      color: C.teal,
    },
    {
      num: "02",
      title: "Price persistence is strongest",
      body: "Smoothed HPI, HPI_lag1, and HPI_lag3 show near-perfect correlation with the target, confirming strong month-to-month housing-price momentum.",
      color: C.amber,
    },
    {
      num: "03",
      title: "Momentum is essential",
      body: "Median income remains a strong explanatory context variable (r = 0.88). Model score is not the full story; economics still explains why the trend exists.",
      color: C.sky,
    },
    {
      num: "04",
      title: "Tree models show extrapolation risk",
      body: "Random Forest, Gradient Boosting, Decision Tree, KNN, and SVR perform weakly on the final test period, proving why chronological diagnostics are essential.",
      color: C.rose,
    },
    {
      num: "05",
      title: "OLAP supports a mixed conclusion",
      body: "The Biden period has the highest average HPI and highest average mortgage rate, so price resilience must be interpreted with supply, timing, and market inertia.",
      color: C.teal2,
    },
  ];

  findings.forEach((f, i) => {
    const y = 1.12 + i * 0.88;
    darkPanel(s, 0.4, y, 9.2, 0.8);

    s.addShape(pres.shapes.OVAL, { x: 0.5, y: y + 0.18, w: 0.44, h: 0.44, fill: { color: f.color }, line: { style: "none" } });
    s.addText(f.num, { x: 0.5, y: y + 0.18, w: 0.44, h: 0.44, fontSize: 11, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });

    s.addText(f.title, { x: 1.08, y: y + 0.06, w: 3.0, h: 0.3, fontSize: 12, color: f.color, bold: true, margin: 0 });
    s.addText(f.body, { x: 1.08, y: y + 0.38, w: 8.4, h: 0.36, fontSize: 9.5, color: C.muted, lineSpacingMultiple: 1.2, margin: 0 });
  });
}

// ══════════════════════════════════════════
// SLIDE 24 – LIMITATIONS & FUTURE WORK
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  lightBg(s);
  sectionChip(s, "FUTURE WORK", 0.4, 0.18);

  s.addText("Limitations & Future Directions", { x: 0.4, y: 0.52, w: 9.2, h: 0.5, fontSize: 28, color: "1E293B", bold: true, fontFace: "Calibri", margin: 0 });

  // Limitations
  s.addText("Current Limitations", { x: 0.4, y: 1.12, w: 4.5, h: 0.3, fontSize: 14, color: "1E293B", bold: true, margin: 0 });
  const limits = [
    { title: "National-level only", desc: "Geographic heterogeneity across states/cities is not captured — regional markets behave very differently." },
    { title: "Supply feature missing", desc: "Housing inventory/supply variable is absent, preventing a complete DiPasquale-Wheaton validation." },
    { title: "No causal inference", desc: "Correlation and ML prediction do not prove causation — observational study limitations apply." },
    { title: "Static dataset", desc: "No live data feed; dataset requires manual update for real-time monitoring." },
  ];

  limits.forEach((l, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 1.5 + i * 0.9, w: 4.5, h: 0.8, fill: { color: "FEF2F2" }, line: { color: "FECACA", width: 1 } });
    s.addText("⚠ " + l.title, { x: 0.55, y: 1.55 + i * 0.9, w: 4.2, h: 0.28, fontSize: 10.5, color: "DC2626", bold: true, margin: 0 });
    s.addText(l.desc, { x: 0.55, y: 1.83 + i * 0.9, w: 4.2, h: 0.4, fontSize: 9, color: "475569", lineSpacingMultiple: 1.25, margin: 0 });
  });

  // Future work
  s.addText("Future Development", { x: 5.1, y: 1.12, w: 4.5, h: 0.3, fontSize: 14, color: "1E293B", bold: true, margin: 0 });
  const future = [
    { title: "Regional Data", desc: "State/city level granularity via Zillow, Redfin, or Census micro-datasets.", color: C.teal },
    { title: "Live API Integration", desc: "Automated FRED, NAR, Census monthly refresh for real-time monitoring.", color: C.teal },
    { title: "SHAP Explainability", desc: "Shapley values for granular per-prediction feature attribution.", color: C.amber },
    { title: "ARIMA / LSTM Forecasting", desc: "Dedicated time-series models for improved long-horizon accuracy.", color: C.amber },
    { title: "Cloud Deployment", desc: "Streamlit Cloud or AWS with authentication, persistent DB, monitoring.", color: C.sky },
    { title: "Model Drift Alerts", desc: "Automated retraining triggers when distribution shift exceeds thresholds.", color: C.sky },
  ];

  future.forEach((f, i) => {
    const y = 1.5 + i * 0.62;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y, w: 4.5, h: 0.55, fill: { color: C.white }, line: { color: "E2E8F0", width: 1 } });
    s.addShape(pres.shapes.OVAL, { x: 5.2, y: y + 0.2, w: 0.14, h: 0.14, fill: { color: f.color }, line: { style: "none" } });
    s.addText(f.title, { x: 5.42, y: y + 0.04, w: 4.05, h: 0.24, fontSize: 10, color: "1E293B", bold: true, margin: 0 });
    s.addText(f.desc, { x: 5.42, y: y + 0.28, w: 4.05, h: 0.22, fontSize: 8.5, color: "64748B", margin: 0 });
  });
}

// ══════════════════════════════════════════
// SLIDE 25 – CONCLUSION / THANK YOU
// ══════════════════════════════════════════
{
  const s = pres.addSlide();
  darkBg(s);

  // Full-width top dark strip
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 1.6, fill: { color: C.navy2 }, line: { style: "none" } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.teal }, line: { style: "none" } });
  sectionChip(s, "CONCLUSION", 0.4, 0.18);

  s.addText("Final Conclusion", { x: 0.4, y: 0.52, w: 9.2, h: 0.72, fontSize: 36, color: C.text, bold: true, fontFace: "Calibri", margin: 0 });

  // Main conclusion statement
  darkPanel(s, 0.4, 1.72, 9.2, 1.3);
  s.addText('"The U.S. housing market is best explained by combining price momentum, income-demand context, model diagnostics, OLAP segmentation, and paper-review theory. The final evidence shows strong trend persistence, but also warns that popular tree models may fail when the test period moves beyond the training range."', {
    x: 0.6, y: 1.8, w: 8.8, h: 1.12, fontSize: 10.5, color: C.text, italic: true, lineSpacingMultiple: 1.4, margin: 0
  });

  // Summary row
  const summaries = [
    { val: "R² 0.999", lbl: "Best Model\n(LinearReg.)", color: C.teal },
    { val: "0.797", lbl: "Best\nRMSE", color: C.amber },
    { val: "291.8", lbl: "Biden\nAvg HPI", color: C.sky },
    { val: "22", lbl: "Dashboard\nPages", color: C.rose },
  ];

  summaries.forEach((sm, i) => {
    darkPanel(s, 0.4 + i * 2.3, 3.15, 2.15, 1.0);
    s.addText(sm.val, { x: 0.4 + i * 2.3, y: 3.2, w: 2.15, h: 0.55, fontSize: 26, color: sm.color, bold: true, align: "center", fontFace: "Calibri", margin: 0 });
    s.addText(sm.lbl, { x: 0.4 + i * 2.3, y: 3.72, w: 2.15, h: 0.38, fontSize: 9, color: C.muted, align: "center", lineSpacingMultiple: 1.1, margin: 0 });
  });

  // Authors box
  darkPanel(s, 0.4, 4.28, 9.2, 0.82);
  s.addText("Ayoub Naouech  ·  Rayen Belhadj", { x: 0.5, y: 4.34, w: 7.0, h: 0.32, fontSize: 15, color: C.text, bold: true, margin: 0 });
  s.addText("IT Business School · Tunisia  ·  Data Mining & Machine Learning  ·  2025–2026  ·  Supervisor: Kaouether Ben Ali", {
    x: 0.5, y: 4.68, w: 9.0, h: 0.24, fontSize: 9, color: C.muted, margin: 0
  });

  s.addText("Built with: Python · scikit-learn · Streamlit · Plotly · PptxGenJS", {
    x: 0.4, y: 5.22, w: 9.2, h: 0.2, fontSize: 8.5, color: C.dim, italic: true, margin: 0
  });

  s.addText("Thank You", { x: 0.4, y: 5.44, w: 9.2, h: 0.3, fontSize: 24, color: C.teal, bold: true, fontFace: "Calibri", margin: 0 });
}

// ── WRITE FILE ──

pres.writeFile({ fileName: "Housing_Market_Presentation.pptx" }).then(() => console.log("✅ Presentation written successfully"))
  .catch(err => { console.error("❌ Error:", err); process.exit(1); });;
