import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadSource } from "./load-source.mjs";

const { DEFAULT_TRAINING_OPTIONS, containsPrivateEvaluationData, trainModels, trainingDownloads } =
  loadSource("../src/api/training.ts");
const { TrainingPanel, TrainingResult } = loadSource("../src/components/TrainingPanel.tsx");
const id = "c".repeat(32);
const base = `/analysis/train/artifacts/${id}`;

function fixture(problemType = "classification") {
  const classification = problemType === "classification";
  const metric = classification ? "balanced_accuracy" : "rmse";
  const winnerScore = classification ? 0.81 : 28.4;
  const runnerScore = classification ? 0.78 : 31.2;
  return {
    filename: "dataset.csv", selected_target: classification ? "Outcome" : "Salary", training_id: id,
    message: "Training completed.",
    report: {
      training_status: "complete", target_column: classification ? "Outcome" : "Salary",
      problem_type: problemType, problem_type_source: "inferred", primary_metric: metric,
      primary_metric_direction: classification ? "higher" : "lower", source_dataframe_modified: false,
      split: { usable_rows: 100, training_rows: 80, test_rows: 20, missing_target_rows_removed: 0,
        stratified: classification, preprocessing_decision_scope: "training_split_only" },
      cross_validation: { used_folds: 5, splitter: classification ? "StratifiedKFold" : "KFold" },
      feature_selection: { method: classification ? "SelectPercentile(f_classif)" : "SelectPercentile(f_regression)",
        selected_feature_count: 3, selected_feature_names: ["Age", "Score", "Region_North"] },
      best_model: { model_name: "hist_gradient_boosting", display_name: classification
        ? "Histogram Gradient Boosting Classifier" : "Histogram Gradient Boosting Regressor",
        cross_validation_primary_score: winnerScore, selected_variant: "tuned", beats_dummy_baseline: true },
      holdout_metrics: classification ? { accuracy: 0.82, balanced_accuracy: 0.8, f1_weighted: 0.81,
        class_labels: [0, 1], confusion_matrix: [[9, 1], [3, 7]] }
        : { rmse: 29.1, mae: 20.3, r2: 0.44 },
      model_recommendation: { recommendation_status: "complete", selection_mode: "automatic_recommendation",
        selection_scope: "training_split_only", holdout_data_used: false,
        registered_non_baseline_model_count: 14, selected_model_count: 2, skipped_model_count: 12,
        selected_models: ["hist_gradient_boosting", "logistic_regression"],
        model_decisions: [{ display_name: "Histogram Gradient Boosting", family: "boosting",
          computational_cost: "medium", decision: "selected", reasons: ["captures_mixed_feature_interactions"] },
          { display_name: "RBF Support Vector Classifier", family: "kernel", computational_cost: "high",
            decision: "skipped", reasons: ["candidate_limit_reached"] }] },
      model_selection_rationale: { rationale_status: "complete", summary: [
        "The winner ranked first using training-only cross-validation.",
        "The untouched holdout did not influence the winner."],
        winner: { display_name: classification ? "Histogram Gradient Boosting Classifier" : "Histogram Gradient Boosting Regressor",
          family: "boosting", computational_cost: "medium", selected_variant: "tuned",
          cross_validation_primary_score: winnerScore, cross_validation_standard_deviation: 0.03,
          selected_because: ["highest_validated_primary_score_after_bounded_tuning", "beats_dummy_baseline"] },
        runner_up: { display_name: classification ? "Logistic Regression" : "Ridge Regression",
          cross_validation_primary_score: runnerScore, score_gap_from_winner: Math.abs(winnerScore - runnerScore) },
        baseline_assessment: { display_name: classification ? "Dummy Classifier" : "Dummy Regressor",
          cross_validation_primary_score: classification ? 0.5 : 50, absolute_improvement: classification ? 0.31 : 21.6,
          winner_beats_baseline: true },
        holdout_validation: { evaluation_scope: "untouched_holdout", used_for_model_selection: false, row_count: 20,
          primary_metric: metric, primary_score: classification ? 0.8 : 29.1, generalization_status: "consistent" },
        decision_confidence: { level: "moderate", warnings: ["winner_and_runner_up_are_close"],
          interpretation: "Evidence strength is not a guarantee of future performance." },
        model_ranking: [{ rank: 1, display_name: classification ? "Histogram Gradient Boosting Classifier" : "Histogram Gradient Boosting Regressor",
          family: "boosting", selected_variant: "tuned", cross_validation_primary_score: winnerScore,
          cross_validation_standard_deviation: 0.03 }, { rank: 2,
          display_name: classification ? "Logistic Regression" : "Ridge Regression", family: "linear",
          selected_variant: "untuned", cross_validation_primary_score: runnerScore,
          cross_validation_standard_deviation: 0.04 }] },
      hyperparameter_tuning: { status: "complete", attempted_model_count: 2, successful_model_count: 2,
        failed_model_count: 0, selected_model_name: "hist_gradient_boosting", selected_variant: "tuned",
        candidates: [{ display_name: "Histogram Gradient Boosting", status: "success",
          untuned_primary_score: classification ? 0.79 : 30, tuned_primary_score: winnerScore,
          improvement: classification ? 0.02 : 1.6, best_parameters: { model__max_iter: 200 } }] },
      explainability: { explainability_status: "complete", evaluation_scope: "untouched_holdout",
        used_for_model_selection: false, evaluation_rows_used: 20, sampling_applied: false,
        permutation_importance: { interpretation: "Positive values mean shuffling worsened performance.",
          features: [{ rank: 1, feature: "Age", importance_mean: 0.16,
            importance_standard_deviation: 0.02, importance_direction: "positive" }] },
        native_importance: { status: "unavailable", reason: "Estimator does not expose native importance.", features: [] } },
    },
    artifacts: { artifact_status: "complete", prediction_equivalence_verified: true, verification_rows: 20,
      downloads: [
        { artifact_type: "pipeline", file_name: "model_pipeline.joblib", artifact_url: `${base}/model_pipeline.joblib`, sha256: "a".repeat(64), size_bytes: 2048 },
        { artifact_type: "training_report", file_name: "training_report.json", artifact_url: `${base}/training_report.json`, size_bytes: 4096 },
        { artifact_type: "permutation_importance_chart", file_name: "permutation_importance_chart.png", artifact_url: `${base}/permutation_importance_chart.png`, size_bytes: 1024 },
        { artifact_type: "professional_report_markdown", file_name: "model_report.md", artifact_url: `${base}/model_report.md` },
        { artifact_type: "professional_report_pdf", file_name: "model_report.pdf", artifact_url: `${base}/model_report.pdf` },
        { artifact_type: "eda_target_distribution", file_name: "target_distribution.png", artifact_url: `${base}/target_distribution.png` },
        { artifact_type: "manifest", file_name: "artifact_manifest.json", artifact_url: `${base}/artifact_manifest.json` },
      ] },
  };
}
const jsonResponse = (value, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { "Content-Type": "application/json" },
});
const render = value => renderToStaticMarkup(createElement(TrainingResult, { response: value }));

test("default training request uses automatic bounded recommendations and complete exports", async t => {
  let request;
  t.mock.method(globalThis, "fetch", async (_, init) => { request = JSON.parse(init.body); return jsonResponse(fixture()); });
  await trainModels("dataset.csv", "Outcome", DEFAULT_TRAINING_OPTIONS, new AbortController().signal);
  assert.equal(request.filename, "dataset.csv"); assert.equal(request.target_column, "Outcome");
  assert.equal(Object.hasOwn(request, "requested_models"), false);
  assert.equal(Object.hasOwn(request, "problem_type"), false);
  assert.equal(request.max_recommended_models, 4); assert.equal(request.run_tuning, true);
  assert.equal(request.tuning_iterations, 12); assert.equal(request.tuning_top_models, 2);
  assert.equal(request.include_explanations, true); assert.equal(request.include_explanation_visualizations, true);
  assert.equal(request.include_professional_report, true); assert.deepEqual(request.professional_report_formats, ["markdown", "pdf"]);
  assert.equal(request.export_artifacts, true); assert.equal(request.allow_potential_leakage, false);
});

test("models tuned never exceeds automatically recommended candidates", async t => {
  t.mock.method(globalThis, "fetch", async (_, init) => {
    assert.equal(JSON.parse(init.body).tuning_top_models, 2); return jsonResponse(fixture());
  });
  await trainModels("dataset.csv", "Outcome", { ...DEFAULT_TRAINING_OPTIONS,
    maxRecommendedModels: 2, tuningTopModels: 4 }, new AbortController().signal);
});

test("mismatched, unexported, or malformed training identities are rejected", async t => {
  const response = fixture(); response.selected_target = "Other";
  const mock = t.mock.method(globalThis, "fetch", async () => jsonResponse(response));
  await assert.rejects(trainModels("dataset.csv", "Outcome", DEFAULT_TRAINING_OPTIONS, new AbortController().signal), /does not match/);
  const noId = fixture(); noId.training_id = null; noId.artifacts = null;
  mock.mock.mockImplementation(async () => jsonResponse(noId));
  await assert.rejects(trainModels("dataset.csv", "Outcome", DEFAULT_TRAINING_OPTIONS, new AbortController().signal), /exported run/);
});

test("private holdout data anywhere in a public response is blocked", async t => {
  for (const key of ["X_holdout", "y_holdout", "evaluation_context"]) {
    const response = fixture(); response.report.nested = { [key]: [{ secret: 1 }] };
    t.mock.method(globalThis, "fetch", async () => jsonResponse(response));
    await assert.rejects(trainModels("dataset.csv", "Outcome", DEFAULT_TRAINING_OPTIONS,
      new AbortController().signal), /private evaluation context/);
    t.mock.restoreAll();
  }
  assert.equal(containsPrivateEvaluationData({ evaluation_scope: "untouched_holdout" }), false);
});

test("API validation and leakage review messages are preserved", async t => {
  const mock = t.mock.method(globalThis, "fetch", async () => jsonResponse({ detail: "Potential target leakage requires review." }, 400));
  await assert.rejects(trainModels("dataset.csv", "Outcome", DEFAULT_TRAINING_OPTIONS, new AbortController().signal), /Potential target leakage/);
  mock.mock.mockImplementation(async () => jsonResponse({ detail: [{ msg: "Maximum is 6" }] }, 422));
  await assert.rejects(trainModels("dataset.csv", "Outcome", DEFAULT_TRAINING_OPTIONS, new AbortController().signal), /Maximum is 6/);
});

test("artifact links require the exact training ID, route, safe name and approved type", () => {
  const response = fixture(); assert.equal(trainingDownloads(response).length, 7);
  const unsafe = [
    { artifact_type: "pipeline", file_name: "../model.joblib", artifact_url: `${base}/../model.joblib` },
    { artifact_type: "pipeline", file_name: "model.svg", artifact_url: `${base}/model.svg` },
    { artifact_type: "pipeline", file_name: "model_pipeline.joblib", artifact_url: "https://evil.invalid/model.joblib" },
    { artifact_type: "pipe line", file_name: "model_pipeline.joblib", artifact_url: `${base}/model_pipeline.joblib` },
  ];
  response.artifacts.downloads.push(...unsafe);
  assert.equal(trainingDownloads(response).length, 7);
  response.training_id = "d".repeat(32); assert.equal(trainingDownloads(response).length, 0);
});

test("classification results show rationale, ranking, holdout matrix, explanation and downloads", () => {
  const html = render(fixture());
  for (const text of ["Histogram Gradient Boosting Classifier", "Winner beats the dummy baseline",
    "Logistic Regression", "Dummy Classifier", "Validated model ranking", "Predicted 0",
    "Raw feature permutation importance", "Age", "model_report.pdf", "model_pipeline.joblib",
    "training data only", "private holdout rows"]) assert.ok(html.includes(text), text);
  assert.match(html, /<img/); assert.match(html, /permutation importance chart for Outcome/);
});

test("regression results display lower-is-better metrics without a confusion matrix", () => {
  const html = render(fixture("regression"));
  assert.match(html, /Histogram Gradient Boosting Regressor/); assert.match(html, /29\.1/);
  assert.match(html, /Ridge Regression/); assert.match(html, /rmse/);
  assert.doesNotMatch(html, /Confusion matrix|Predicted 0/);
});

test("partial training and confidence warnings remain explicit", () => {
  const response = fixture(); response.report.training_status = "partial";
  const html = render(response); assert.match(html, /Training completed partially/);
  assert.match(html, /winner and runner up are close/);
});

test("disabled explanation has a reason and creates no chart image", () => {
  const response = fixture(); response.report.explainability = {
    explainability_status: "skipped", reason: "disabled_by_request",
  }; response.artifacts.downloads = response.artifacts.downloads.filter(item => !item.artifact_type.endsWith("_chart"));
  const html = render(response); assert.match(html, /disabled by request/); assert.doesNotMatch(html, /<img/);
});

test("unsafe artifacts never appear as links or image sources", () => {
  const response = fixture(); response.artifacts.downloads = [{ artifact_type: "pipeline",
    file_name: "model_pipeline.joblib", artifact_url: "javascript:alert(1)" }];
  const html = render(response); assert.doesNotMatch(html, /javascript:|<img/);
});

test("training UI defaults to automatic recommendations and explicit safe controls", () => {
  const html = renderToStaticMarkup(createElement(TrainingPanel, {
    filename: "dataset.csv", targetColumn: "Outcome", onStatusChange: () => {},
  }));
  assert.match(html, /Automatic mode is active/); assert.match(html, /4 relevant models/);
  assert.match(html, /Tune leading models/); assert.match(html, /Explain the winner/);
  assert.match(html, /Create PDF and Markdown report/); assert.match(html, /Start model training/);
  assert.match(html, /Allow features flagged as potential target leakage/);
  assert.doesNotMatch(html, /value="true" checked=""[^>]*potential target leakage/);
});

test("workflow enables training only after EDA and results only after training", () => {
  const app = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8");
  const eda = readFileSync(new URL("../src/components/EdaPanel.tsx", import.meta.url), "utf8");
  const training = readFileSync(new URL("../src/components/TrainingPanel.tsx", import.meta.url), "utf8");
  assert.match(app, /trainingComplete \? 4 : edaComplete \? 3 : targetResult \? 2/);
  assert.match(app, /targetResult && edaComplete && <TrainingPanel/);
  assert.match(eda, /onStatusChange\?\.\(false\)/); assert.match(eda, /onStatusChange\?\.\(true\)/);
  assert.match(training, /setResult\(null\); onStatusChange\(false\)/);
  assert.match(training, /setResult\(response\); onStatusChange\(true\)/);
});
