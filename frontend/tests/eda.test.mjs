import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadSource } from "./load-source.mjs";

const { runEda, edaArtifactUrl, LatestRequest } = loadSource("../src/api/eda.ts");
const { EdaReportView, EdaPanel, display } = loadSource("../src/components/EdaPanel.tsx");
const id = "a".repeat(32);
// Mirrors verified generate_eda_report and public EDA endpoint field names.
function fixture(target = "Survived", type = "classification") {
  return {
    filename: "example.csv", selected_target: target, analysis_id: id,
    report: {
      target_column: target, report_status: "complete", automatic_data_modification_performed: false,
      summary: { rows: 100, candidate_features: 2, total_missing_cells: 3, duplicate_rows: 0, problem_type: type },
      target_statistics: { problem_type: type, problem_type_source: "inferred", observed_target_rows: 100,
        missing_target_rows: 0, classification_statistics: type === "classification" ? {
          class_count: 2, imbalance_level: "balanced", recommended_metrics: ["balanced_accuracy"],
          class_distribution: [{ class_value: "Approved", count: 60, percentage: 60 },
            { class_value: "Rejected", count: 40, percentage: 40 }],
        } : null, regression_statistics: type === "regression" ? {
          mean: 45.25, median: 40, minimum: 1, maximum: 150, standard_deviation: 8,
          distribution_shape: "right_skewed", outlier_count: 2, recommended_metrics: ["rmse", "mae"],
        } : null },
      dataset_overview: { missingness: { column_summary: [{ column: "Age", missing_values: 3, missing_percentage: 3 }] } },
      numerical_statistics: { numerical_statistics: { Age: { mean: 32, median: 31, minimum: 1,
        maximum: 80, standard_deviation: 10, outlier_count: 1 } } },
      categorical_statistics: { categorical_statistics: { Region: { unique_values: 2,
        most_frequent_value: "North", missing_percentage: 0, cardinality_level: "low", recommended_encoding: "one_hot" } } },
      feature_target_relationships: { ranked_features: [{ feature: "Age", analysis_method: "eta",
        primary_score: 0.24, relationship_strength: "weak", potential_leakage: false }] },
      preprocessing_context: { cleaning_plan: { drop_columns: { RecordID: ["identifier"] },
        review_columns: {}, kept_features: ["Age", "Region"] }, outlier_report: { rows_removed: 0,
        robust_scaler_features: ["Age"] }, leakage_report: { potential_leakage_count: 0 } },
      insights: { executive_summary: "Review data before training.", prioritized_insights: [
        { severity: "medium", title: "Missing ages", explanation: "Three observations lack age.",
          recommendation: "Fit an imputer on training rows only.", evidence: { missing_count: 3 } },
      ] },
      visualizations: { visualization_status: "complete", generated_plot_count: 1, skipped_plot_count: 1,
        plot_error_count: 0, generated_plots: [{ plot_id: "target_distribution", title: "Target distribution",
          file_name: "target_distribution.png", artifact_url: `/analysis/eda/artifacts/${id}/target_distribution.png` }],
        skipped_plots: [{ plot_id: "datetime_overview", reason: "No datetime feature." }],
        sampling: { sampling_performed: false, plotted_rows: 100 }, plot_errors: [] },
      section_status: { dataset_overview: "success", target_statistics: "success" },
      failed_sections: [], warnings: [], recommendations: ["Review missing values."],
      limitations: ["Association does not prove causation."],
    },
  };
}
const render = (response, chartsRequested = true) => renderToStaticMarkup(createElement(EdaReportView, { response, chartsRequested }));
const jsonResponse = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });

test("EDA posts selected filename and target, all sections and bounded PNG settings", async t => {
  const signal = new AbortController().signal;
  let received;
  t.mock.method(globalThis, "fetch", async (url, init) => { received = { url, init }; return jsonResponse(fixture()); });
  await runEda("example.csv", "Survived", true, signal);
  assert.equal(received.url, "http://127.0.0.1:8000/analysis/eda");
  assert.equal(received.init.method, "POST");
  assert.equal(received.init.signal, signal);
  assert.deepEqual(JSON.parse(received.init.body), { filename: "example.csv", target_column: "Survived",
    include_feature_relationships: true, include_preprocessing_context: true, include_insights: true,
    include_visualizations: true, visualization_image_format: "png", visualization_dpi: 120 });
});

test("statistics-only requests disable plots and preserve backend type inference", async t => {
  t.mock.method(globalThis, "fetch", async (_, init) => {
    const payload = JSON.parse(init.body);
    assert.equal(payload.include_visualizations, false);
    assert.equal(payload.target_column, "Fare");
    assert.equal(Object.hasOwn(payload, "problem_type"), false);
    const data = fixture("Fare", "regression"); data.analysis_id = null;
    return jsonResponse(data);
  });
  const response = await runEda("example.csv", "Fare", false, new AbortController().signal);
  assert.equal(response.report.summary.problem_type, "regression");
});

test("mismatched dataset or target reports are rejected", async t => {
  t.mock.method(globalThis, "fetch", async () => jsonResponse(fixture("Fare", "regression")));
  await assert.rejects(runEda("example.csv", "Survived", true, new AbortController().signal), /does not match/);
});

test("backend validation and not-found details reach the error view", async t => {
  const mock = t.mock.method(globalThis, "fetch", async () => jsonResponse({ detail: [
    { msg: "Target is required" }, { msg: "Invalid setting" },
  ] }, 422));
  await assert.rejects(runEda("example.csv", "Survived", true, new AbortController().signal), /Target is required Invalid setting/);
  mock.mock.mockImplementation(async () => jsonResponse({ detail: "Uploaded dataset was not found." }, 404));
  await assert.rejects(runEda("example.csv", "Survived", true, new AbortController().signal), /was not found/);
});

test("network, non-JSON and malformed JSON failures are actionable", async t => {
  const mock = t.mock.method(globalThis, "fetch", async () => { throw new TypeError("Network down"); });
  await assert.rejects(runEda("example.csv", "Survived", true, new AbortController().signal), /backend is unavailable/);
  mock.mock.mockImplementation(async () => new Response("<html>proxy</html>", { status: 200 }));
  await assert.rejects(runEda("example.csv", "Survived", true, new AbortController().signal), /expected JSON/);
  mock.mock.mockImplementation(async () => new Response("{", { headers: { "Content-Type": "application/json" } }));
  await assert.rejects(runEda("example.csv", "Survived", true, new AbortController().signal), /unreadable JSON/);
});

test("cancellation is not mislabeled as a backend connection error", async t => {
  const controller = new AbortController(); controller.abort();
  t.mock.method(globalThis, "fetch", async () => { throw new DOMException("Aborted", "AbortError"); });
  await assert.rejects(runEda("example.csv", "Survived", true, controller.signal), { name: "AbortError" });
});

test("late completion cannot commit after cancel, rerun, or selection invalidation", async () => {
  const guard = new LatestRequest(); const old = guard.start();
  let complete;
  let committed = false;
  const pending = new Promise(resolve => { complete = resolve; }).then(() => {
    if (old.isCurrent()) committed = true;
  });
  const current = guard.start();
  assert.equal(old.signal.aborted, true); assert.equal(current.isCurrent(), true);
  complete(); await pending; assert.equal(committed, false);
  guard.cancel(); assert.equal(current.isCurrent(), false);
  assert.equal(current.signal.aborted, true);
});

test("chart URLs are restricted to the current run and safe PNG names", () => {
  const response = fixture(); const plot = response.report.visualizations.generated_plots[0];
  assert.equal(edaArtifactUrl(response, plot), `http://127.0.0.1:8000/analysis/eda/artifacts/${id}/target_distribution.png`);
  for (const url of ["javascript:alert(1)", "https://example.org/p.png", "//example.org/p.png",
    `/analysis/eda/artifacts/${"b".repeat(32)}/target_distribution.png`, `${plot.artifact_url}?redirect=1`]) {
    assert.equal(edaArtifactUrl(response, { ...plot, artifact_url: url }), null);
  }
  for (const name of ["../secret.png", "..%2Fsecret.png", "chart.svg", "chart\\file.png"]) {
    assert.equal(edaArtifactUrl(response, { ...plot, file_name: name }), null);
  }
  assert.equal(edaArtifactUrl({ ...response, analysis_id: null }, plot), null);
});

test("classification report renders actual classes, missingness, statistics and insights", () => {
  const html = render(fixture());
  for (const text of ["Approved", "Rejected", "Class imbalance", "Missing ages", "Age", "North",
    "Numerical descriptive statistics", "Ranked feature–target associations", "training rows only",
    "This EDA run did not modify your dataset", "Outlier assessment"]) assert.ok(html.includes(text), text);
  assert.match(html, /alt="Target distribution for target Survived"/);
  assert.match(html, /No datetime feature/);
});

test("regression report shows continuous target statistics, not classification fields", () => {
  const html = render(fixture("Fare", "regression"));
  assert.match(html, /45\.25/); assert.match(html, /rmse, mae/);
  assert.match(html, /Target outliers flagged/);
  assert.doesNotMatch(html, /Target class distribution|Class imbalance/);
});

test("partial reports preserve completed sections and expose error and skip states", () => {
  const response = fixture();
  response.report.report_status = "partial"; response.report.target_statistics = null;
  response.report.failed_sections = ["target_statistics"];
  response.report.section_errors = [{ section: "target_statistics", error_type: "ValueError", message: "Invalid target distribution" }];
  response.report.section_status.target_statistics = "failed";
  const html = render(response);
  assert.match(html, /Some sections are incomplete/); assert.match(html, /Missing ages/);
  assert.match(html, /Target statistics are unavailable/); assert.match(html, /Invalid target distribution/);
});

test("missing optional sections are not reported as healthy or fabricated zeros", () => {
  const response = { filename: "sparse.csv", selected_target: "Goal", analysis_id: null,
    report: { target_column: "Goal", report_status: "partial" } };
  const html = render(response);
  assert.match(html, /Not available/); assert.match(html, /Cleaning recommendations are unavailable/);
  assert.doesNotMatch(html, /This EDA run did not modify/);
  assert.match(html, /modification status was not confirmed/);
});

test("datetime summaries and plot failure reasons remain visible", () => {
  const response = fixture();
  response.report.datetime_statistics = { datetime_statistics: { EventDate: {
    earliest_datetime: "2025-01-01", latest_datetime: "2025-12-31", time_span_days: 364,
    invalid_datetime_values: 2, detection_method: "native_datetime_dtype",
  } } };
  response.report.visualizations.visualization_status = "partial";
  response.report.visualizations.plot_error_count = 1;
  response.report.visualizations.plot_errors = [{ plot_id: "missingness", message: "Image rendering failed" }];
  const html = render(response);
  assert.match(html, /EventDate/); assert.match(html, /2025-12-31/);
  assert.match(html, /Image rendering failed/);
});

test("charts disabled means no image requests; unsafe links never become image sources", () => {
  const response = fixture();
  assert.doesNotMatch(render(response, false), /<img/);
  assert.match(render(response, false), /Charts were turned off/);
  response.report.visualizations.generated_plots[0].artifact_url = "https://evil.invalid/data";
  const html = render(response);
  assert.doesNotMatch(html, /<img|evil\.invalid/);
  assert.match(html, /missing or unsafe/);
});

test("dataset and insight text is escaped and never executed as markup", () => {
  const response = fixture(); response.selected_target = '<img src=x onerror="alert(1)">';
  response.report.insights.prioritized_insights[0].title = "<script>alert(1)</script>";
  const html = render(response);
  assert.match(html, /&lt;script&gt;/); assert.doesNotMatch(html, /<script>|<img src=x/);
});

test("large schemas use a bounded table with explicit pagination", () => {
  const response = fixture();
  response.report.numerical_statistics.numerical_statistics = Object.fromEntries(
    Array.from({ length: 45 }, (_, index) => [`feature_${index}`, { mean: index }]));
  const html = render(response);
  assert.match(html, /Page 1 of 3/); assert.match(html, /45 rows/);
  assert.match(html, /feature_14/); assert.doesNotMatch(html, /feature_15/);
});

test("formatting preserves zero and false while treating null or nonfinite as missing", () => {
  assert.equal(display(0), "0"); assert.equal(display(false), "No");
  assert.equal(display(null), "Not available"); assert.equal(display(Infinity), "Not available");
});

test("initial Explore is an explicit action and contains accessible controls", () => {
  const html = renderToStaticMarkup(createElement(EdaPanel, { filename: "example.csv", targetColumn: "Survived" }));
  assert.match(html, /Run EDA/); assert.match(html, /Include charts/);
  assert.match(html, /aria-live="polite"/); assert.match(html, /not a cleaned dataset or a trained model/);
});

test("selection changes invalidate setup and unmount the prior EDA report", () => {
  const app = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8");
  const panel = readFileSync(new URL("../src/components/EdaPanel.tsx", import.meta.url), "utf8");
  assert.match(app, /function chooseTarget[\s\S]*?invalidateSelection\(\)/);
  assert.match(app, /function resetForFile[\s\S]*?invalidateSelection\(\)/);
  assert.match(app, /if \(request\.isCurrent\(\)\) setTargetResult/);
  assert.match(app, /targetResult && <EdaPanel/); assert.match(panel, /return \(\) => requests\.cancel\(\)/);
  assert.match(panel, /if \(request\.isCurrent\(\)\) \{ setResult/);
  assert.match(app, /disabled=\{index > activeStep\}/);
});

for (const problemType of ["classification", "regression"]) {
  test(`real EDA module fixture renders ${problemType} statistics and artifact links without mutation`, () => {
    const response = JSON.parse(readFileSync(new URL(`./fixtures/eda-${problemType}.json`, import.meta.url), "utf8"));
    const original = JSON.stringify(response);
    const html = render(response);
    assert.match(html, /synthetic\.csv/);
    assert.match(html, /Measurement/); assert.match(html, /ObservedAt/);
    assert.ok(html.includes(problemType));
    assert.ok(html.includes(display(response.report.numerical_statistics.numerical_statistics.Measurement.mean)));
    const plots = response.report.visualizations.generated_plots;
    assert.ok(plots.length > 0);
    assert.equal((html.match(/<img /g) || []).length, plots.length);
    for (const plot of plots) assert.ok(html.includes(edaArtifactUrl(response, plot)));
    assert.equal(JSON.stringify(response), original);
    assert.doesNotMatch(original, /\/tmp\/|file_path|X_holdout|y_holdout/);
  });
}
