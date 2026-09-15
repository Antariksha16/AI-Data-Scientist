import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { display } from "./EdaPanel";
import { items, LatestRequest, record, words, type DataRecord } from "../api/eda";
import {
  DEFAULT_TRAINING_OPTIONS, trainModels, trainingDownloads,
  type DownloadArtifact, type TrainingOptions, type TrainingResponse,
} from "../api/training";

const humanize = (value: unknown) => String(value ?? "Not available").replaceAll("_", " ");
const percent = (value: unknown) => typeof value === "number" ? `${(value * 100).toFixed(2)}%` : display(value);
const fileSize = (value: number | undefined) => value === undefined ? null
  : value < 1024 ? `${value} B` : value < 1024 ** 2 ? `${(value / 1024).toFixed(1)} KB`
    : `${(value / 1024 ** 2).toFixed(1)} MB`;

function Box({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warning" | "success" }) {
  return <div className={`train-message ${tone}`}>{children}</div>;
}
function Facts({ data, fields }: { data: DataRecord; fields: [string, string][] }) {
  return <dl className="training-facts">{fields.map(([key, title]) => <div key={key}>
    <dt>{title}</dt><dd>{display(data[key])}</dd>
  </div>)}</dl>;
}
function SimpleTable({ caption, rows, columns, empty }: {
  caption: string; rows: DataRecord[]; columns: [string, string, ((value: unknown, row: DataRecord) => string)?][]; empty: string;
}) {
  if (!rows.length) return <p className="empty-note">{empty}</p>;
  return <div className="table-scroll" tabIndex={0} role="region" aria-label={caption}>
    <table><caption>{caption}</caption><thead><tr>{columns.map(([key, title]) =>
      <th scope="col" key={key}>{title}</th>)}</tr></thead><tbody>{rows.map((row, index) =>
        <tr key={`${display(row.model_name || row.feature || row.rank)}-${index}`}>{columns.map(([key, , format]) =>
          <td key={key}>{format ? format(row[key], row) : display(row[key])}</td>)}</tr>)}</tbody></table>
  </div>;
}
function ArtifactLink({ artifact, label }: { artifact: DownloadArtifact; label?: string }) {
  return <a className="artifact-card" href={artifact.artifact_url} download={artifact.file_name}
    target="_blank" rel="noopener noreferrer">
    <strong>{label || humanize(artifact.artifact_type)}</strong>
    <span>{artifact.file_name}{fileSize(artifact.size_bytes) ? ` · ${fileSize(artifact.size_bytes)}` : ""}</span>
  </a>;
}
function ArtifactChart({ artifact, target }: { artifact: DownloadArtifact; target: string }) {
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  return <figure><figcaption>{humanize(artifact.artifact_type)}</figcaption>
    {failed ? <Box tone="warning"><strong>Chart could not be loaded.</strong><p>The numerical explanation and download remain available.</p>
      <button type="button" className="secondary-button" onClick={() => { setAttempt(value => value + 1); setFailed(false); }}>Retry chart</button></Box>
      : <img key={attempt} src={artifact.artifact_url} alt={`${humanize(artifact.artifact_type)} for ${target}`}
          loading="lazy" onError={() => setFailed(true)} />}
    <ArtifactLink artifact={artifact} label="Open / save chart" />
  </figure>;
}

export function TrainingResult({ response }: { response: TrainingResponse }) {
  const report = response.report;
  const best = record(report.best_model);
  const split = record(report.split);
  const crossValidation = record(report.cross_validation);
  const selection = record(report.feature_selection);
  const recommendation = record(report.model_recommendation);
  const rationale = record(report.model_selection_rationale);
  const winner = record(rationale.winner);
  const runnerUp = record(rationale.runner_up);
  const baseline = record(rationale.baseline_assessment);
  const holdout = record(rationale.holdout_validation);
  const confidence = record(rationale.decision_confidence);
  const tuning = record(report.hyperparameter_tuning);
  const explanation = record(report.explainability);
  const permutation = record(explanation.permutation_importance);
  const native = record(explanation.native_importance);
  const artifacts = trainingDownloads(response);
  const charts = artifacts.filter(item => item.artifact_type.endsWith("_chart") && item.file_name.endsWith(".png"));
  const reportArtifacts = artifacts.filter(item => item.artifact_type.startsWith("professional_report_"));
  const coreArtifacts = artifacts.filter(item => ["pipeline", "training_report", "manifest"].includes(item.artifact_type));
  const otherArtifacts = artifacts.filter(item => !charts.includes(item) && !reportArtifacts.includes(item) && !coreArtifacts.includes(item));
  const metric = String(report.primary_metric || "primary metric");
  const statusPartial = report.training_status !== "complete";

  return <section id="training-results" className="training-results" aria-labelledby="training-result-title">
    <div className="winner-card">
      <div><p className="eyebrow">Selected model</p><h3 id="training-result-title">{display(best.display_name)}</h3>
        <p>{humanize(report.problem_type)} · {humanize(best.selected_variant)}</p></div>
      <div className="winner-score"><span>CV {humanize(metric)}</span><strong>{display(best.cross_validation_primary_score)}</strong></div>
    </div>
    {statusPartial && <Box tone="warning"><strong>Training completed partially.</strong>
      <p>Some candidate or tuning attempts failed. The successful winner and available evidence are preserved below.</p></Box>}
    <Box tone={best.beats_dummy_baseline === true ? "success" : "warning"}>
      <strong>{best.beats_dummy_baseline === true ? "Winner beats the dummy baseline" : "Baseline improvement was not established"}</strong>
      <p>The winner was chosen using training-only cross-validation and tuning. The untouched holdout was evaluated afterward.</p>
    </Box>

    <section className="training-section"><h3>Why this model was selected</h3>
      <ul className="training-summary">{words(rationale.summary).map((line, index) => <li key={index}>{line}</li>)}</ul>
      <Facts data={winner} fields={[["family", "Model family"], ["computational_cost", "Computational cost"],
        ["selected_variant", "Selected variant"], ["cross_validation_standard_deviation", "CV standard deviation"]]} />
      <div className="reason-pills">{words(winner.selected_because).map(reason =>
        <span key={reason}>{humanize(reason)}</span>)}</div>
      {Object.keys(runnerUp).length > 0 && <p><strong>Runner-up:</strong> {display(runnerUp.display_name)},
        score {display(runnerUp.cross_validation_primary_score)}; gap from winner {display(runnerUp.score_gap_from_winner)}.</p>}
      <div className="decision-grid">
        <article><h4>Dummy baseline</h4><p>{display(baseline.display_name)} · {display(baseline.cross_validation_primary_score)}</p>
          <p>Absolute improvement: {display(baseline.absolute_improvement)}</p></article>
        <article><h4>Decision confidence</h4><p className="confidence-value">{humanize(confidence.level)}</p>
          <p>{display(confidence.interpretation)}</p></article>
        <article><h4>Untouched holdout</h4><p>{humanize(holdout.generalization_status)}</p>
          <p>{display(holdout.row_count)} rows · {humanize(holdout.primary_metric)} {display(holdout.primary_score)}</p></article>
      </div>
      {words(confidence.warnings).length > 0 && <Box tone="warning"><strong>Confidence warnings</strong>
        <ul>{words(confidence.warnings).map(item => <li key={item}>{humanize(item)}</li>)}</ul></Box>}
    </section>

    <section className="training-section"><h3>Model comparison</h3>
      <p>Higher is better for classification scores; lower is better for regression errors. Models below were compared using the same training folds.</p>
      <SimpleTable caption="Validated model ranking" rows={items(rationale.model_ranking).map(record)}
        columns={[["rank", "Rank"], ["display_name", "Model"], ["family", "Family"],
          ["selected_variant", "Variant"], ["cross_validation_primary_score", `CV ${humanize(metric)}`],
          ["cross_validation_standard_deviation", "CV deviation"]]} empty="No ranked alternatives were returned." />
      <details className="training-details"><summary>Why models were recommended or skipped</summary>
        <Facts data={recommendation} fields={[["selection_mode", "Selection mode"],
          ["selection_scope", "Recommendation scope"], ["registered_non_baseline_model_count", "Registered models"],
          ["selected_model_count", "Models trained"], ["skipped_model_count", "Models skipped"],
          ["holdout_data_used", "Holdout used for recommendation"]]} />
        <SimpleTable caption="Dataset-aware recommendation decisions" rows={items(recommendation.model_decisions).map(record)}
          columns={[["display_name", "Model"], ["family", "Family"], ["computational_cost", "Cost"],
            ["decision", "Decision"], ["reasons", "Reason", value => humanize(display(value))]]}
          empty="Recommendation decisions are unavailable." />
      </details>
    </section>

    <section className="training-section"><h3>Training and tuning</h3>
      <Facts data={split} fields={[["usable_rows", "Usable rows"], ["training_rows", "Training rows"],
        ["test_rows", "Untouched test rows"], ["missing_target_rows_removed", "Missing targets removed"],
        ["stratified", "Stratified split"], ["preprocessing_decision_scope", "Preprocessing scope"]]} />
      <Facts data={crossValidation} fields={[["used_folds", "Cross-validation folds"], ["splitter", "Fold method"]]} />
      <Facts data={selection} fields={[["method", "Feature-selection method"], ["selected_feature_count", "Selected features"],
        ["selected_feature_names", "Final transformed features"]]} />
      <details className="training-details"><summary>Hyperparameter tuning details</summary>
        <Facts data={tuning} fields={[["status", "Tuning status"], ["attempted_model_count", "Models attempted"],
          ["successful_model_count", "Successful"], ["failed_model_count", "Failed"],
          ["selected_model_name", "Selected model key"], ["selected_variant", "Final variant"]]} />
        <SimpleTable caption="Tuning candidates" rows={items(tuning.candidates).map(record)}
          columns={[["display_name", "Model"], ["status", "Status"], ["untuned_primary_score", "Untuned score"],
            ["tuned_primary_score", "Tuned score"], ["improvement", "Improvement"],
            ["best_parameters", "Best parameters"]]} empty="No tuning candidates were returned." />
      </details>
    </section>

    <section className="training-section"><h3>Untouched holdout evaluation</h3>
      <p>This result was measured after model selection and did not choose the winner.</p>
      <Facts data={record(report.holdout_metrics)} fields={Object.keys(record(report.holdout_metrics))
        .filter(key => !["confusion_matrix", "class_labels"].includes(key)).map(key => [key, humanize(key)])} />
      {items(record(report.holdout_metrics).confusion_matrix).length > 0 && <SimpleTable
        caption="Confusion matrix (actual rows, predicted columns)"
        rows={items(record(report.holdout_metrics).confusion_matrix).map((values, index) => ({
          actual: items(record(report.holdout_metrics).class_labels)[index],
          ...Object.fromEntries(items(values).map((value, column) => [`predicted_${column}`, value])),
        }))}
        columns={[["actual", "Actual class"], ...items(record(report.holdout_metrics).class_labels)
          .map((value, index): [string, string] => [`predicted_${index}`, `Predicted ${display(value)}`])]}
        empty="No confusion matrix is available." />}
    </section>

    <section className="training-section"><h3>Model explanation</h3>
      {explanation.explainability_status !== "complete" ? <Box tone="warning">Explainability was {humanize(explanation.explainability_status)}:
        {` ${humanize(explanation.reason)}`}</Box> : <>
        <Facts data={explanation} fields={[["evaluation_scope", "Evaluation data"], ["used_for_model_selection", "Used to select winner"],
          ["evaluation_rows_used", "Rows explained"], ["sampling_applied", "Sampling applied"]]} />
        <p>{display(permutation.interpretation)}</p>
        <SimpleTable caption="Raw feature permutation importance" rows={items(permutation.features).map(record)}
          columns={[["rank", "Rank"], ["feature", "Raw feature"], ["importance_mean", "Mean importance"],
            ["importance_standard_deviation", "Deviation"], ["importance_direction", "Direction"]]}
          empty="No permutation importance was returned." />
        <details className="training-details"><summary>Native estimator importance</summary>
          {native.status === "complete" ? <SimpleTable caption="Selected transformed-feature importance"
            rows={items(native.features).map(record)} columns={[["rank", "Rank"], ["feature", "Feature"],
              ["importance", "Importance"], ["effect_direction", "Direction"]]}
            empty="No native importance values were returned." />
            : <p>{display(native.reason)}</p>}
        </details>
        {charts.length > 0 && <div className="training-chart-grid">{charts.map(chart =>
          <ArtifactChart key={chart.artifact_type} artifact={chart} target={response.selected_target} />)}</div>}
      </>}
    </section>

    <section className="training-section"><h3>Downloads</h3>
      <p>These files belong to training run <code>{response.training_id}</code>. Keep the manifest with the pipeline.</p>
      {reportArtifacts.length > 0 && <><h4>Professional model report</h4><div className="artifact-grid">
        {reportArtifacts.map(item => <ArtifactLink artifact={item} key={item.artifact_type} />)}</div></>}
      <h4>Model bundle</h4><div className="artifact-grid">{coreArtifacts.map(item =>
        <ArtifactLink artifact={item} key={item.artifact_type} label={item.artifact_type === "pipeline" ? "Trained model pipeline" : undefined} />)}</div>
      {otherArtifacts.length > 0 && <details className="training-details"><summary>Supporting report charts</summary>
        <div className="artifact-grid">{otherArtifacts.map(item => <ArtifactLink artifact={item} key={item.artifact_type} />)}</div>
      </details>}
      <Box tone="warning"><strong>Load model files only from trusted training runs.</strong>
        <p>The manifest and SHA-256 metadata help detect accidental changes. A `.joblib` model is executable serialized content.</p></Box>
    </section>
    <Box><strong>Training boundaries</strong><p>Recommendation, preprocessing, feature selection and tuning used training data only.
      Raw private holdout rows are not included in this page or its public report.</p></Box>
  </section>;
}

export function TrainingPanel({ filename, targetColumn, onStatusChange, onResultChange }: {
  filename: string;
  targetColumn: string;
  onStatusChange: (complete: boolean) => void;
  onResultChange?: (response: TrainingResponse | null) => void;
}) {
  const latest = useRef(new LatestRequest());
  const [options, setOptions] = useState<TrainingOptions>(DEFAULT_TRAINING_OPTIONS);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [result, setResult] = useState<TrainingResponse | null>(null);
  useEffect(() => { const requests = latest.current; return () => requests.cancel(); }, []);
  const durationNote = useMemo(() => options.includeProfessionalReport
    ? "Comparison, tuning, explanations, charts and reports can take a few minutes."
    : "Comparison and tuning can take a few minutes.", [options.includeProfessionalReport]);
  const update = <K extends keyof TrainingOptions>(key: K, value: TrainingOptions[K]) =>
    setOptions(current => ({ ...current, [key]: value }));

  async function startTraining() {
    const request = latest.current.start();
    setBusy(true); setError(null); setNotice(null); setResult(null); onStatusChange(false); onResultChange?.(null);
    try {
      const response = await trainModels(filename, targetColumn, options, request.signal);
      if (request.isCurrent()) { setResult(response); onStatusChange(true); onResultChange?.(response); }
    } catch (caught) {
      if (request.isCurrent()) setError(caught instanceof Error ? caught.message : "Training failed. Please retry.");
    } finally { if (request.isCurrent()) setBusy(false); }
  }

  return <section id="train-models" className="panel train-panel" aria-labelledby="train-title">
    <div className="section-heading"><span className="section-number">04</span><div>
      <h2 id="train-title">Train and compare models</h2>
      <p>Use dataset-aware recommendations, training-only validation and an untouched final test.</p>
    </div></div>
    <div className="training-config">
      <div className="training-target"><span className="eyebrow">Prediction target</span><strong>{targetColumn}</strong><span>{filename}</span></div>
      <fieldset><legend>Automatic candidate limit</legend><select aria-label="Maximum recommended models"
        value={options.maxRecommendedModels} disabled={busy}
        onChange={event => {
          const limit = Number(event.target.value);
          setOptions(current => ({ ...current, maxRecommendedModels: limit,
            tuningTopModels: Math.min(current.tuningTopModels, limit) }));
        }}>
        {[2, 3, 4, 5, 6].map(value => <option value={value} key={value}>{value} relevant models</option>)}</select>
        <small>The recommender chooses diverse, suitable model families using training rows only.</small></fieldset>
      <div className="training-switches">
        <label><input type="checkbox" checked={options.runTuning} disabled={busy}
          onChange={event => update("runTuning", event.target.checked)} /><span><strong>Tune leading models</strong><small>Bounded randomized search</small></span></label>
        <label><input type="checkbox" checked={options.includeExplanations} disabled={busy}
          onChange={event => update("includeExplanations", event.target.checked)} /><span><strong>Explain the winner</strong><small>Untouched holdout permutation analysis</small></span></label>
        <label><input type="checkbox" checked={options.includeProfessionalReport} disabled={busy}
          onChange={event => update("includeProfessionalReport", event.target.checked)} /><span><strong>Create PDF and Markdown report</strong><small>Includes selection reasons and charts</small></span></label>
      </div>
      <details className="training-details"><summary>Advanced bounded options</summary>
        <div className="advanced-grid">
          <label>Tuning iterations<select value={options.tuningIterations} disabled={busy || !options.runTuning}
            onChange={event => update("tuningIterations", Number(event.target.value))}>
            {[6, 12, 24, 36].map(value => <option value={value} key={value}>{value}</option>)}</select></label>
          <label>Models to tune<select value={options.tuningTopModels} disabled={busy || !options.runTuning}
            onChange={event => update("tuningTopModels", Number(event.target.value))}>
            {[1, 2, 3, 4].filter(value => value <= options.maxRecommendedModels)
              .map(value => <option value={value} key={value}>{value}</option>)}</select></label>
        </div>
        <label className="leakage-consent"><input type="checkbox" checked={options.allowPotentialLeakage} disabled={busy}
          onChange={event => update("allowPotentialLeakage", event.target.checked)} /><span><strong>Allow features flagged as potential target leakage</strong>
          <small>Leave disabled unless you reviewed the feature meaning and timing. Enabling this acknowledges the risk; it does not prove the features are safe.</small></span></label>
      </details>
      <Box><strong>Automatic mode is active.</strong><p>The system will not train all registered algorithms. It will compare up to {options.maxRecommendedModels} relevant models plus a dummy baseline.</p></Box>
      <div className="training-actions"><button className="primary-button" type="button" disabled={busy} onClick={startTraining}>
        {busy ? "Training models…" : result ? "Train again" : "Start model training"}</button>
        {busy && <button className="secondary-button" type="button" onClick={() => {
          latest.current.cancel(); setBusy(false); setNotice("Stopped waiting for this run. The backend may continue training and creating artifacts.");
        }}>Cancel request</button>}</div>
    </div>
    <div aria-live="polite" role="status">{busy && <Box><strong>Training is in progress.</strong><p>{durationNote} Keep both local servers running.</p></Box>}
      {notice && <Box>{notice}</Box>}{result && <p className="sr-only">Training complete. Results and downloads are below.</p>}</div>
    {error && <div className="eda-error" role="alert"><strong>Training could not finish.</strong><p>{error}</p>
      <p>Your dataset, target and settings remain selected. Review leakage warnings or adjust bounded options, then retry.</p></div>}
    {!busy && !result && !error && <Box>Start training when you are ready. The untouched test set will not be used to recommend, tune or select the model.</Box>}
    {result && <TrainingResult response={result} />}
  </section>;
}
