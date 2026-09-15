import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  type DataRecord, type EdaResponse, edaArtifactUrl, items,
  LatestRequest, record, runEda, words,
} from "../api/eda";

export function display(value: unknown): string {
  if (value === null || value === undefined) return "Not available";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return Number.isFinite(value)
    ? new Intl.NumberFormat("en", { maximumFractionDigits: 4 }).format(value) : "Not available";
  if (Array.isArray(value)) return value.length ? value.map(display).join(", ") : "None reported";
  if (typeof value === "object") return Object.entries(record(value))
    .map(([key, item]) => `${label(key)}: ${display(item)}`).join(" · ") || "None reported";
  return String(value);
}
const label = (value: string) => value.replaceAll("_", " ");

function Notice({ children, warning = false }: { children: ReactNode; warning?: boolean }) {
  return <p className={`eda-note ${warning ? "eda-note-warning" : ""}`}>{children}</p>;
}

function Facts({ data, fields }: { data: DataRecord; fields: [string, string][] }) {
  return <dl className="eda-facts">{fields.map(([key, title]) => <div key={key}>
    <dt>{title}</dt><dd>{display(data[key])}</dd>
  </div>)}</dl>;
}

function TextList({ values, empty = "None reported." }: { values: unknown; empty?: string }) {
  const list = items(values);
  return list.length ? <ul className="eda-list">{list.map((item, index) =>
    <li key={index}>{display(item)}</li>)}</ul> : <p className="empty-note">{empty}</p>;
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return <section className="eda-section"><h3>{title}</h3>{children}</section>;
}

/** Paginate large schemas; never silently truncate the supplied report rows. */
function ReportTable({ caption, rows, columns, empty }: {
  caption: string; rows: DataRecord[]; columns: [string, string][]; empty: string;
}) {
  const [page, setPage] = useState(0);
  const pageSize = 15;
  const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
  const currentPage = Math.min(page, pageCount - 1);
  if (!rows.length) return <p className="empty-note">{empty}</p>;
  return <>
    <div className="table-scroll" tabIndex={0} role="region" aria-label={caption}>
      <table><caption>{caption}</caption><thead><tr>
        {columns.map(([key, title]) => <th scope="col" key={key}>{title}</th>)}
      </tr></thead><tbody>{rows.slice(currentPage * pageSize, (currentPage + 1) * pageSize)
        .map((row, index) => <tr key={index}>{columns.map(([key]) =>
          <td key={key}>{display(row[key])}</td>)}</tr>)}</tbody></table>
    </div>
    {pageCount > 1 && <div className="eda-pagination">
      <button type="button" className="secondary-button" disabled={currentPage === 0}
        onClick={() => setPage(currentPage - 1)}>Previous</button>
      <span>Page {currentPage + 1} of {pageCount} · {rows.length} rows</span>
      <button type="button" className="secondary-button" disabled={currentPage + 1 >= pageCount}
        onClick={() => setPage(currentPage + 1)}>Next</button>
    </div>}
  </>;
}

function namedRows(value: unknown): DataRecord[] {
  return Object.entries(record(value)).map(([column, data]) => ({ ...record(data), column }));
}

function Chart({ response, plot }: { response: EdaResponse; plot: DataRecord }) {
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const url = edaArtifactUrl(response, plot);
  const title = typeof plot.title === "string" ? plot.title : label(String(plot.plot_id || "EDA chart"));
  return <figure className="eda-chart">
    <figcaption>{title}</figcaption>
    {!url ? <Notice warning>The chart link was missing or unsafe and was not opened.</Notice>
      : failed ? <div role="status"><Notice warning>This chart could not be loaded. The statistical report is still available.</Notice>
        <button className="secondary-button" type="button" onClick={() => {
          setAttempt(attempt + 1); setFailed(false);
        }}>Retry chart</button></div>
      : <img key={attempt} src={url} alt={`${title} for target ${response.selected_target}`}
          loading="lazy" onError={() => setFailed(true)} />}
    {url && <a className="eda-link" href={url} target="_blank" rel="noopener noreferrer">
      Open / save PNG <span className="sr-only">— {title}, opens a new tab</span>
    </a>}
  </figure>;
}

export function EdaReportView({ response, chartsRequested }: { response: EdaResponse; chartsRequested: boolean }) {
  const report = response.report;
  const overview = record(report.dataset_overview);
  const summary = record(report.summary);
  const target = record(report.target_statistics);
  const classification = record(target.classification_statistics);
  const regression = record(target.regression_statistics);
  const numerical = record(record(report.numerical_statistics).numerical_statistics);
  const categorical = record(record(report.categorical_statistics).categorical_statistics);
  const relationships = record(report.feature_target_relationships);
  const context = record(report.preprocessing_context);
  const cleaning = record(context.cleaning_plan);
  const insights = record(report.insights);
  const visualizations = record(report.visualizations);
  const statuses = Object.entries(record(report.section_status));
  const failures = words(report.failed_sections);
  const partial = report.report_status !== "complete" || failures.length > 0
    || words(report.partial_sections).length > 0;

  return <div className="eda-report">
    <div className="eda-report-heading">
      <div><p className="eyebrow">Your exploration report</p><h3>{response.selected_target}</h3>
        <p>{response.filename} · {display(summary.problem_type || target.problem_type)}</p></div>
      <span className={`eda-badge ${partial ? "warning" : "success"}`}>{label(display(report.report_status))}</span>
    </div>
    {partial && <Notice warning>Some sections are incomplete. Completed findings remain available below.
      {failures.length > 0 && ` Failed sections: ${failures.map(label).join(", ")}.`}</Notice>}
    <Facts data={summary} fields={[["rows", "Dataset rows"], ["candidate_features", "Raw candidate features"],
      ["total_missing_cells", "Missing cells"], ["duplicate_rows", "Duplicate rows"]]} />
    <Notice>{report.automatic_data_modification_performed === false
      ? "This EDA run did not modify your dataset. Findings are recommendations, not approved cleaning or training decisions."
      : "Dataset-modification status was not confirmed as false by this report. Review the backend report before continuing."}</Notice>

    <Section title="Prioritized insights">
      {typeof insights.executive_summary === "string" && <p>{insights.executive_summary}</p>}
      {items(insights.prioritized_insights).length ? <div className="eda-insights">
        {items(insights.prioritized_insights).map((value, index) => {
          const insight = record(value);
          const severity = words([insight.severity])[0] || "info";
          return <article className="eda-insight" key={index}>
            <span className={`eda-badge ${["critical", "high", "medium"].includes(severity) ? "warning" : "neutral"}`}>{severity}</span>
            <h4>{display(insight.title)}</h4><p>{display(insight.explanation)}</p>
            <p className="eda-action"><strong>Recommended action:</strong> {display(insight.recommendation)}</p>
            {Object.keys(record(insight.evidence)).length > 0 && <details><summary>Supporting evidence</summary>
              <Facts data={record(insight.evidence)} fields={Object.keys(record(insight.evidence)).map(key => [key, label(key)])} />
            </details>}
          </article>;
        })}</div> : <p className="empty-note">No insight items are available in this report.</p>}
      {insights.insights_truncated === true && <Notice>The backend limited the displayed insights.
        Returned: {display(insights.insight_count)}; generated: {display(insights.generated_insight_count)}.</Notice>}
    </Section>

    <Section title="Target profile">
      {!report.target_statistics ? <Notice warning>Target statistics are unavailable. Check section status below.</Notice>
        : <>
          <Facts data={target} fields={[["problem_type", "Problem type"], ["problem_type_source", "Determined by"],
            ["observed_target_rows", "Observed rows"], ["missing_target_rows", "Missing target rows"]]} />
          {target.problem_type === "classification" && <>
            <Facts data={classification} fields={[["class_count", "Classes"], ["imbalance_level", "Class imbalance"],
              ["recommended_metrics", "Suggested evaluation metrics"]]} />
            <ReportTable caption="Target class distribution" rows={items(classification.class_distribution).map(record)}
              columns={[["class_value", "Class"], ["count", "Rows"], ["percentage", "Percentage (%)"]]}
              empty="No class distribution is available." />
            {classification.classes_truncated === true && <Notice>Only the backend’s leading classes are shown.</Notice>}
          </>}
          {target.problem_type === "regression" && <Facts data={regression} fields={[["mean", "Mean"],
            ["median", "Median"], ["minimum", "Minimum"], ["maximum", "Maximum"],
            ["standard_deviation", "Standard deviation"], ["distribution_shape", "Distribution"],
            ["outlier_count", "Target outliers flagged"], ["recommended_metrics", "Suggested evaluation metrics"]]} />}
        </>}
    </Section>

    <Section title="Missingness & feature statistics">
      <ReportTable caption="Columns with missing values" rows={items(record(overview.missingness).column_summary).map(record)}
        columns={[["column", "Column"], ["missing_values", "Missing rows"], ["missing_percentage", "Missing (%)"]]}
        empty={report.dataset_overview ? "No columns with missing values were reported." : "Missingness statistics are unavailable."} />
      <details className="eda-details" open><summary>Numerical features</summary>
        <ReportTable caption="Numerical descriptive statistics" rows={namedRows(numerical)}
          columns={[["column", "Feature"], ["mean", "Mean"], ["median", "Median"],
            ["minimum", "Min"], ["maximum", "Max"], ["standard_deviation", "Std. deviation"], ["outlier_count", "Outliers"]]}
          empty={report.numerical_statistics ? "No numerical features were reported." : "Numerical statistics are unavailable."} />
      </details>
      <details className="eda-details"><summary>Categorical features</summary>
        <ReportTable caption="Categorical descriptive statistics" rows={namedRows(categorical)}
          columns={[["column", "Feature"], ["unique_values", "Unique values"], ["most_frequent_value", "Most common"],
            ["missing_percentage", "Missing (%)"], ["cardinality_level", "Cardinality"], ["recommended_encoding", "Suggested encoding"]]}
          empty={report.categorical_statistics ? "No categorical features were reported." : "Categorical statistics are unavailable."} />
      </details>
      <details className="eda-details"><summary>Datetime features</summary>
        <ReportTable caption="Datetime descriptive statistics"
          rows={namedRows(record(report.datetime_statistics).datetime_statistics)}
          columns={[["column", "Feature"], ["earliest_datetime", "Earliest"], ["latest_datetime", "Latest"],
            ["time_span_days", "Span (days)"], ["invalid_datetime_values", "Invalid dates"], ["detection_method", "Detection"]]}
          empty={report.datetime_statistics ? "No datetime features were detected." : "Datetime statistics are unavailable."} />
      </details>
    </Section>

    <Section title="Feature–target relationships">
      <p>These associations describe the uploaded data, not causal effects or final training feature selection.
        Scores from different statistical methods are not directly equivalent.</p>
      <ReportTable caption="Ranked feature–target associations" rows={items(relationships.ranked_features).map(record)}
        columns={[["feature", "Feature"], ["analysis_method", "Method"], ["primary_score", "Association score"],
          ["relationship_strength", "Strength"], ["potential_leakage", "Leakage review"]]}
        empty={report.feature_target_relationships ? "No usable associations were returned." : "Feature relationships are unavailable."} />
      {items(relationships.skipped_features).length > 0 && <Notice>Skipped features: {display(relationships.skipped_features)}.</Notice>}
      <details className="eda-details"><summary>Feature–feature relationship overview</summary>
        <Facts data={record(report.feature_feature_relationships)} fields={[["analyzed_pair_count", "Pairs analyzed"],
          ["high_association_pair_count", "Highly associated pairs"], ["omitted_due_feature_limit", "Features omitted by analysis limit"]]} />
        <TextList values={record(report.feature_feature_relationships).high_association_pairs}
          empty="No high-association pairs are available in this report." />
      </details>
    </Section>

    <Section title="Cleaning, outliers & leakage review">
      <p>Outlier flags do not imply errors. Robust scaling reduces sensitivity to extremes; it does not remove outliers.
        Training must learn preprocessing and feature selection from training data only.</p>
      {!context.cleaning_plan ? <Notice warning>Cleaning recommendations are unavailable.</Notice> : <Facts data={cleaning}
        fields={[["drop_columns", "Proposed exclusions"], ["review_columns", "Needs human review"], ["kept_features", "Retained candidates"]]} />}
      <details className="eda-details"><summary>Outlier assessment</summary>
        {context.outlier_report ? <Facts data={record(context.outlier_report)}
          fields={Object.keys(record(context.outlier_report)).map(key => [key, label(key)])} />
          : <p className="empty-note">Outlier context is unavailable; descriptive outlier counts may still be shown above.</p>}
      </details>
      <details className="eda-details"><summary>Potential target leakage</summary>
        {context.leakage_report ? <Facts data={record(context.leakage_report)}
          fields={Object.keys(record(context.leakage_report)).map(key => [key, label(key)])} />
          : <p className="empty-note">Leakage context is unavailable. Absence of a flag does not prove leakage is absent.</p>}
      </details>
    </Section>

    <Section title="Visualizations">
      {!chartsRequested ? <Notice>Charts were turned off for this run. Enable charts and run EDA again to generate them.</Notice>
        : !report.visualizations ? <Notice warning>The visualization section is unavailable. Statistical results remain above.</Notice>
        : <>
          <Facts data={visualizations} fields={[["visualization_status", "Chart status"],
            ["generated_plot_count", "Generated"], ["skipped_plot_count", "Skipped"], ["plot_error_count", "Errors"]]} />
          <div className="eda-chart-grid">{items(visualizations.generated_plots).map((value, index) =>
            <Chart response={response} plot={record(value)} key={`${response.analysis_id}-${index}`} />)}</div>
          <p className="eda-caption">Open a chart in a new tab, then use your browser’s Save image option to download it.</p>
          {items(visualizations.skipped_plots).length > 0 && <details className="eda-details"><summary>Why some charts were skipped</summary>
            <TextList values={visualizations.skipped_plots} /></details>}
          {items(visualizations.plot_errors).length > 0 && <details className="eda-details" open><summary>Chart generation errors</summary>
            <TextList values={visualizations.plot_errors} /></details>}
          {Object.keys(record(visualizations.sampling)).length > 0 && <details className="eda-details"><summary>Chart sampling</summary>
            <Facts data={record(visualizations.sampling)} fields={Object.keys(record(visualizations.sampling)).map(key => [key, label(key)])} />
          </details>}
        </>}
    </Section>

    <Section title="Warnings, recommendations & section status">
      <details className="eda-details" open><summary>Report warnings</summary><TextList values={report.warnings} /></details>
      <details className="eda-details"><summary>Recommended next actions</summary><TextList values={report.recommendations} /></details>
      <details className="eda-details"><summary>Analysis limitations</summary><TextList values={report.limitations} /></details>
      <details className="eda-details"><summary>Section completion</summary>
        <ReportTable caption="EDA section status" rows={statuses.map(([section, status]) => ({ section: label(section), status }))}
          columns={[["section", "Section"], ["status", "Status"]]} empty="No section-status information was returned." />
        {Object.keys(record(report.section_errors)).length > 0 && <Facts data={record(report.section_errors)}
          fields={Object.keys(record(report.section_errors)).map(key => [key, label(key)])} />}
        {items(report.section_errors).length > 0 && <TextList values={report.section_errors} />}
      </details>
    </Section>
    <Notice>EDA is complete. Continue to <a className="eda-link" href="#train-models">Train models</a> when you have reviewed the findings.
      Training remains a separate explicit action.</Notice>
  </div>;
}

export function EdaPanel({ filename, targetColumn, onStatusChange }: {
  filename: string; targetColumn: string; onStatusChange?: (complete: boolean) => void;
}) {
  const latest = useRef(new LatestRequest());
  const [includeCharts, setIncludeCharts] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [result, setResult] = useState<{ response: EdaResponse; charts: boolean } | null>(null);
  useEffect(() => {
    const requests = latest.current;
    return () => requests.cancel();
  }, []);

  async function analyze() {
    const request = latest.current.start();
    const charts = includeCharts;
    setBusy(true); setError(null); setNotice(null); setResult(null); onStatusChange?.(false);
    try {
      const response = await runEda(filename, targetColumn, charts, request.signal);
      if (request.isCurrent()) { setResult({ response, charts }); onStatusChange?.(true); }
    } catch (caught) {
      if (request.isCurrent()) setError(caught instanceof Error ? caught.message : "EDA failed. Please retry.");
    } finally {
      if (request.isCurrent()) setBusy(false);
    }
  }

  return <section id="explore" className="panel explore-section" aria-labelledby="explore-title">
    <div className="section-heading"><span className="section-number">03</span><div>
      <h2 id="explore-title">Explore your dataset</h2>
      <p>Understand quality, relationships, and modeling risks before training.</p>
    </div></div>
    <div className="eda-controls">
      <div><span className="eyebrow">Ready to explore</span><strong>{targetColumn}</strong><span>{filename}</span></div>
      <label className="eda-checkbox"><input type="checkbox" checked={includeCharts} disabled={busy}
        onChange={event => setIncludeCharts(event.target.checked)} />Include charts</label>
      <button type="button" className="primary-button" onClick={analyze} disabled={busy}>
        {busy ? "Analyzing dataset…" : result ? "Run EDA again" : "Run EDA"}</button>
      {busy && <button className="secondary-button" type="button" onClick={() => {
        latest.current.cancel(); setBusy(false);
        setNotice("Stopped waiting for this report. The backend may continue calculating it.");
      }}>Cancel request</button>}
    </div>
    <div aria-live="polite" role="status">
      {busy && <Notice>Analyzing {targetColumn}. Charts can take a little longer. You can cancel waiting; no model is being trained.</Notice>}
      {notice && <Notice>{notice}</Notice>}
      {result && <p className="sr-only">EDA {display(result.response.report.report_status)} for {targetColumn}. Results are below.</p>}
    </div>
    {error && <div className="eda-error" role="alert"><strong>EDA could not finish.</strong><p>{error}</p>
      <p>Your dataset and target remain selected. Use Run EDA to retry.</p></div>}
    {!result && !busy && !error && <Notice>Run an exploratory analysis of your selected dataset and target.
      This creates a report, not a cleaned dataset or a trained model.</Notice>}
    {result && <EdaReportView key={result.response.analysis_id || "statistics-only"}
      response={result.response} chartsRequested={result.charts} />}
  </section>;
}
