import { useEffect, useRef, useState } from "react";

import { display } from "./EdaPanel";
import { LatestRequest, items, record } from "../api/eda";
import {
  buildPredictionRecords,
  emptyPredictionValues,
  featureKind,
  loadPredictionSchema,
  predictRecords,
  type PredictionInputSchema,
  type PredictionInputValues,
  type PredictionResponse,
  type PredictionSchemaResponse,
} from "../api/prediction";
import { type TrainingResponse } from "../api/training";

const MAX_BROWSER_ROWS = 50;

function humanize(value: unknown): string {
  return String(value ?? "Not available").replaceAll("_", " ");
}

function percentage(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(2)}%`
    : "Not available";
}

function PredictionField({
  feature,
  rowIndex,
  value,
  schema,
  optional,
  disabled,
  onChange,
}: {
  feature: string;
  rowIndex: number;
  value: string;
  schema: PredictionInputSchema;
  optional: boolean;
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  const kind = featureKind(schema, feature);
  const dtype = schema.pandas_dtypes[feature];
  const id = `prediction-${rowIndex}-${feature.replace(/[^a-z0-9_-]/gi, "-")}`;
  const helpId = `${id}-help`;
  return <label className="prediction-field" htmlFor={id}>
    <span><strong>{feature}</strong><small>{optional ? "Optional" : "Model input"}</small></span>
    {kind === "boolean" ? <select id={id} value={value} disabled={disabled}
      aria-describedby={helpId} onChange={event => onChange(event.target.value)}>
      <option value="">Missing value</option><option value="true">True</option><option value="false">False</option>
    </select> : <input id={id} value={value} disabled={disabled}
      type={kind === "number" ? "number" : "text"} step={kind === "number" ? "any" : undefined}
      inputMode={kind === "number" ? "decimal" : undefined}
      placeholder={kind === "number" ? "Enter a number" : "Enter a value"}
      aria-describedby={helpId} onChange={event => onChange(event.target.value)} />}
    <small id={helpId}>{dtype ? `${dtype} · ` : ""}{optional
      ? "Blank fields are omitted."
      : "Blank is treated as missing and handled by the fitted preprocessor."}</small>
  </label>;
}

export function PredictionResult({ response }: { response: PredictionResponse }) {
  const report = response.report;
  const rows = items(report.predictions).map(record);
  const classification = report.problem_type === "classification";
  const verification = record(report.artifact_verification);
  return <section id="prediction-results" className="prediction-results" aria-labelledby="prediction-results-title">
    <div className="prediction-result-heading">
      <div><p className="eyebrow">Prediction complete</p><h3 id="prediction-results-title">
        {display(report.target_column)} predictions</h3>
        <p>{display(report.model_display_name)} · {humanize(report.selected_variant)}</p></div>
      <strong>{display(report.prediction_count)} {Number(report.prediction_count) === 1 ? "record" : "records"}</strong>
    </div>
    <div className="prediction-output-grid">
      {rows.map((row, index) => {
        const probabilities = items(row.probabilities).map(record);
        return <article className="prediction-output" key={index}>
          <div className="prediction-output-title"><span>Record {index + 1}</span>
            {classification && typeof row.confidence === "number" && <small>{percentage(row.confidence)} confidence</small>}</div>
          <div className="prediction-value"><span>Predicted {display(report.target_column)}</span>
            <strong>{display(row.prediction)}</strong></div>
          {classification && probabilities.length > 0 && <div className="probability-list" aria-label={`Class probabilities for record ${index + 1}`}>
            {probabilities.map((entry, probabilityIndex) => {
              const probability = typeof entry.probability === "number" ? entry.probability : 0;
              return <div className="probability-row" key={`${display(entry.class_label)}-${probabilityIndex}`}>
                <div><span>Class {display(entry.class_label)}</span><strong>{percentage(probability)}</strong></div>
                <div className="probability-track" aria-hidden="true"><span style={{ width: `${Math.max(0, Math.min(100, probability * 100))}%` }} /></div>
              </div>;
            })}
          </div>}
          {!classification && <p className="regression-note">This is a continuous numerical estimate, not a probability.</p>}
        </article>;
      })}
    </div>
    <div className="prediction-verification">
      <span aria-hidden="true">✓</span><div><strong>Verified model pipeline</strong>
        <p>{humanize(verification.verification_status)} · model loaded successfully · source records unchanged</p></div>
    </div>
  </section>;
}

export function PredictionWorkspace({
  schema,
  values,
  includeProbabilities,
  busy,
  onValuesChange,
  onIncludeProbabilitiesChange,
  onPredict,
  onCancel,
}: {
  schema: PredictionSchemaResponse;
  values: PredictionInputValues[];
  includeProbabilities: boolean;
  busy: boolean;
  onValuesChange: (values: PredictionInputValues[]) => void;
  onIncludeProbabilitiesChange: (value: boolean) => void;
  onPredict: () => void;
  onCancel: () => void;
}) {
  const input = schema.input_schema;
  const required = input.required_features;
  const optional = input.optional_features;
  function update(rowIndex: number, feature: string, value: string) {
    onValuesChange(values.map((row, index) => index === rowIndex ? { ...row, [feature]: value } : row));
  }
  return <>
    <div className="prediction-model-card"><div><p className="eyebrow">Verified fitted model</p>
      <h3>{schema.model_display_name || schema.model_name || "Trained model"}</h3>
      <p>Predicting <strong>{schema.target_column}</strong> · {humanize(schema.problem_type)} · {humanize(schema.selected_variant)}</p></div>
      <span className="verified-pill">✓ Pipeline verified</span></div>

    <div className="prediction-records">
      {values.map((row, rowIndex) => <fieldset className="prediction-record" key={rowIndex}>
        <legend>Prediction record {rowIndex + 1}</legend>
        {values.length > 1 && <button type="button" className="remove-record" disabled={busy}
          aria-label={`Remove prediction record ${rowIndex + 1}`}
          onClick={() => onValuesChange(values.filter((_, index) => index !== rowIndex))}>Remove</button>}
        <div className="prediction-field-grid">{required.map(feature => <PredictionField key={feature}
          feature={feature} rowIndex={rowIndex} value={row[feature] || ""} schema={input}
          optional={false} disabled={busy} onChange={value => update(rowIndex, feature, value)} />)}</div>
        {optional.length > 0 && <details className="optional-fields"><summary>Optional source fields ({optional.length})</summary>
          <p>These fields are accepted by the raw pipeline but can be omitted.</p>
          <div className="prediction-field-grid">{optional.map(feature => <PredictionField key={feature}
            feature={feature} rowIndex={rowIndex} value={row[feature] || ""} schema={input}
            optional disabled={busy} onChange={value => update(rowIndex, feature, value)} />)}</div>
        </details>}
      </fieldset>)}
    </div>

    <div className="prediction-row-actions"><button type="button" className="secondary-button"
      disabled={busy || values.length >= MAX_BROWSER_ROWS}
      onClick={() => onValuesChange([...values, emptyPredictionValues(input)])}>Add another record</button>
      <button type="button" className="text-button" disabled={busy}
        onClick={() => onValuesChange([emptyPredictionValues(input)])}>Clear records</button>
      <span>{values.length} of {MAX_BROWSER_ROWS} browser records</span></div>

    {schema.problem_type === "classification" && <label className="probability-toggle">
      <input type="checkbox" checked={includeProbabilities} disabled={busy}
        onChange={event => onIncludeProbabilitiesChange(event.target.checked)} />
      <span><strong>Show class probabilities and confidence</strong>
        <small>Available only when the selected classifier supports verified probability estimates.</small></span>
    </label>}

    <div className="prediction-actions"><button type="button" className="primary-button" disabled={busy} onClick={onPredict}>
      {busy ? "Generating predictions…" : "Generate predictions"}</button>
      {busy && <button type="button" className="secondary-button" onClick={onCancel}>Cancel request</button>}</div>
  </>;
}

export function PredictionPanel({ training }: { training: TrainingResponse }) {
  const schemaRequests = useRef(new LatestRequest());
  const predictionRequests = useRef(new LatestRequest());
  const [schema, setSchema] = useState<PredictionSchemaResponse | null>(null);
  const [values, setValues] = useState<PredictionInputValues[]>([]);
  const [includeProbabilities, setIncludeProbabilities] = useState(true);
  const [loadingSchema, setLoadingSchema] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  async function refreshSchema() {
    const request = schemaRequests.current.start();
    predictionRequests.current.cancel();
    setLoadingSchema(true); setPredicting(false); setError(null); setResult(null); setSchema(null);
    try {
      const response = await loadPredictionSchema(training, request.signal);
      if (request.isCurrent()) {
        setSchema(response);
        setValues([emptyPredictionValues(response.input_schema)]);
      }
    } catch (caught) {
      if (request.isCurrent()) setError(caught instanceof Error ? caught.message : "Unable to load model inputs.");
    } finally {
      if (request.isCurrent()) setLoadingSchema(false);
    }
  }

  useEffect(() => {
    void refreshSchema();
    const schemaLatest = schemaRequests.current;
    const predictionLatest = predictionRequests.current;
    return () => { schemaLatest.cancel(); predictionLatest.cancel(); };
  }, [training.training_id]);

  function changeValues(next: PredictionInputValues[]) {
    setValues(next); setResult(null); setError(null);
  }

  async function runPrediction() {
    if (!schema) return;
    try {
      buildPredictionRecords(values, schema.input_schema);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Review the prediction values.");
      return;
    }
    const request = predictionRequests.current.start();
    setPredicting(true); setError(null); setResult(null);
    try {
      const response = await predictRecords(schema, values, includeProbabilities, request.signal);
      if (request.isCurrent()) {
        setResult(response);
        requestAnimationFrame(() => document.getElementById("prediction-results")?.scrollIntoView({ block: "start" }));
      }
    } catch (caught) {
      if (request.isCurrent()) setError(caught instanceof Error ? caught.message : "Prediction failed. Please retry.");
    } finally {
      if (request.isCurrent()) setPredicting(false);
    }
  }

  return <section id="predict" className="panel prediction-panel" aria-labelledby="prediction-title">
    <div className="section-heading"><span className="section-number">06</span><div>
      <h2 id="prediction-title">Predict with the selected model</h2>
      <p>Enter new raw feature values. The saved preprocessing and model pipeline will run without retraining.</p>
    </div></div>
    {loadingSchema && <div className="prediction-message" role="status" aria-live="polite">
      <strong>Loading verified model inputs…</strong><p>The model bundle is checked before its schema is displayed.</p></div>}
    {!loadingSchema && error && !schema && <div className="eda-error" role="alert"><strong>Model inputs could not be loaded.</strong>
      <p>{error}</p><button type="button" className="secondary-button" onClick={() => void refreshSchema()}>Retry schema</button></div>}
    {schema && <PredictionWorkspace schema={schema} values={values}
      includeProbabilities={includeProbabilities} busy={predicting}
      onValuesChange={changeValues} onIncludeProbabilitiesChange={value => {
        setIncludeProbabilities(value); setResult(null);
      }} onPredict={() => void runPrediction()} onCancel={() => {
        predictionRequests.current.cancel(); setPredicting(false);
        setError("Stopped waiting for this prediction request. No model or source data was changed.");
      }} />}
    {predicting && <div className="prediction-message" role="status" aria-live="polite"><strong>Generating predictions…</strong>
      <p>The verified fitted pipeline is processing {values.length} {values.length === 1 ? "record" : "records"}.</p></div>}
    {!loadingSchema && error && schema && <div className="eda-error" role="alert"><strong>Prediction could not finish.</strong><p>{error}</p></div>}
    {result && <PredictionResult response={result} />}
  </section>;
}
