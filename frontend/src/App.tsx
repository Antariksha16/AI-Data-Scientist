import {
  type ChangeEvent,
  type DragEvent,
  useMemo,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  API_BASE_URL,
  type TargetSuggestion,
  type TargetValidationResponse,
  type UploadResponse,
  uploadDataset,
  validateTarget,
} from "./api/client";
import {
  AlertIcon,
  ArrowIcon,
  CheckIcon,
  DatabaseIcon,
  FileIcon,
  TargetIcon,
  UploadIcon,
} from "./components/Icons";
import { EdaPanel } from "./components/EdaPanel";
import { LatestRequest } from "./api/eda";
import { TrainingPanel } from "./components/TrainingPanel";
import { PredictionPanel } from "./components/PredictionPanel";
import { type TrainingResponse } from "./api/training";

const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

const workflow = [
  { label: "Upload data", detail: "CSV or Excel" },
  { label: "Choose target", detail: "Validate prediction goal" },
  { label: "Explore", detail: "Quality and relationships" },
  { label: "Train models", detail: "Compare and tune" },
  { label: "Results", detail: "Explain and export" },
  { label: "Predict", detail: "Use verified model" },
];

function suggestionName(suggestion: TargetSuggestion): string | null {
  for (const key of ["column", "column_name", "target_column"] as const) {
    const value = suggestion[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return null;
}

function suggestionScore(suggestion: TargetSuggestion): string | null {
  return typeof suggestion.score === "number"
    ? suggestion.score.toFixed(1).replace(/\.0$/, "")
    : null;
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "Missing";
  if (typeof value === "object") {
    const serialized = JSON.stringify(value);
    return serialized.length > 70 ? `${serialized.slice(0, 69)}…` : serialized;
  }
  const text = String(value);
  return text.length > 70 ? `${text.slice(0, 69)}…` : text;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const setupRequests = useRef(new LatestRequest());
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedTarget, setSelectedTarget] = useState("");
  const [targetResult, setTargetResult] =
    useState<TargetValidationResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [validating, setValidating] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [edaComplete, setEdaComplete] = useState(false);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [trainingResult, setTrainingResult] = useState<TrainingResponse | null>(null);

  useEffect(() => {
    const requests = setupRequests.current;
    return () => requests.cancel();
  }, []);

  function invalidateSelection() {
    setupRequests.current.cancel();
    setTargetResult(null);
    setEdaComplete(false);
    setTrainingComplete(false);
    setTrainingResult(null);
    setUploading(false);
    setValidating(false);
    setError(null);
  }

  function chooseTarget(value: string) {
    invalidateSelection();
    setSelectedTarget(value);
  }

  const activeStep = trainingResult ? 5 : trainingComplete ? 4 : edaComplete ? 3 : targetResult ? 2 : upload ? 1 : 0;
  const missingCellCount = useMemo(() => {
    if (!upload) return 0;
    return Object.values(upload.dataset_analysis.missing_values).reduce(
      (total, count) => total + count,
      0,
    );
  }, [upload]);

  const suggestions = useMemo(
    () =>
      (upload?.target_suggestions || [])
        .map((suggestion) => ({
          suggestion,
          name: suggestionName(suggestion),
        }))
        .filter(
          (item): item is { suggestion: TargetSuggestion; name: string } =>
            item.name !== null,
        )
        .slice(0, 5),
    [upload],
  );

  function resetForFile(file: File) {
    invalidateSelection();
    setSelectedFile(file);
    setUpload(null);
    setSelectedTarget("");
    setTargetResult(null);
    setError(null);
  }

  function acceptFile(file: File | undefined) {
    if (!file) return;
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!extension || !["csv", "xlsx", "xls"].includes(extension)) {
      setError("Choose a CSV, XLSX, or XLS dataset.");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setError("The dataset exceeds the 100 MB upload limit.");
      return;
    }
    resetForFile(file);
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  async function inspectDataset() {
    if (!selectedFile) {
      inputRef.current?.click();
      return;
    }
    const request = setupRequests.current.start();
    setUploading(true);
    setError(null);
    try {
      const result = await uploadDataset(selectedFile, request.signal);
      if (!request.isCurrent()) return;
      setUpload(result);
      const suggested = result.target_suggestions
        .map(suggestionName)
        .find((name): name is string => name !== null);
      setSelectedTarget(suggested || "");
    } catch (caught) {
      if (request.isCurrent()) setError(caught instanceof Error ? caught.message : "Upload failed.");
    } finally {
      if (request.isCurrent()) setUploading(false);
    }
  }

  async function confirmTarget() {
    if (!upload || !selectedTarget) return;
    const request = setupRequests.current.start();
    setValidating(true);
    setError(null);
    setTargetResult(null);
    try {
      const result = await validateTarget(upload.filename, selectedTarget, request.signal);
      if (request.isCurrent()) setTargetResult(result);
    } catch (caught) {
      if (request.isCurrent()) setError(
        caught instanceof Error ? caught.message : "Target validation failed.",
      );
    } finally {
      if (request.isCurrent()) setValidating(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="workflow-panel" aria-label="Analysis workflow">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            <DatabaseIcon />
          </div>
          <div>
            <strong>AI Data Scientist</strong>
            <span>Transparent AutoML</span>
          </div>
        </div>

        <nav className="workflow-steps">
          {workflow.map((step, index) => {
            const complete = index < activeStep;
            const active = index === activeStep;
            return (
              <button
                type="button"
                disabled={index > activeStep}
                aria-current={active ? "step" : undefined}
                onClick={() => {
                  const destination = index === 0 ? (upload ? "dataset-preview" : "dataset-upload")
                    : index === 1 ? "choose-target" : index === 2 ? "explore"
                      : index === 3 ? "train-models" : index === 4 ? "training-results" : "predict";
                  document.getElementById(destination)?.scrollIntoView({ block: "start" });
                }}
                className={`workflow-step ${active ? "active" : ""} ${
                  complete ? "complete" : ""
                }`}
                key={step.label}
              >
                <span className="step-index">
                  {complete ? <CheckIcon /> : index + 1}
                </span>
                <span className="workflow-copy">
                  <strong>{step.label}</strong>
                  <span>{step.detail}</span>
                </span>
              </button>
            );
          })}
        </nav>

        <div className="privacy-note">
          <span className="status-dot" />
          <div>
            <strong>Local workspace</strong>
            <span>Files are sent only to {API_BASE_URL}</span>
          </div>
        </div>
      </aside>

      <main className="main-workspace">
        <header className="workspace-header">
          <div>
            <p className="eyebrow">New analysis</p>
            <h1>Prepare your dataset</h1>
            <p>
              Upload tabular data, inspect its structure, and confirm what the
              model should predict.
            </p>
          </div>
          <div className="api-status" title={`Connected API: ${API_BASE_URL}`}>
            <span className="status-dot" /> Backend configured
          </div>
        </header>

        {error && (
          <div className="error-banner" role="alert">
            <AlertIcon />
            <div>
              <strong>Unable to continue</strong>
              <span>{error}</span>
            </div>
          </div>
        )}

        {!upload && (
          <section id="dataset-upload" className="panel upload-section" aria-labelledby="upload-title">
            <div className="section-heading">
              <span className="section-number">01</span>
              <div>
                <h2 id="upload-title">Upload a dataset</h2>
                <p>CSV and Excel files up to 100 MB are supported.</p>
              </div>
            </div>

            <div
              className={`drop-zone ${dragging ? "dragging" : ""} ${
                selectedFile ? "has-file" : ""
              }`}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={onFileChange}
                className="sr-only"
                id="dataset-file"
              />
              <div className="upload-illustration">
                {selectedFile ? <FileIcon /> : <UploadIcon />}
              </div>
              {selectedFile ? (
                <>
                  <strong>{selectedFile.name}</strong>
                  <span>{formatBytes(selectedFile.size)} · ready to inspect</span>
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => inputRef.current?.click()}
                  >
                    Choose a different file
                  </button>
                </>
              ) : (
                <>
                  <strong>Drop your dataset here</strong>
                  <span>or select a file from your computer</span>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => inputRef.current?.click()}
                  >
                    Browse files
                  </button>
                </>
              )}
            </div>

            <div className="panel-actions">
              <span>Your source file is never modified by the analysis.</span>
              <button
                type="button"
                className="primary-button"
                onClick={inspectDataset}
                disabled={uploading}
              >
                {uploading ? "Inspecting dataset…" : "Inspect dataset"}
                {!uploading && <ArrowIcon />}
              </button>
            </div>
          </section>
        )}

        {upload && (
          <>
            <section className="dataset-summary" aria-label="Dataset summary">
              <article>
                <span>Rows</span>
                <strong>{upload.rows.toLocaleString()}</strong>
                <small>records detected</small>
              </article>
              <article>
                <span>Columns</span>
                <strong>{upload.columns.toLocaleString()}</strong>
                <small>available fields</small>
              </article>
              <article>
                <span>Missing cells</span>
                <strong>{missingCellCount.toLocaleString()}</strong>
                <small>handled during preprocessing</small>
              </article>
              <article>
                <span>Duplicate rows</span>
                <strong>{upload.dataset_analysis.duplicate_rows}</strong>
                <small>flagged for review</small>
              </article>
            </section>

            <section id="dataset-preview" className="panel preview-section" aria-labelledby="preview-title">
              <div className="section-heading compact">
                <span className="section-number complete-number">
                  <CheckIcon />
                </span>
                <div>
                  <h2 id="preview-title">{upload.filename}</h2>
                  <p>First {upload.preview.length} rows · source preview only</p>
                </div>
                <button
                  type="button"
                  className="text-button change-file"
                  onClick={() => {
                    invalidateSelection();
                    setUpload(null);
                    setTargetResult(null);
                    setSelectedTarget("");
                    setEdaComplete(false);
                    setTrainingComplete(false);
                    setTrainingResult(null);
                  }}
                >
                  Change file
                </button>
              </div>
              <div className="table-scroll" tabIndex={0}>
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      {upload.column_names.map((column) => (
                        <th key={column}>{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {upload.preview.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        <td className="row-number">{rowIndex + 1}</td>
                        {upload.column_names.map((column) => (
                          <td
                            key={`${rowIndex}-${column}`}
                            className={
                              row[column] === null || row[column] === undefined
                                ? "missing-cell"
                                : ""
                            }
                          >
                            {formatCell(row[column])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section id="choose-target" className="panel target-section" aria-labelledby="target-title">
              <div className="section-heading">
                <span className="section-number">02</span>
                <div>
                  <h2 id="target-title">Choose the prediction target</h2>
                  <p>
                    Suggestions are ranked from the uploaded dataset. You remain
                    in control of the final target.
                  </p>
                </div>
              </div>

              <div className="target-layout">
                <div className="suggestion-column">
                  <h3>Suggested targets</h3>
                  {suggestions.length ? (
                    <div className="suggestion-list">
                      {suggestions.map(({ suggestion, name }, index) => {
                        const selected = name === selectedTarget;
                        return (
                          <button
                            type="button"
                            className={`suggestion-card ${selected ? "selected" : ""}`}
                            onClick={() => chooseTarget(name)}
                            key={name}
                          >
                            <span className="suggestion-rank">{index + 1}</span>
                            <span className="suggestion-copy">
                              <strong>{name}</strong>
                              <small>
                                {suggestion.problem_type
                                  ? `${suggestion.problem_type} candidate`
                                  : "recommended candidate"}
                              </small>
                            </span>
                            {suggestionScore(suggestion) && (
                              <span className="score-pill">
                                {suggestionScore(suggestion)} score
                              </span>
                            )}
                            <span className="selection-indicator">
                              {selected && <CheckIcon />}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  ) : (
                    <p className="empty-note">
                      No ranked suggestions were returned. Select any usable
                      column manually.
                    </p>
                  )}
                </div>

                <div className="target-control">
                  <div className="target-icon">
                    <TargetIcon />
                  </div>
                  <h3>Confirm your target</h3>
                  <p>The target is the value the final model will learn to predict.</p>
                  <label htmlFor="target-column">Target column</label>
                  <select
                    id="target-column"
                    value={selectedTarget}
                    onChange={(event) => chooseTarget(event.target.value)}
                  >
                    <option value="">Select a column</option>
                    {upload.column_names.map((column) => (
                      <option value={column} key={column}>
                        {column}
                      </option>
                    ))}
                  </select>

                  <button
                    type="button"
                    className="primary-button full-width"
                    onClick={confirmTarget}
                    disabled={!selectedTarget || validating}
                  >
                    {validating ? "Validating target…" : "Validate target"}
                    {!validating && <ArrowIcon />}
                  </button>

                  {targetResult && (
                    <div className="target-success" aria-live="polite">
                      <div className="success-heading">
                        <span><CheckIcon /></span>
                        <strong>Target ready for analysis</strong>
                      </div>
                      <dl>
                        <div>
                          <dt>Problem type</dt>
                          <dd>{targetResult.target_analysis.problem_type}</dd>
                        </div>
                        <div>
                          <dt>Unique values</dt>
                          <dd>{targetResult.target_analysis.unique_values}</dd>
                        </div>
                        <div>
                          <dt>Candidate features</dt>
                          <dd>{targetResult.candidate_feature_count}</dd>
                        </div>
                        <div>
                          <dt>Missing target</dt>
                          <dd>{targetResult.target_analysis.missing_percentage}%</dd>
                        </div>
                      </dl>
                      <p>
                        Dataset setup is complete. This selection will be used by
                        EDA, preprocessing, and model training.
                      </p>
                      <a href="#explore" className="eda-link">Explore dataset →</a>
                    </div>
                  )}
                </div>
              </div>
            </section>
            {targetResult && <EdaPanel
              key={`${upload.filename}:${targetResult.selected_target}`}
              filename={upload.filename}
              targetColumn={targetResult.selected_target}
              onStatusChange={(complete) => {
                setEdaComplete(complete);
                if (!complete) {
                  setTrainingComplete(false);
                  setTrainingResult(null);
                }
              }}
            />}
            {targetResult && edaComplete && <TrainingPanel
              key={`${upload.filename}:${targetResult.selected_target}`}
              filename={upload.filename}
              targetColumn={targetResult.selected_target}
              onStatusChange={(complete) => {
                setTrainingComplete(complete);
                if (!complete) setTrainingResult(null);
              }}
              onResultChange={setTrainingResult}
            />}
            {trainingResult && <PredictionPanel
              key={trainingResult.training_id || "prediction"}
              training={trainingResult}
            />}
          </>
        )}
      </main>
    </div>
  );
}

export default App;
