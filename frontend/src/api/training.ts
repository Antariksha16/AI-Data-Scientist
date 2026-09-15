import { API_BASE_URL, ApiError, requestJson } from "./client";
import { type DataRecord, items, record } from "./eda";

export interface TrainingOptions {
  maxRecommendedModels: number;
  runTuning: boolean;
  tuningIterations: number;
  tuningTopModels: number;
  includeExplanations: boolean;
  includeProfessionalReport: boolean;
  allowPotentialLeakage: boolean;
}

export interface TrainingResponse {
  filename: string;
  selected_target: string;
  training_id: string | null;
  report: DataRecord;
  artifacts: DataRecord | null;
  message: string;
}

export const DEFAULT_TRAINING_OPTIONS: TrainingOptions = {
  maxRecommendedModels: 4,
  runTuning: true,
  tuningIterations: 12,
  tuningTopModels: 2,
  includeExplanations: true,
  includeProfessionalReport: true,
  allowPotentialLeakage: false,
};

const forbiddenKeys = new Set(["X_holdout", "y_holdout", "evaluation_context"]);
export function containsPrivateEvaluationData(value: unknown): boolean {
  if (Array.isArray(value)) return value.some(containsPrivateEvaluationData);
  if (value === null || typeof value !== "object") return false;
  return Object.entries(value).some(([key, item]) =>
    forbiddenKeys.has(key) || containsPrivateEvaluationData(item));
}

export async function trainModels(
  filename: string,
  targetColumn: string,
  options: TrainingOptions,
  signal: AbortSignal,
): Promise<TrainingResponse> {
  const result = await requestJson<TrainingResponse>("/analysis/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      filename,
      target_column: targetColumn,
      max_recommended_models: options.maxRecommendedModels,
      run_tuning: options.runTuning,
      tuning_iterations: options.tuningIterations,
      tuning_top_models: Math.min(options.tuningTopModels, options.maxRecommendedModels),
      include_explanations: options.includeExplanations,
      explainability_repeats: 5,
      explainability_max_rows: 2000,
      explainability_top_n: 20,
      include_explanation_visualizations: options.includeExplanations,
      explanation_image_format: "png",
      explanation_visualization_dpi: 120,
      include_professional_report: options.includeProfessionalReport,
      professional_report_formats: ["markdown", "pdf"],
      include_eda_visualizations_in_report: options.includeProfessionalReport,
      export_artifacts: true,
      allow_potential_leakage: options.allowPotentialLeakage,
      training_configuration_overrides: {},
    }),
  }, "Unable to train and compare models.");

  const report = record(result.report);
  if (result.filename !== filename || result.selected_target !== targetColumn
      || report.target_column !== targetColumn
      || typeof report.training_status !== "string"
      || !result.training_id || !/^[a-f0-9]{32}$/.test(result.training_id)
      || !result.artifacts) {
    throw new ApiError("The training response does not match this dataset, target, or exported run. Please retry.", 502);
  }
  if (containsPrivateEvaluationData(result)) {
    throw new ApiError("The server returned private evaluation context. The result was not displayed.", 502);
  }
  return result;
}

export interface DownloadArtifact {
  artifact_type: string;
  file_name: string;
  artifact_url: string;
  sha256?: string;
  size_bytes?: number;
}

export function trainingDownloads(response: TrainingResponse): DownloadArtifact[] {
  const id = response.training_id;
  if (!id || !/^[a-f0-9]{32}$/.test(id)) return [];
  const prefix = `/analysis/train/artifacts/${id}/`;
  return items(record(response.artifacts).downloads).flatMap(value => {
    const item = record(value);
    const type = item.artifact_type;
    const name = item.file_name;
    const url = item.artifact_url;
    if (typeof type !== "string" || typeof name !== "string" || typeof url !== "string"
        || !/^[a-z0-9_]+$/.test(type) || !/^[a-z0-9_]+\.(joblib|json|md|pdf|png)$/.test(name)
        || url !== `${prefix}${name}`) return [];
    return [{ artifact_type: type, file_name: name, artifact_url: `${API_BASE_URL}${url}`,
      ...(typeof item.sha256 === "string" ? { sha256: item.sha256 } : {}),
      ...(typeof item.size_bytes === "number" ? { size_bytes: item.size_bytes } : {}) }];
  });
}
