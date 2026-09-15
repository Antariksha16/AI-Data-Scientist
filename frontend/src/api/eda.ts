import { API_BASE_URL, ApiError, requestJson } from "./client";

export type DataRecord = Record<string, unknown>;
export interface EdaResponse {
  filename: string;
  selected_target: string;
  analysis_id: string | null;
  report: DataRecord;
}

export const record = (value: unknown): DataRecord =>
  value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as DataRecord : {};
export const items = (value: unknown): unknown[] => Array.isArray(value) ? value : [];
export const words = (value: unknown): string[] =>
  items(value).filter((item): item is string => typeof item === "string");

export async function runEda(
  filename: string,
  targetColumn: string,
  includeVisualizations: boolean,
  signal: AbortSignal,
): Promise<EdaResponse> {
  const result = await requestJson<EdaResponse>("/analysis/eda", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      filename,
      target_column: targetColumn,
      include_feature_relationships: true,
      include_preprocessing_context: true,
      include_insights: true,
      include_visualizations: includeVisualizations,
      visualization_image_format: "png",
      visualization_dpi: 120,
    }),
  }, "Unable to generate the EDA report.");
  if (result.filename !== filename || result.selected_target !== targetColumn
      || record(result.report).target_column !== targetColumn
      || typeof record(result.report).report_status !== "string") {
    throw new ApiError("The EDA response does not match this dataset and target. Please retry.", 502);
  }
  return result;
}

/** Only backend-issued PNG artifacts for this exact analysis may be opened. */
export function edaArtifactUrl(response: EdaResponse, plot: DataRecord): string | null {
  const id = response.analysis_id;
  const name = plot.file_name;
  if (!id || !/^[a-f0-9]{32}$/.test(id) || typeof name !== "string"
      || !/^[a-z0-9_]+\.png$/.test(name)) return null;
  const path = `/analysis/eda/artifacts/${id}/${name}`;
  return plot.artifact_url === path ? `${API_BASE_URL}${path}` : null;
}

/** Invalidates late completions even when a server ignores browser cancellation. */
export class LatestRequest {
  private generation = 0;
  private controller: AbortController | null = null;
  cancel() {
    this.generation += 1;
    this.controller?.abort();
    this.controller = null;
  }
  start() {
    this.cancel();
    const generation = this.generation;
    const controller = new AbortController();
    this.controller = controller;
    return {
      signal: controller.signal,
      isCurrent: () => generation === this.generation && !controller.signal.aborted,
    };
  }
}
