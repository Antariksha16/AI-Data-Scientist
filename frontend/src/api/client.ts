const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();

export const API_BASE_URL = (
  configuredBaseUrl || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

export type ProblemType = "classification" | "regression" | "undetermined";

export interface DatasetAnalysis {
  rows: number;
  columns: number;
  column_names: string[];
  data_types: Record<string, string>;
  numerical_columns: string[];
  categorical_columns: string[];
  missing_values: Record<string, number>;
  missing_percentage: Record<string, number>;
  duplicate_rows: number;
  unique_counts: Record<string, number>;
  constant_columns: string[];
}

export interface TargetSuggestion {
  column?: string;
  column_name?: string;
  target_column?: string;
  score?: number;
  problem_type?: ProblemType;
  reasons?: string[];
  [key: string]: unknown;
}

export interface UploadResponse {
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
  preview: Array<Record<string, unknown>>;
  dataset_analysis: DatasetAnalysis;
  target_suggestions: TargetSuggestion[];
}

export interface TargetAnalysis {
  target_column: string;
  data_type: string;
  unique_values: number;
  missing_values: number;
  missing_percentage: number;
  problem_type: ProblemType;
}

export interface TargetValidationResponse {
  filename: string;
  selected_target: string;
  target_analysis: TargetAnalysis;
  candidate_feature_count: number;
  candidate_features: string[];
  message: string;
}

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function errorMessage(payload: unknown, fallback: string): string {
  if (typeof payload === "object" && payload !== null && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (typeof item === "object" && item !== null && "msg" in item) {
            return String((item as { msg: unknown }).msg);
          }
          return String(item);
        })
        .join(" ");
    }
  }
  return fallback;
}

async function parseJson(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) return null;
  return response.json();
}

export async function requestJson<T>(
  path: string,
  init: RequestInit,
  fallbackError: string,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch (error) {
    if (init.signal?.aborted) throw error;
    throw new ApiError(
      "The backend is unavailable. Confirm that FastAPI is running on port 8000.",
      0,
    );
  }

  let payload: unknown;
  try {
    payload = await parseJson(response);
  } catch (error) {
    if (init.signal?.aborted) throw error;
    throw new ApiError("The backend returned unreadable JSON. Please retry.", response.status);
  }
  if (!response.ok) {
    throw new ApiError(errorMessage(payload, fallbackError), response.status);
  }
  if (payload === null || typeof payload !== "object") {
    throw new ApiError("The backend did not return the expected JSON response.", response.status);
  }
  return payload as T;
}

export function uploadDataset(file: File, signal?: AbortSignal): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return requestJson<UploadResponse>(
    "/upload",
    { method: "POST", body: formData, signal },
    "Unable to upload and inspect the dataset.",
  );
}

export function validateTarget(
  filename: string,
  targetColumn: string,
  signal?: AbortSignal,
): Promise<TargetValidationResponse> {
  return requestJson<TargetValidationResponse>(
    "/analysis/target",
    {
      method: "POST",
      signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename,
        target_column: targetColumn,
      }),
    },
    "Unable to validate the selected target.",
  );
}
