import { ApiError, requestJson } from "./client";
import { type DataRecord, items, record } from "./eda";
import { containsPrivateEvaluationData, type TrainingResponse } from "./training";

export type PredictionProblemType = "classification" | "regression";
export type PredictionInputValues = Record<string, string>;

export interface PredictionInputSchema {
  schema_version: string;
  schema_source: string;
  raw_feature_count: number;
  raw_feature_names: string[];
  required_features: string[];
  optional_features: string[];
  numerical_features: string[];
  categorical_features: string[];
  pandas_dtypes: Record<string, string>;
  sample_values_stored: false;
}

export interface PredictionSchemaResponse {
  training_id: string;
  schema_status: "complete";
  target_column: string;
  problem_type: PredictionProblemType;
  model_name: string | null;
  model_display_name: string | null;
  selected_variant: string | null;
  input_schema: PredictionInputSchema;
  artifact_verification: {
    verification_status: "complete";
    pipeline_load_verified: true;
  };
  message: string;
}

export interface PredictionResponse {
  training_id: string;
  report: DataRecord;
  message: string;
}

const TRAINING_ID_PATTERN = /^[a-f0-9]{32}$/;
const numericDtypePattern = /(?:^|\W)(?:u?int\d*|float\d*|decimal|number)(?:$|\W)/i;
const booleanDtypePattern = /(?:^|\W)(?:bool|boolean)(?:$|\W)/i;

function strings(value: unknown, field: string): string[] {
  if (!Array.isArray(value) || value.some(item => typeof item !== "string" || !item.trim())) {
    throw new ApiError(`The model input schema contains an invalid ${field} list.`, 502);
  }
  const values = value as string[];
  if (new Set(values).size !== values.length) {
    throw new ApiError(`The model input schema contains duplicate ${field}.`, 502);
  }
  return [...values];
}

function sameSet(left: Set<string>, right: Set<string>): boolean {
  return left.size === right.size && [...left].every(value => right.has(value));
}

export function validatePredictionSchema(
  value: unknown,
  expectedTrainingId: string,
): PredictionSchemaResponse {
  const response = record(value);
  const schema = record(response.input_schema);
  const verification = record(response.artifact_verification);
  const rawFeatures = strings(schema.raw_feature_names, "raw features");
  const requiredFeatures = strings(schema.required_features, "required features");
  const optionalFeatures = strings(schema.optional_features, "optional features");
  const numericalFeatures = strings(schema.numerical_features, "numerical features");
  const categoricalFeatures = strings(schema.categorical_features, "categorical features");
  const rawSet = new Set(rawFeatures);
  const requiredSet = new Set(requiredFeatures);
  const optionalSet = new Set(optionalFeatures);
  const numericalSet = new Set(numericalFeatures);
  const categoricalSet = new Set(categoricalFeatures);
  const roleSet = new Set([...requiredFeatures, ...optionalFeatures]);

  if (!TRAINING_ID_PATTERN.test(expectedTrainingId)
      || response.training_id !== expectedTrainingId
      || response.schema_status !== "complete"
      || !["classification", "regression"].includes(String(response.problem_type))
      || typeof response.target_column !== "string" || !response.target_column
      || !rawFeatures.length || !requiredFeatures.length
      || schema.raw_feature_count !== rawFeatures.length
      || schema.sample_values_stored !== false
      || !sameSet(rawSet, roleSet)
      || requiredFeatures.some(feature => optionalSet.has(feature))
      || numericalFeatures.some(feature => !requiredSet.has(feature) || categoricalSet.has(feature))
      || categoricalFeatures.some(feature => !requiredSet.has(feature) || numericalSet.has(feature))
      || verification.verification_status !== "complete"
      || verification.pipeline_load_verified !== true
      || containsPrivateEvaluationData(response)) {
    throw new ApiError(
      "The prediction schema is incomplete, unsafe, or does not match this verified training run.",
      502,
    );
  }

  const rawDtypes = record(schema.pandas_dtypes);
  const pandasDtypes = Object.fromEntries(
    Object.entries(rawDtypes).flatMap(([feature, dtype]) =>
      rawSet.has(feature) && typeof dtype === "string" ? [[feature, dtype]] : []),
  );

  return {
    training_id: expectedTrainingId,
    schema_status: "complete",
    target_column: response.target_column,
    problem_type: response.problem_type as PredictionProblemType,
    model_name: typeof response.model_name === "string" ? response.model_name : null,
    model_display_name: typeof response.model_display_name === "string"
      ? response.model_display_name : null,
    selected_variant: typeof response.selected_variant === "string"
      ? response.selected_variant : null,
    input_schema: {
      schema_version: String(schema.schema_version || "1.0"),
      schema_source: String(schema.schema_source || "artifact_manifest"),
      raw_feature_count: rawFeatures.length,
      raw_feature_names: rawFeatures,
      required_features: requiredFeatures,
      optional_features: optionalFeatures,
      numerical_features: numericalFeatures,
      categorical_features: categoricalFeatures,
      pandas_dtypes: pandasDtypes,
      sample_values_stored: false,
    },
    artifact_verification: {
      verification_status: "complete",
      pipeline_load_verified: true,
    },
    message: typeof response.message === "string" ? response.message : "",
  };
}

export async function loadPredictionSchema(
  training: TrainingResponse,
  signal: AbortSignal,
): Promise<PredictionSchemaResponse> {
  const trainingId = training.training_id;
  if (!trainingId || !TRAINING_ID_PATTERN.test(trainingId)) {
    throw new ApiError("A valid exported training run is required for prediction.", 400);
  }
  const payload = await requestJson<unknown>(
    `/analysis/predict/schema/${trainingId}`,
    { method: "GET", signal },
    "Unable to load the model input requirements.",
  );
  const schema = validatePredictionSchema(payload, trainingId);
  if (schema.target_column !== training.selected_target) {
    throw new ApiError("The prediction schema does not match the selected target.", 502);
  }
  return schema;
}

export function featureKind(
  schema: PredictionInputSchema,
  feature: string,
): "number" | "boolean" | "text" {
  const dtype = schema.pandas_dtypes[feature] || "";
  if (schema.numerical_features.includes(feature) || numericDtypePattern.test(dtype)) return "number";
  if (booleanDtypePattern.test(dtype)) return "boolean";
  return "text";
}

export function emptyPredictionValues(schema: PredictionInputSchema): PredictionInputValues {
  return Object.fromEntries(schema.raw_feature_names.map(feature => [feature, ""]));
}

export function buildPredictionRecords(
  rows: PredictionInputValues[],
  schema: PredictionInputSchema,
): DataRecord[] {
  if (!rows.length) throw new Error("Add at least one prediction record.");
  const required = new Set(schema.required_features);
  const optional = new Set(schema.optional_features);

  return rows.map((row, rowIndex) => {
    const output: DataRecord = {};
    for (const feature of schema.raw_feature_names) {
      const rawValue = row[feature] ?? "";
      const value = rawValue.trim();
      if (!value && optional.has(feature)) continue;
      if (!value) {
        output[feature] = null;
        continue;
      }
      const kind = featureKind(schema, feature);
      if (kind === "number") {
        const parsed = Number(value);
        if (!Number.isFinite(parsed)) {
          throw new Error(`Record ${rowIndex + 1}: ${feature} must be a finite number or blank.`);
        }
        output[feature] = parsed;
      } else if (kind === "boolean") {
        if (!['true', 'false'].includes(value.toLowerCase())) {
          throw new Error(`Record ${rowIndex + 1}: ${feature} must be true, false, or blank.`);
        }
        output[feature] = value.toLowerCase() === "true";
      } else {
        output[feature] = value;
      }
    }
    for (const feature of required) {
      if (!Object.hasOwn(output, feature)) output[feature] = null;
    }
    return output;
  });
}

function validJsonScalar(value: unknown): boolean {
  return value === null || typeof value === "string" || typeof value === "boolean"
    || (typeof value === "number" && Number.isFinite(value));
}

function validPredictionRow(
  value: unknown,
  index: number,
  problemType: PredictionProblemType,
  probabilityStatus: unknown,
): boolean {
  const row = record(value);
  if (row.row_index !== index || !Object.hasOwn(row, "prediction") || !validJsonScalar(row.prediction)) return false;
  if (problemType === "regression") {
    return typeof row.prediction === "number" && Number.isFinite(row.prediction)
      && !Object.hasOwn(row, "confidence") && !Object.hasOwn(row, "probabilities");
  }
  if (probabilityStatus !== "complete") {
    return !Object.hasOwn(row, "confidence") && !Object.hasOwn(row, "probabilities");
  }
  if (typeof row.confidence !== "number" || !Number.isFinite(row.confidence)
      || row.confidence < 0 || row.confidence > 1) return false;
  const probabilities = items(row.probabilities).map(record);
  return probabilities.length > 0 && probabilities.every(item =>
    validJsonScalar(item.class_label) && typeof item.probability === "number"
      && Number.isFinite(item.probability) && item.probability >= 0 && item.probability <= 1);
}

export function validatePredictionResponse(
  value: unknown,
  schema: PredictionSchemaResponse,
  recordCount: number,
): PredictionResponse {
  const response = record(value);
  const report = record(response.report);
  const predictions = items(report.predictions);
  const verification = record(report.artifact_verification);
  const validProbabilityStatus = schema.problem_type === "regression"
    ? report.probability_status === "not_applicable"
    : ["complete", "disabled", "unavailable"].includes(String(report.probability_status));
  if (response.training_id !== schema.training_id
      || report.prediction_status !== "complete"
      || report.problem_type !== schema.problem_type
      || report.target_column !== schema.target_column
      || (schema.model_name !== null && report.model_name !== schema.model_name)
      || (schema.model_display_name !== null && report.model_display_name !== schema.model_display_name)
      || (schema.selected_variant !== null && report.selected_variant !== schema.selected_variant)
      || report.prediction_count !== recordCount
      || predictions.length !== recordCount
      || !validProbabilityStatus
      || !predictions.every((row, index) => validPredictionRow(
        row, index, schema.problem_type, report.probability_status))
      || verification.verification_status !== "complete"
      || verification.pipeline_load_verified !== true
      || report.source_records_modified !== false
      || containsPrivateEvaluationData(response)) {
    throw new ApiError(
      "The prediction response is incomplete, unsafe, or does not match this model run.",
      502,
    );
  }
  return {
    training_id: schema.training_id,
    report,
    message: typeof response.message === "string" ? response.message : "",
  };
}

export async function predictRecords(
  schema: PredictionSchemaResponse,
  values: PredictionInputValues[],
  includeProbabilities: boolean,
  signal: AbortSignal,
): Promise<PredictionResponse> {
  const records = buildPredictionRecords(values, schema.input_schema);
  const payload = await requestJson<unknown>("/analysis/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      training_id: schema.training_id,
      records,
      include_probabilities: schema.problem_type === "classification" && includeProbabilities,
    }),
  }, "Unable to generate predictions.");
  return validatePredictionResponse(payload, schema, records.length);
}
