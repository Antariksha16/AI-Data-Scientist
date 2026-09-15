import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { loadSource } from "./load-source.mjs";

const {
  buildPredictionRecords,
  emptyPredictionValues,
  featureKind,
  loadPredictionSchema,
  predictRecords,
  validatePredictionResponse,
  validatePredictionSchema,
} = loadSource("../src/api/prediction.ts");
const { PredictionResult, PredictionWorkspace } = loadSource("../src/components/PredictionPanel.tsx");

const classificationId = "d".repeat(32);
const regressionId = "e".repeat(32);

function schemaFixture(problemType = "classification") {
  const classification = problemType === "classification";
  return {
    training_id: classification ? classificationId : regressionId,
    schema_status: "complete",
    target_column: classification ? "Outcome" : "Salary",
    problem_type: problemType,
    model_name: classification ? "logistic_regression" : "ridge",
    model_display_name: classification ? "Logistic Regression" : "Ridge Regression",
    selected_variant: "tuned",
    input_schema: {
      schema_version: "1.0", schema_source: "artifact_manifest", raw_feature_count: 4,
      raw_feature_names: ["Age", "Department", "Active", "RecordID"],
      required_features: ["Age", "Department", "Active"], optional_features: ["RecordID"],
      numerical_features: ["Age"], categorical_features: ["Department", "Active"],
      pandas_dtypes: { Age: "float64", Department: "str", Active: "bool", RecordID: "int64" },
      sample_values_stored: false,
    },
    artifact_verification: { verification_status: "complete", pipeline_load_verified: true },
    message: "Supply every required feature.",
  };
}

function trainingFixture(problemType = "classification") {
  const schema = schemaFixture(problemType);
  return {
    filename: "dataset.csv", selected_target: schema.target_column,
    training_id: schema.training_id, report: { training_status: "complete", target_column: schema.target_column },
    artifacts: { artifact_status: "complete" }, message: "Training completed.",
  };
}

function predictionFixture(problemType = "classification") {
  const schema = schemaFixture(problemType);
  const classification = problemType === "classification";
  return {
    training_id: schema.training_id,
    report: {
      prediction_status: "complete", problem_type: problemType, target_column: schema.target_column,
      model_name: schema.model_name, model_display_name: schema.model_display_name, selected_variant: "tuned",
      prediction_count: 2, probability_status: classification ? "complete" : "not_applicable",
      class_labels: classification ? [0, 1] : [],
      predictions: classification ? [
        { row_index: 0, prediction: 0, confidence: 0.8,
          probabilities: [{ class_label: 0, probability: 0.8 }, { class_label: 1, probability: 0.2 }] },
        { row_index: 1, prediction: 1, confidence: 0.7,
          probabilities: [{ class_label: 0, probability: 0.3 }, { class_label: 1, probability: 0.7 }] },
      ] : [
        { row_index: 0, prediction: 45000.25 }, { row_index: 1, prediction: 61200.75 },
      ],
      artifact_verification: { verification_status: "complete", pipeline_load_verified: true },
      source_records_modified: false,
    },
    message: "Predictions generated.",
  };
}

const jsonResponse = (value, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { "Content-Type": "application/json" },
});

test("verified schema is loaded only from the current training identifier", async t => {
  let url;
  t.mock.method(globalThis, "fetch", async requested => { url = requested; return jsonResponse(schemaFixture()); });
  const result = await loadPredictionSchema(trainingFixture(), new AbortController().signal);
  assert.match(String(url), new RegExp(`/analysis/predict/schema/${classificationId}$`));
  assert.equal(result.training_id, classificationId);
  assert.equal(result.artifact_verification.pipeline_load_verified, true);
  assert.equal(result.input_schema.sample_values_stored, false);
});

test("schema rejects mismatched identity, private evaluation data, and stored samples", () => {
  const mismatch = schemaFixture(); mismatch.training_id = "a".repeat(32);
  assert.throws(() => validatePredictionSchema(mismatch, classificationId), /incomplete, unsafe/);
  const privateSchema = schemaFixture(); privateSchema.evaluation_context = { X_holdout: [{ Age: 20 }] };
  assert.throws(() => validatePredictionSchema(privateSchema, classificationId), /incomplete, unsafe/);
  const samples = schemaFixture(); samples.input_schema.sample_values_stored = true;
  assert.throws(() => validatePredictionSchema(samples, classificationId), /incomplete, unsafe/);
});

test("schema feature roles must be complete, disjoint, and internally consistent", () => {
  const missingRole = schemaFixture(); missingRole.input_schema.optional_features = [];
  assert.throws(() => validatePredictionSchema(missingRole, classificationId), /incomplete, unsafe/);
  const typedOptional = schemaFixture(); typedOptional.input_schema.numerical_features.push("RecordID");
  assert.throws(() => validatePredictionSchema(typedOptional, classificationId), /incomplete, unsafe/);
  const duplicate = schemaFixture(); duplicate.input_schema.raw_feature_names.push("Age");
  assert.throws(() => validatePredictionSchema(duplicate, classificationId), /duplicate/);
});

test("model target must match the training response target", async t => {
  t.mock.method(globalThis, "fetch", async () => jsonResponse(schemaFixture()));
  const training = trainingFixture(); training.selected_target = "DifferentTarget";
  await assert.rejects(loadPredictionSchema(training, new AbortController().signal), /does not match/);
});

test("input kinds use fitted numerical, boolean, and dtype metadata", () => {
  const input = schemaFixture().input_schema;
  assert.equal(featureKind(input, "Age"), "number");
  assert.equal(featureKind(input, "Active"), "boolean");
  assert.equal(featureKind(input, "Department"), "text");
  assert.equal(featureKind(input, "RecordID"), "number");
});

test("records convert numerical and boolean values while omitting blank optional fields", () => {
  const input = schemaFixture().input_schema;
  const rows = [{ Age: " 34.5 ", Department: " Sales ", Active: "true", RecordID: "" },
    { Age: "", Department: "", Active: "false", RecordID: "98" }];
  assert.deepEqual(buildPredictionRecords(rows, input), [
    { Age: 34.5, Department: "Sales", Active: true },
    { Age: null, Department: null, Active: false, RecordID: 98 },
  ]);
  assert.deepEqual(Object.keys(emptyPredictionValues(input)), input.raw_feature_names);
});

test("invalid numerical values are blocked before an API request", () => {
  assert.throws(() => buildPredictionRecords([
    { Age: "not-a-number", Department: "IT", Active: "true", RecordID: "" },
  ], schemaFixture().input_schema), /Record 1: Age must be a finite number/);
});

test("classification prediction sends bounded raw records and probability preference", async t => {
  let request;
  t.mock.method(globalThis, "fetch", async (_, init) => { request = JSON.parse(init.body); return jsonResponse(predictionFixture()); });
  const schema = validatePredictionSchema(schemaFixture(), classificationId);
  const values = [
    { Age: "20", Department: "IT", Active: "true", RecordID: "" },
    { Age: "55", Department: "HR", Active: "false", RecordID: "2" },
  ];
  const result = await predictRecords(schema, values, true, new AbortController().signal);
  assert.equal(request.training_id, classificationId); assert.equal(request.include_probabilities, true);
  assert.deepEqual(request.records[0], { Age: 20, Department: "IT", Active: true });
  assert.equal(result.report.prediction_count, 2);
});

test("regression prediction always disables probability calculation", async t => {
  let request;
  t.mock.method(globalThis, "fetch", async (_, init) => { request = JSON.parse(init.body); return jsonResponse(predictionFixture("regression")); });
  const schema = validatePredictionSchema(schemaFixture("regression"), regressionId);
  const values = [
    { Age: "20", Department: "IT", Active: "true", RecordID: "" },
    { Age: "55", Department: "HR", Active: "false", RecordID: "" },
  ];
  const result = await predictRecords(schema, values, true, new AbortController().signal);
  assert.equal(request.include_probabilities, false);
  assert.equal(result.report.probability_status, "not_applicable");
});

test("prediction response must match model, row count, verification, and immutability", () => {
  const schema = validatePredictionSchema(schemaFixture(), classificationId);
  const mismatch = predictionFixture(); mismatch.training_id = "a".repeat(32);
  assert.throws(() => validatePredictionResponse(mismatch, schema, 2), /incomplete, unsafe/);
  const wrongRows = predictionFixture(); wrongRows.report.prediction_count = 3;
  assert.throws(() => validatePredictionResponse(wrongRows, schema, 2), /incomplete, unsafe/);
  const modified = predictionFixture(); modified.report.source_records_modified = true;
  assert.throws(() => validatePredictionResponse(modified, schema, 2), /incomplete, unsafe/);
});

test("malformed probability output and private holdout content are rejected", () => {
  const schema = validatePredictionSchema(schemaFixture(), classificationId);
  const badProbability = predictionFixture(); badProbability.report.predictions[0].confidence = 2;
  assert.throws(() => validatePredictionResponse(badProbability, schema, 2), /incomplete, unsafe/);
  const privateResult = predictionFixture(); privateResult.report.X_holdout = [{ secret: 1 }];
  assert.throws(() => validatePredictionResponse(privateResult, schema, 2), /incomplete, unsafe/);
});

test("schema-driven workspace renders required, optional, batch, and probability controls", () => {
  const schema = validatePredictionSchema(schemaFixture(), classificationId);
  const html = renderToStaticMarkup(createElement(PredictionWorkspace, {
    schema, values: [emptyPredictionValues(schema.input_schema)], includeProbabilities: true,
    busy: false, onValuesChange: () => {}, onIncludeProbabilitiesChange: () => {},
    onPredict: () => {}, onCancel: () => {},
  }));
  for (const text of ["Verified fitted model", "Logistic Regression", "Prediction record 1", "Age",
    "Department", "Active", "Optional source fields (1)", "RecordID", "Add another record",
    "Show class probabilities and confidence", "Generate predictions"]) assert.ok(html.includes(text), text);
  assert.doesNotMatch(html, /sample value|prediction-0-Outcome/i);
});

test("classification results show predictions, confidence, probabilities, and verification", () => {
  const response = validatePredictionResponse(predictionFixture(),
    validatePredictionSchema(schemaFixture(), classificationId), 2);
  const html = renderToStaticMarkup(createElement(PredictionResult, { response }));
  for (const text of ["Outcome predictions", "Logistic Regression", "Record 1", "Predicted Outcome",
    "80.00% confidence", "Class 0", "Class 1", "Verified model pipeline", "source records unchanged"])
    assert.ok(html.includes(text), text);
});

test("regression results present continuous estimates without probability controls", () => {
  const response = validatePredictionResponse(predictionFixture("regression"),
    validatePredictionSchema(schemaFixture("regression"), regressionId), 2);
  const html = renderToStaticMarkup(createElement(PredictionResult, { response }));
  assert.match(html, /Salary predictions/); assert.match(html, /45,000\.25/);
  assert.match(html, /continuous numerical estimate/); assert.doesNotMatch(html, /Class probabilities|confidence/);
});

test("workflow exposes prediction only after one exported training result", () => {
  const app = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8");
  const training = readFileSync(new URL("../src/components/TrainingPanel.tsx", import.meta.url), "utf8");
  assert.match(app, /trainingResult \? 5 : trainingComplete \? 4/);
  assert.match(app, /trainingResult && <PredictionPanel/);
  assert.match(app, /onResultChange=\{setTrainingResult\}/);
  assert.match(training, /onResultChange\?\.\(null\)/);
  assert.match(training, /onResultChange\?\.\(response\)/);
});
