import assert from "node:assert/strict";
import test from "node:test";

import { reviewSqlRelevance } from "../agent/lib/jev.ts";
import { boundQuestionFor, parseAnalystMessage } from "../agent/lib/question_context.ts";

function evaluator(verdict: "relevant" | "mismatch", confidence: number, issue: string) {
  return async () => ({
    answers: {
      verdict: { type: "choice", choice: verdict, confidence, probabilities: {} },
      issue: { type: "choice", choice: issue, confidence: 0.9, probabilities: {} },
    },
  }) as any;
}

const state = {
  question: "Which region has the most revenue?",
  sql: "SELECT region, SUM(total) FROM orders GROUP BY region",
  databaseSchema: [{ table: "orders", columns: [{ name: "region", type: "TEXT" }] }],
};

test("relevant SQL proceeds", async () => {
  const review = await reviewSqlRelevance(state, evaluator("relevant", 0.94, "none"));
  assert.equal(review.actionable, false);
  assert.equal(review.status, "ok");
});

test("confident SQL mismatch blocks execution with a revision reason", async () => {
  const review = await reviewSqlRelevance(state, evaluator("mismatch", 0.91, "wrong_metric"));
  assert.equal(review.actionable, true);
  assert.match(review.issue, /different metric/i);
});

test("low-confidence SQL mismatch proceeds", async () => {
  const review = await reviewSqlRelevance(state, evaluator("mismatch", 0.30, "wrong_filter"));
  assert.equal(review.actionable, false);
  assert.equal(review.status, "low_confidence");
});

test("Jev failure proceeds without authorizing anything", async () => {
  const review = await reviewSqlRelevance(state, (async () => { throw new Error("offline"); }) as any);
  assert.equal(review.actionable, false);
  assert.equal(review.status, "error");
});

test("SQL review question is bound to the authoritative Eve turn message", () => {
  const binding = parseAnalystMessage(
    "Database ID: demo\nQuestion: Which region has the most completed-order revenue?",
    "turn_7",
  );
  assert.equal(
    boundQuestionFor(binding, "demo", "turn_7"),
    "Which region has the most completed-order revenue?",
  );
  assert.throws(() => boundQuestionFor(binding, "demo", "turn_spoofed"), /not bound/);
  assert.throws(() => boundQuestionFor(binding, "other", "turn_7"), /not bound/);
});

test("mismatch without an issue is inconclusive and does not block", async () => {
  const review = await reviewSqlRelevance(state, evaluator("mismatch", 0.96, "none"));
  assert.equal(review.status, "inconsistent");
  assert.equal(review.actionable, false);
  assert.match(review.issue, /inconclusive/i);
});

test("relevant verdict ignores a contradictory mismatch issue", async () => {
  const review = await reviewSqlRelevance(state, evaluator("relevant", 0.96, "wrong_metric"));
  assert.equal(review.status, "inconsistent");
  assert.equal(review.actionable, false);
  assert.equal(review.issue, "The SQL addresses the question.");
});
