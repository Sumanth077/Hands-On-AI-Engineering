import { evaluate } from "eve/ai";

export const JEV_MIN_CONFIDENCE = 0.70;

export interface SqlReview {
  check: "SQL relevance";
  verdict: "relevant" | "mismatch" | "unavailable";
  confidence: number | null;
  issue: string;
  actionable: boolean;
  status: "ok" | "low_confidence" | "inconsistent" | "error";
}

type Evaluate = typeof evaluate;

const issueText: Record<string, string> = {
  none: "The SQL addresses the question.",
  wrong_metric: "The SQL calculates a different metric from the one requested.",
  wrong_filter: "The SQL filters the wrong records or time range.",
  wrong_grouping: "The SQL groups or compares the data differently from the question.",
  wrong_entities: "The SQL uses the wrong tables, entities, or join relationship.",
};

export async function reviewSqlRelevance(
  state: { question: string; sql: string; databaseSchema: unknown },
  evaluateFn: Evaluate = evaluate,
): Promise<SqlReview> {
  try {
    const result = await evaluateFn({
      model: process.env.JEV_MODEL ?? "typesafe-ai/jev",
      state,
      maxRetries: 1,
      abortSignal: AbortSignal.timeout(Number(process.env.JEV_TIMEOUT_SECONDS ?? "10") * 1000),
      questions: {
        verdict: {
          type: "choice",
          instructions: "Judge whether the proposed SQLite SELECT directly addresses the user's data question using the supplied schema.",
          criteria: {
            relevant: "The query's metric, filters, grouping, entities, and requested detail match the question.",
            mismatch: "A material part of the query does not match the question.",
          },
        },
        issue: {
          type: "choice",
          instructions: "Select the single most important mismatch, or none.",
          criteria: {
            none: issueText.none,
            wrong_metric: issueText.wrong_metric,
            wrong_filter: issueText.wrong_filter,
            wrong_grouping: issueText.wrong_grouping,
            wrong_entities: issueText.wrong_entities,
          },
        },
      },
    });
    const verdict = result.answers.verdict.choice as "relevant" | "mismatch";
    const confidence = result.answers.verdict.confidence;
    const issueKey = result.answers.issue.choice;
    const threshold = Number(process.env.JEV_MIN_CONFIDENCE ?? JEV_MIN_CONFIDENCE);
    const inconsistent = (verdict === "mismatch") === (issueKey === "none");
    const issue = inconsistent
      ? verdict === "mismatch"
        ? "Jev reported a mismatch without identifying a specific issue; the query proceeded as inconclusive."
        : issueText.none
      : issueText[issueKey] ?? "Review the proposed SQL against the question.";
    return {
      check: "SQL relevance",
      verdict,
      confidence,
      issue,
      actionable: verdict === "mismatch" && confidence >= threshold && !inconsistent,
      status: inconsistent ? "inconsistent" : confidence >= threshold ? "ok" : "low_confidence",
    };
  } catch (error) {
    return {
      check: "SQL relevance",
      verdict: "unavailable",
      confidence: null,
      issue: `Review unavailable: ${error instanceof Error ? error.message : String(error)}`,
      actionable: false,
      status: "error",
    };
  }
}
