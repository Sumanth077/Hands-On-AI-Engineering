import { defineState } from "eve/context";

export interface QuestionBinding {
  databaseId: string;
  question: string;
  turnId: string;
}

export const currentQuestion = defineState<QuestionBinding | null>(
  "nl-data-analyst.current-question",
  () => null,
);

export function parseAnalystMessage(message: string, turnId: string): QuestionBinding | null {
  const match = /^Database ID: ([^\r\n]+)\r?\nQuestion: ([\s\S]+)$/.exec(message);
  if (!match) return null;
  const databaseId = match[1].trim();
  const question = match[2].trim();
  if (!databaseId || !question) return null;
  return { databaseId, question, turnId };
}

export function boundQuestionFor(
  binding: QuestionBinding | null,
  databaseId: string,
  turnId: string,
): string {
  if (!binding || binding.databaseId !== databaseId || binding.turnId !== turnId) {
    throw new Error("The SQL call is not bound to the current user question. Start a new analysis turn.");
  }
  return binding.question;
}
