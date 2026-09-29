import { defineHook } from "eve/hooks";
import { currentQuestion, parseAnalystMessage } from "../lib/question_context";

export default defineHook({
  events: {
    "message.received"(event) {
      if (event.data.kind === "execution.background_task") return;
      const binding = parseAnalystMessage(event.data.message, event.data.turnId);
      currentQuestion.update(() => binding);
    },
  },
});
