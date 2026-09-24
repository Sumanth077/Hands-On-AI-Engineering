import { defineAgent } from "eve";

export default defineAgent({
  model: "alibaba/qwen3.8-omni-flash",
  defaultTools: false,
});
