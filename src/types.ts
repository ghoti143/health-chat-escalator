export type EscalationDecision = {
  escalate: boolean;
  category: string | null;
  confidence: "high" | "medium" | "low";
  reasoning: string;
};
