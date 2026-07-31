# Chapter 6: Guardrails, Safety, and Human Approval

## Why This Chapter Matters
Production agents need explicit safety boundaries. Prompting alone is not enough for policy compliance.

## Learning Objectives
- Implement input and output guardrails.
- Define escalation paths for sensitive actions.
- Design human-in-the-loop approval checkpoints.

## Core Concepts
- Policy validation and blocked categories.
- Sensitive data detection and redaction.
- Confidence thresholds for automated actions.
- Audit trails for compliance.

## Practical Framework
Safety pipeline:
1. Pre-check user input.
2. Block or sanitize unsafe content.
3. Validate model output against policy.
4. Route high-risk actions to human approval.

## Exercise
Create a risk matrix for actions:
- Low risk: auto-execute.
- Medium risk: notify + log.
- High risk: require human approval.

## Output
A guardrail policy table and escalation workflow.
