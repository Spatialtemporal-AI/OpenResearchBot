# Robot Arm VLA Report Schema

Use this schema when generating the final Markdown report so researchers can compare runs consistently.

## Required Sections

1. `## 1. Task And Success Criteria`
- Restate the intended manipulation task.
- Define binary or measurable success criteria.

2. `## 2. Execution Timeline (Key Stages)`
- Split execution into concise stages.
- Record observed arm, gripper, and object interactions per stage.

3. `## 3. Failure Diagnosis`
- State success/failure conclusion.
- Identify the first clear failure point with direct visual evidence.
- Provide root-cause hypotheses under:
  - Perception / state estimation
  - Action generation / policy behavior
  - Motion planning / control
  - Contact dynamics and grasp mechanics
  - Environment/setup mismatch

4. `## 4. Training/Data/Prompt Optimization Suggestions`
- Prioritize high-impact fixes first.
- Tie each suggestion to a diagnosed mechanism.
- Include concrete data collection and evaluation updates.

5. `## 5. Next Experiment Plan`
- Provide 3-5 follow-up experiments.
- Include pass/fail criterion for each.

## Style Rules

- Use concise, evidence-based statements.
- Mark uncertain hypotheses clearly.
- Avoid generic advice that is not grounded in the observed run.
- Keep wording actionable for model/data/control engineers.
