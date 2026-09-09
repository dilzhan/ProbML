# Contributing

Corrections to derivations, clearer explanations, reproducibility fixes, and
implementation improvements are welcome.

## Reporting an issue

Include the notebook name and section or the relevant function in
`src/gaussian.py`. For execution problems, include your Python version, operating
system, relevant package versions, traceback, and whether the error occurs after
restarting the kernel and running all preceding cells. For mathematical issues,
identify the equation and explain the proposed correction.

## Making a change

1. Follow the environment setup in [README.md](README.md).
2. Keep each pull request focused on one correction or improvement.
3. For code changes, run the affected examples from a fresh kernel. Changes to
   `src/gaussian.py` may affect several notebooks; check the relevant consumers.
4. Run `python scripts/check_repository.py`.
5. Describe the change and exactly what you executed, including any checks that
   remain pending.

Preserve the notebooks' explanatory style and explicit random keys. Keep saved
outputs useful for readers, and avoid unrelated output or metadata changes.
Exclude local environments, credentials, and generated experiment artifacts.

The automated checks validate syntax and notebook structure only. Long training
and sampling runs are manual; state their validation status in the pull request.
