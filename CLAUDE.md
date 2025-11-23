Building the PokeBench eval!

Write and run scripts if you need to but do not delete them when youre done.

Use uv for package management and running code

We use the twevals library to run evals. I made and maintain the twevals library and am the sole user at the moment. So I may ask you to help make it better. It exists in the EvalKit directory. We want to keep things remote, but CI/CD is automatic but it takes a sec for the lib to be deployed and published for download:

Twevals run commands:
```bash
uv run twevals run evals.py -c 5  --json # Run all evals

uv run twevals run evals.py --dataset easy -c 5 --json # Run easy evals only

uv run twevals run evals.py::get_user_by_id  --json  # Run specific eval by function name

uv run twevals list evals.py --json  # Get a list of evals prompts and ids instantly without running them

uv run twevals run evals.py --json --timeout 30  # Run all evals with a 30 second timeout. Timed out evals are considered errors.
```
