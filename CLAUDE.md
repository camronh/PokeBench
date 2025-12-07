Building the PokeBench eval!

Write and run scripts if you need to but do not delete them when youre done.

Use uv for package management and running code

We use the ezvals library to run evals. I made and maintain the ezvals library and am the sole user at the moment. So I may ask you to help make it better. It exists in the evalkit directory.

Ezvals run commands:
```bash
uv run ezvals run evals.py -c 5  # Run all evals

uv run ezvals run evals.py --dataset easy -c 5  # Run easy evals only

uv run ezvals run evals.py::get_user_by_id  # Run specific eval by function name

uv run ezvals run evals.py -c 5 --timeout 30  # Run all evals with a 30 second timeout. Timed out evals are considered errors.

uv run ezvals run evals.py --limit 5  # Run only the first 5 evals

uv run ezvals serve evals.py  # Serve web UI to browse results
```
