llm-bench v0.1
Goal: one command runs a model several times and reports TTFT and tokens per second as p50 and p95, logging every run as JSON. You write all the code; say "hint" or "reveal" in chat when stuck. Work on a branch named v0.1.
Target layout:
llm-bench/
  bench/__init__.py
  bench/client.py     # talks to the model server
  bench/stats.py      # pure calculations
  bench/run.py        # the command-line tool
  tests/test_stats.py
  results/            # JSON lines output, add to .gitignore
1. Client. In bench/client.py write a function that sends a streaming request to /api/generate and returns a dictionary with: ttft_s (time to first text chunk), total_s, eval_count, eval_duration_s, load_duration_s, and tokens_per_s (eval_count divided by eval_duration in seconds). Pass options with num_predict and num_ctx. Use time.perf_counter() for timing and a timeout on the request.
2. Settings. Read the server address from the OLLAMA_URL environment variable, defaulting to http://localhost:11434.
3. Command line. In bench/run.py use argparse for --model, --prompt, --runs (default 5), --num-predict (default 128), --num-ctx (default 2048), --label (a hardware label such as laptop-mx250), and --out (default results/runs.jsonl). Run it with python -m bench.run.
4. Warm-up. Send one request first and leave it out of the results. Log that it happened.
5. JSON lines. After each run, append one JSON object to the output file: timestamp, model, label, all settings, and every number from step 1.
6. Summary and errors. In bench/stats.py write a function that takes a list of numbers and returns p50 and p95. Print a short summary table at the end. Catch connection errors and timeouts, log them with logging.error, and exit with code 1.
7. Tests. In tests/test_stats.py write at least three pytest tests for the stats function: a known list gives the expected median, p95 is never smaller than p50, and an empty list raises an error. Run pytest. Commit, push, open a PR, merge.
Done means:
• python -m bench.run --model llama3.2:1b --runs 5 --label laptop-mx250 prints p50 and p95 for TTFT and tokens per second.
• results/runs.jsonl has 5 new lines.
• pytest passes.
• With Ollama stopped, the tool prints a clear error and echo $LASTEXITCODE in PowerShell shows 1.
• The README has a "Results" table with your numbers, and the v0.1 PR is merged