# AI Workbench

```bash
uv sync --extra dev
uv run ai-workbench
```

Enter an absolute project directory, click **Open project**, select a few text
files, and inspect the context preview. You can also use **Browse / Open Folder**
or reopen a recent project. The last valid project and model selection are restored
on startup. Enter a prompt and use **ASK**, **CHANGE**, or **CONTINUE**; proposed
changes still require explicit **Accept** or **Reject** review.

On first launch, model templates and an empty secrets file are created under
`~/.config/ai-workbench/`. Add `DEEPSEEK_API_KEY=...` to `secrets.env` (or export
that environment variable), then choose a configured model. The UI reports only
whether the credential is available and never displays its value.

Run tests with `uv run pytest`. If `uv` is unavailable, a standard virtual
environment and `pip install -e '.[dev]'` work as well.
