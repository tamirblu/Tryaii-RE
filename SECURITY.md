# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in TryAii-DRE, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please email: **security@tryaii.com**

Include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge your report within 48 hours and aim to release a fix within 7 days for critical issues.

## Scope

- TryAii-DRE core routing engine (Python and Node packages)
- SDK packages (tryaii-dre-sdk)
- OpenRouter integration

## Best Practices

- Never commit API keys or secrets. Use environment variables.
- The `.env` file is gitignored by default. Keep it that way.
- When using the OpenRouter integration, store your API key in `OPENROUTER_API_KEY` env var, not in code.
