# HersiHealth AI extraction pipeline

A single Vercel function (`api/extract.ts`) that Medplum calls whenever a
patient uploads a document. It reads the file, types up what it says, and files
a review `Task` for the patient to confirm.

## How it runs

```
patient uploads → DocumentReference created in Medplum
                     ↓  Subscription (rest-hook, create only)
              POST /api/extract
                     ↓
  1. verify Authorization: Bearer $WEBHOOK_SHARED_SECRET (constant-time)
  2. client-credentials login to Medplum, download the Binary (20 MB cap)
  3. classify   → { document_type, readable }
  4. gate       → unreadable / "unclear" stops here with a "Retake photo" Task
  5. extract    → per-type JSON Schema, every field carries its own confidence
  6. explain    → plain language; never a diagnosis, never treatment advice
  7. write back → Task (status: ready) + DocumentReference.docStatus=preliminary
```

Failures never return 500 — a 500 makes Medplum retry and double-process the
document. They are recorded as a `failed` Task plus
`docStatus: entered-in-error`.

### What the apps read

The pipeline writes **no extension on the DocumentReference**. Everything lands
on a `Task`:

| Field | Value |
| --- | --- |
| `focus` | `DocumentReference/<id>` |
| `for` | the patient |
| `status` | `ready` (awaiting the patient's review), `completed`, `failed` |
| `code.text` | `Review AI-extracted document data`, `Retake photo`, `Extraction failed: …`, `File too large to process` |
| `output` | `documentType`, `extractedData` (JSON), `explanation`, `confidence` |

The web app reads these in `apps/web/src/lib/ai.ts`.

`DiagnosticReport` / `Observation` are deliberately **not** written here — per
the project's FHIR rules those are created only once a human has confirmed the
extraction.

## Setup

1. `cp .env.example .env` and fill it in. The Medplum client must be a
   `ClientApplication` with write access to `Task` and `DocumentReference`.
2. Deploy: `vercel --prod`. Add every variable from `.env` to the Vercel
   project's environment as well.
3. Put the deployed URL in `WEBHOOK_URL`, then connect Medplum to it:

   ```
   npm run setup-subscription
   ```

Step 3 is the one that is easy to forget: without the Subscription the function
is deployed but nothing ever calls it.

## Checks

```
npm run typecheck
```

## Notes

- `AI_PROVIDER` selects `gemini` (default) or `claude`. The key for the
  unselected provider can stay empty — `getProvider()` only throws for the one
  actually in use.
- The Subscription is registered for `create` only. An update-triggered
  subscription would fire on the pipeline's own `docStatus` PATCH and loop
  forever.
