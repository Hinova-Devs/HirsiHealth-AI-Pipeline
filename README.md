# HersiHealth AI extraction pipeline

A single Vercel function (`api/extract.ts`) that Medplum calls whenever a
patient uploads a document. It reads the file, types up what it says, marks the
parts it is likely to have got wrong, and files a review `Task` for the patient
to confirm.

It is a **separate service from the app's upload code** on purpose. The upload
path (`hersihealth-mobile/src/lib/document-upload.ts`) finishes the moment the
`DocumentReference` exists; this runs afterwards, asynchronously, off a
Subscription. Extraction can be slow, rate-limited, or completely broken and
the patient's document is still uploaded, encrypted, cached and viewable.

## Files

```
api/extract.ts    the webhook: auth, download, orchestration, Task write-back
lib/schemas.ts    document types, per-type JSON Schemas, the provider contract
lib/glm.ts        Z.ai GLM provider — retry/backoff, JSON recovery, narrowing
lib/review.ts     the draft/confirmed data contract + the code-level defences
scripts/          subscription setup, self-checks
```

Only `api/` is routed; `lib/` is pulled in by import. It exists so a provider
can reach the schemas without importing the handler back.

## How it runs

```
patient uploads → DocumentReference created in Medplum
                     ↓  Subscription (rest-hook, create only)
              POST /api/extract
                     ↓
  1. verify Authorization: Bearer $WEBHOOK_SHARED_SECRET (constant-time)
  2. client-credentials login to Medplum, download the Binary (20 MB cap)
  3. classify   → { document_type, readable }          ← short, cheap prompt
  4. gate       → unreadable → "Retake photo"
                  unclear    → "Categorize manually"   ← never a guessed schema
  5. extract    → the JSON Schema for THAT type, not a generic one
  6. review     → code-level plausibility checks (see below) → reviewFlags
  7. explain    → plain language; never a diagnosis, never treatment advice
  8. write back → Task (status: ready) + DocumentReference.docStatus=preliminary
```

Failures never return 500 — a 500 makes Medplum retry and double-process the
document. They are recorded as a `failed` Task titled *"AI processing
unavailable — please enter details manually"*, with the technical cause in an
`error` output. `docStatus` is deliberately left untouched on failure:
`entered-in-error` would say the *document* should not exist, when all that
failed was the reading of it.

## The two stages, and why they are two

Stage 1 asks one question — which of six categories is this, and is the photo
readable — with a 300-token ceiling and no field extraction. Stage 2 then
extracts against the schema for that specific type. A lab panel and an eyewear
slip share almost no fields; one generic schema would either drop half of each
document or invite the model to fill in fields the page does not have.

When stage 1 returns `unclear`, or returns a category that is not in the enum,
`coerceClassification()` degrades it to `unclear` and the document goes to the
patient for manual categorisation. Forcing a schema onto an unrecognised
document is worse than admitting we do not know: it produces a confident,
wrongly-shaped draft the patient then has to unpick field by field.

## Known model behaviour, and the code that catches it

These come from testing GLM against real sample documents. None of them were
fixed by prompt wording — *"if unreadable, use null"* was tried and the model
kept guessing confidently — so each has a check in `lib/review.ts` that puts
the field in front of the patient. They flag; they never correct.

| Observed | Flag | Check |
| --- | --- | --- |
| Numeric values with units are reliable; names and prose are not | `free_text` | any key matching name/comment/note/impression/finding/… holding a non-empty string |
| A 2026 slip read as 2016, no hedging in the output | `implausible_date` | any date key more than 2 years from today (birth/DOB keys excluded) |
| One document's dates disagreeing with each other | `inconsistent_date` | spread between the document's own dates > 2 years |
| One eye's reading copied into the other eye's **blank** cell | `mirrored_value` | registered parallel sibling keys (`right_eye`/`left_eye`, `distance_vision`/`near_vision`) that serialise identically |
| The model's own per-item doubt | `low_model_confidence` | `confidence: "low"` on any item, surfaced per-field rather than averaged into one number |

Parallel structures are also defended in the schema itself: every optical
reading is `["number", "null"]` inside its own per-eye, per-distance object, so
"blank on this side only" is expressible rather than implied away by symmetry.

## The data contract

`lib/review.ts` defines the two states, and the boundary between them is the
whole point of the pipeline:

- **`DraftExtraction`** — what the AI *said*. Lives on a `Task`, carries
  `extractedData`, `explanation`, `modelConfidence` and `flags`. Never clinical
  truth.
- **`ConfirmedExtraction`** — what the patient *approved*. Carries `values`
  (the patient's version, which may differ from the draft), `confirmedAt`,
  `confirmedBy` and `editedPaths`.

**Only a `ConfirmedExtraction` may become a FHIR clinical resource.** This
pipeline writes no `Observation`, `Condition`, `AllergyIntolerance` or
`DiagnosticReport` — not as a matter of taste, as a hard rule.

The app side of that boundary lives in
`hersihealth-mobile/src/lib/extraction-review.ts`: `parseDraftTask()` reads the
draft off the Task, `confirmedToObservations()` builds the resources from a
confirmed one (lab results only so far). Its types mirror `lib/review.ts` and
the two must be changed together — the Task outputs below are the wire format.
The confirmation screen itself is a separate task.

### What the apps read

The pipeline writes **no extension on the DocumentReference**. Everything lands
on a `Task`:

| Field | Value |
| --- | --- |
| `focus` | `DocumentReference/<id>` |
| `for` | the patient |
| `status` | `ready` (awaiting the patient's review), `completed`, `failed` |
| `code.text` | `Review AI-extracted document data`, `Retake photo`, `Categorize manually`, `AI processing unavailable — please enter details manually`, `File too large to process` |
| `output` | `documentType`, `extractedData` (JSON), `explanation`, `confidence`, `reviewFlags` (JSON), `rawModelResponse` (JSON, audit only), `reason`, `error` |

`rawModelResponse` is the literal text of every model call, kept deliberately
apart from anything the patient later approves — it is there to answer "what
did the AI actually say", and must never be read back as data. It lives on the
`Task` rather than in the function logs because the `Task` is inside the
patient compartment and the logs are not.

## Reliability

- **Retry/backoff.** Z.ai's shared free tier answers `429` with body code
  `1305` ("temporarily overloaded") several times a day. `withRetry()` in
  `lib/glm.ts` gives each stage 3 attempts with 1s then 3s waits, for `429`,
  `5xx`, code `1305` and connection failures. `401`/`403`/`400` fall straight
  through — a bad key should fail in a second, not in five.
- **The SDK's own retries are off** (`maxRetries: 0`). Left on they multiply
  against the above: 6 calls per stage, 18 per document, well past the 60s
  function budget.
- **JSON validation.** `parseJsonObject()` recovers the object from ``` fences
  or surrounding prose before parsing; a parse failure surfaces as the
  "enter details manually" Task, never as a crash.
- **No `response_format`.** The free vision tier does not honour it reliably,
  so the schema is restated in the prompt instead.

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
npm run typecheck   # types
npm run check       # JSON recovery, classification narrowing, retry predicate, review flags
```

## Notes

- `AI_PROVIDER` selects `glm` (default), `gemini`, `claude` or `openrouter`.
  Credentials for the unselected providers can stay empty — `getProvider()`
  only throws for the one actually in use.
- **GLM is vision-only.** A PDF upload is not sent to the model at all; it is
  routed to "Categorize manually" so the patient is asked once, immediately,
  rather than after three failed stages. Point `AI_PROVIDER` at `gemini` or
  `claude` if PDF extraction matters.
- **`glm-4.6v-flash` is free as of 2026-09.** That is a tier decision Z.ai can
  change, and this pipeline sends every patient upload through it — re-check
  z.ai's pricing before shipping, and set `ZAI_MODEL` to override.
- **Free tiers and patient data.** Same caution as OpenRouter's `:free` models:
  a shared free tier may retain what you send, and what you send here is a
  patient's lab slip. Confirm Z.ai's retention terms before pointing this at
  real uploads.
- Model ids get retired. `OPENROUTER_MODEL` is deliberately required rather
  than defaulted, so a dead id fails at setup instead of mid-pipeline. Gemini
  ids are pinned in `GeminiProvider` and need re-pinning when Google retires a
  version — which is exactly how the 2.5 models broke.
- **The package is ESM** (`"type": "module"`), and `tsconfig.json` says
  `nodenext`. @vercel/node reads that same tsconfig, so the two must stay in
  step: a `module` setting that emits ESM without `"type": "module"` kills the
  function at cold start. The header comment in `tsconfig.json` has the full
  story — it is the trap this project has already fallen into once.
- The Subscription is registered for `create` only. An update-triggered
  subscription would fire on the pipeline's own `docStatus` PATCH and loop
  forever.
