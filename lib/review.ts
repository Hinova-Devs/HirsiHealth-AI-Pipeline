/**
 * The draft/confirmed data contract, and the code-level defences that decide
 * what the patient is *made* to look at before confirming.
 *
 * Why these live in code and not in the prompt: every rule below corresponds
 * to a failure the model produced *confidently* against real test documents,
 * and telling it "use null if unreadable" did not stop any of them.
 *
 *  - Numbers with units came back right. Names and free text did not.        → `free_text`
 *  - A 2026 slip was read as 2016 with no hedging in the output.             → `implausible_date`
 *  - One eye's reading was copied into the other eye's blank cell.           → `mirrored_value`
 *
 * Nothing here writes to Medplum, and nothing here is clinical truth. A
 * `DraftExtraction` is what the AI *said*; a `ConfirmedExtraction` is what the
 * patient *approved*. Only the second may ever become an Observation /
 * Condition / DiagnosticReport.
 */

import type { DocumentType } from './schemas.ts';

export type Confidence = 'high' | 'medium' | 'low';

export type ReviewReason =
  /** Name or free-text field — the model's weakest output class. */
  | 'free_text'
  /** Date is implausibly far from today. */
  | 'implausible_date'
  /** Two dates on the same document disagree by more than the tolerance. */
  | 'inconsistent_date'
  /** Parallel structures (two eyes, two rows) came back identical — likely copied. */
  | 'mirrored_value'
  /** The model itself marked this item low confidence. */
  | 'low_model_confidence';

export interface ReviewFlag {
  /** Dotted path into `extractedData`, e.g. `results.0.value`, `right_eye.near_vision.sph`. */
  path: string;
  reason: ReviewReason;
  /** One sentence, safe to show the patient. */
  detail: string;
}

/**
 * Staging state. Produced by the pipeline, stored on a `Task` (never on a
 * clinical resource), consumed by the confirmation screen.
 */
export interface DraftExtraction {
  documentReferenceId: string;
  documentType: DocumentType;
  /** Exactly what the model returned, after JSON recovery. Not clinical truth. */
  extractedData: Record<string, unknown>;
  /** Plain-language summary for the patient. Never a diagnosis. */
  explanation: string;
  /** The model's own overall self-rating. Advisory only. */
  modelConfidence: Confidence;
  /** Fields the confirmation UI must surface for explicit review. */
  flags: ReviewFlag[];
}

/**
 * What the confirmation screen hands back. `values` is the patient's version,
 * not the model's — a field the patient corrected carries their value here and
 * its path is listed in `editedPaths`.
 *
 * This is the ONLY shape a FHIR write may be built from.
 */
export interface ConfirmedExtraction {
  documentReferenceId: string;
  documentType: DocumentType;
  values: Record<string, unknown>;
  /** ISO 8601. */
  confirmedAt: string;
  /** `Patient/<id>` — who pressed confirm. */
  confirmedBy: string;
  /** Dotted paths the patient changed. Everything else was accepted as written. */
  editedPaths: string[];
}

// ============================================================================
// Field classification
// ============================================================================

/** Keys whose values are names or prose. Reliable numbers do not live here. */
const FREE_TEXT_KEY = /name|comment|note|impression|finding|description|recommend|signed|reason/i;

/** Keys that hold a date. Matched on the key, not the value, to avoid parsing "6/6" as a date. */
const DATE_KEY = /date|issued|collected|reported|administered/i;

/** Dates that are *supposed* to be far from today, so the today-distance check must skip them. */
const HISTORICAL_DATE_KEY = /birth|dob/i;

/** Sibling keys that describe two independent sides of the same measurement. */
const PARALLEL_KEYS: readonly (readonly [string, string])[] = [
  ['right_eye', 'left_eye'],
  ['distance_vision', 'near_vision'],
];

const YEAR_MS = 365.25 * 24 * 60 * 60 * 1000;
/** A document more than this far from today, or from its own other dates, gets flagged. */
export const DATE_TOLERANCE_YEARS = 2;

/** Parses a date-ish string. Returns undefined for anything that is not clearly a date. */
function parseDateish(value: unknown): Date | undefined {
  if (typeof value !== 'string' || !/\d{4}/.test(value)) {
    return undefined;
  }
  const ms = Date.parse(value);
  return Number.isNaN(ms) ? undefined : new Date(ms);
}

/** Depth-first walk yielding every leaf and object node with its dotted path. */
function walk(
  node: unknown,
  path: string,
  visit: (path: string, key: string, value: unknown) => void,
): void {
  if (Array.isArray(node)) {
    node.forEach((item, i) => walk(item, path ? `${path}.${i}` : String(i), visit));
    return;
  }
  if (node === null || typeof node !== 'object') {
    return;
  }
  for (const [key, value] of Object.entries(node as Record<string, unknown>)) {
    const childPath = path ? `${path}.${key}` : key;
    visit(childPath, key, value);
    walk(value, childPath, visit);
  }
}

/**
 * Runs every defence over one extraction and returns the fields the patient
 * must look at.
 *
 * `now` is injectable so the self-check does not drift as the clock moves.
 */
export function reviewFlags(extractedData: Record<string, unknown>, now: Date = new Date()): ReviewFlag[] {
  const flags: ReviewFlag[] = [];
  const dates: { path: string; date: Date }[] = [];

  walk(extractedData, '', (path, key, value) => {
    // ── Defence 1: names and free text are the model's weakest output ──────
    if (FREE_TEXT_KEY.test(key) && typeof value === 'string' && value.trim()) {
      flags.push({
        path,
        reason: 'free_text',
        detail: 'Names and written text are often misread. Check this against the document.',
      });
    }

    // ── The model's own per-item confidence, surfaced rather than averaged ──
    if (key === 'confidence' && value === 'low') {
      flags.push({
        path,
        reason: 'low_model_confidence',
        detail: 'The reader was unsure about this entry.',
      });
    }

    // ── Defence 2 (part 1): collect dates for the plausibility checks ──────
    if (DATE_KEY.test(key)) {
      const date = parseDateish(value);
      if (date) {
        dates.push({ path, date });
        if (
          !HISTORICAL_DATE_KEY.test(key) &&
          Math.abs(date.getTime() - now.getTime()) > DATE_TOLERANCE_YEARS * YEAR_MS
        ) {
          flags.push({
            path,
            reason: 'implausible_date',
            detail: `Read as ${date.toISOString().slice(0, 10)}, which is far from today. Digits in years are easy to misread — please confirm.`,
          });
        }
      }
    }

    // ── Defence 3: two sides of a parallel structure came back identical ───
    if (value !== null && typeof value === 'object') {
      const child = value as Record<string, unknown>;
      for (const [a, b] of PARALLEL_KEYS) {
        if (a in child && b in child && JSON.stringify(child[a]) === JSON.stringify(child[b])) {
          flags.push({
            path: `${path ? `${path}.` : ''}${b}`,
            reason: 'mirrored_value',
            detail: `"${b}" is identical to "${a}". The reader sometimes copies one side into the other's blank fields — check whether it really is blank.`,
          });
        }
      }
    }
  });

  // Same check at the root, which `walk` never visits as a node.
  for (const [a, b] of PARALLEL_KEYS) {
    if (
      a in extractedData &&
      b in extractedData &&
      JSON.stringify(extractedData[a]) === JSON.stringify(extractedData[b])
    ) {
      flags.push({
        path: b,
        reason: 'mirrored_value',
        detail: `"${b}" is identical to "${a}". The reader sometimes copies one side into the other's blank fields — check whether it really is blank.`,
      });
    }
  }

  // ── Defence 2 (part 2): dates on one document that disagree with each other.
  // A collection date and a report date sit days apart; two years apart means
  // one of them lost a digit.
  if (dates.length > 1) {
    const times = dates.map((d) => d.date.getTime());
    if (Math.max(...times) - Math.min(...times) > DATE_TOLERANCE_YEARS * YEAR_MS) {
      for (const { path } of dates) {
        flags.push({
          path,
          reason: 'inconsistent_date',
          detail: 'The dates read off this document are years apart from each other. At least one is wrong.',
        });
      }
    }
  }

  return flags;
}

/** Reads the model's self-rating, defaulting to the pessimistic end. */
export function modelConfidenceOf(extractedData: Record<string, unknown>): Confidence {
  const value = extractedData.overall_confidence;
  return value === 'high' || value === 'medium' ? value : 'low';
}
