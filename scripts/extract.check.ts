/**
 * Self-checks for the non-trivial pipeline logic — the JSON recovery, the
 * classification narrowing, the retry predicate, and the review flags. These
 * are the places a model's bad day turns into bad data. Run them with:
 *
 *   npm run check
 */

import assert from 'node:assert/strict';
import { parseJsonObject } from '../lib/schemas.ts';
import { coerceClassification, isTransientZaiError } from '../lib/glm.ts';
import { reviewFlags, modelConfidenceOf } from '../lib/review.ts';

// ── parseJsonObject ────────────────────────────────────────────────────────

// Bare JSON — what a model honouring response_format returns.
assert.deepEqual(parseJsonObject('{"document_type":"lab_result","readable":true}'), {
  document_type: 'lab_result',
  readable: true,
});

// Fenced, with and without the language tag.
assert.deepEqual(parseJsonObject('```json\n{"readable":true}\n```'), { readable: true });
assert.deepEqual(parseJsonObject('```\n{"readable":true}\n```'), { readable: true });

// Prose on either side — common on free models that ignore response_format.
assert.deepEqual(
  parseJsonObject('Here is the JSON you asked for:\n{"readable":false}\nHope that helps!'),
  { readable: false },
);

// Leading whitespace / newlines.
assert.deepEqual(parseJsonObject('\n\n  {"readable":true}  \n'), { readable: true });

// Nested braces must not confuse the outermost-brace slice.
const nested = parseJsonObject('```json\n{"results":[{"test_name":"HGB","value":"14.8"}]}\n```');
assert.deepEqual(nested, { results: [{ test_name: 'HGB', value: '14.8' }] });

// Failures must be loud and carry the offending text, so a failed Task is
// debuggable rather than just "undefined".
assert.throws(() => parseJsonObject('I cannot read this document.'), /did not return a JSON object/);
assert.throws(() => parseJsonObject(''), /did not return a JSON object/);
// An array-wrapped object is recovered rather than rejected.
assert.deepEqual(parseJsonObject('[{"a":1}]'), { a: 1 });
assert.throws(() => parseJsonObject('{not valid json}'), SyntaxError);

// ── coerceClassification ───────────────────────────────────────────────────

assert.deepEqual(coerceClassification({ document_type: 'lab_result', readable: true }), {
  document_type: 'lab_result',
  readable: true,
});

// A type the model invented must NOT be forced onto a schema.
assert.equal(coerceClassification({ document_type: 'dental_chart', readable: true }).document_type, 'unclear');
assert.equal(coerceClassification({}).document_type, 'unclear');

// A missing `readable` is not a licence to reject the document.
assert.equal(coerceClassification({ document_type: 'prescription' }).readable, true);
assert.equal(coerceClassification({ document_type: 'prescription', readable: false }).readable, false);

// ── isTransientZaiError ────────────────────────────────────────────────────

// The one this exists for: Z.ai free-tier overload.
assert.equal(isTransientZaiError({ status: 429 }), true);
assert.equal(isTransientZaiError(new Error('429 {"error":{"code":"1305","message":"overloaded"}}')), true);
assert.equal(isTransientZaiError({ status: 503 }), true);
assert.equal(isTransientZaiError(new Error('fetch failed')), true);
// Credentials and quota must fail fast, not after three sleeps.
assert.equal(isTransientZaiError({ status: 401 }), false);
assert.equal(isTransientZaiError({ status: 400 }), false);
assert.equal(isTransientZaiError(new Error('insufficient balance')), false);

// ── reviewFlags ────────────────────────────────────────────────────────────

const NOW = new Date('2026-09-12T00:00:00Z');
const reasonsAt = (flags: ReturnType<typeof reviewFlags>, path: string) =>
  flags.filter((f) => f.path === path).map((f) => f.reason);

// A clean lab result: numeric values with units pass, the physician's name does not.
const lab = reviewFlags(
  {
    test_date: '2026-09-01',
    physician_name: 'Dr A. Yusuf',
    results: [{ test_name: 'Glucose', value: '95', unit: 'mg/dL', confidence: 'high' }],
    overall_confidence: 'high',
  },
  NOW,
);
assert.deepEqual(reasonsAt(lab, 'physician_name'), ['free_text']);
assert.deepEqual(reasonsAt(lab, 'test_date'), []);
assert.deepEqual(reasonsAt(lab, 'results.0.value'), [], 'numeric values with units are the trustworthy class');

// The real failure: a 2026 slip read as 2016, with no hedge in the output.
const misread = reviewFlags({ prescription_date: '2016-04-11' }, NOW);
assert.deepEqual(reasonsAt(misread, 'prescription_date'), ['implausible_date']);

// A date of birth is legitimately decades old and must not be flagged for it.
assert.deepEqual(reviewFlags({ date_of_birth: '1974-02-03' }, NOW), []);

// Two dates on one document that disagree by years: both get flagged.
const disagree = reviewFlags({ date_collected: '2026-09-01', date_reported: '2016-09-03' }, NOW);
assert.deepEqual(reasonsAt(disagree, 'date_collected'), ['inconsistent_date']);
assert.ok(reasonsAt(disagree, 'date_reported').includes('inconsistent_date'));

// The observed eyewear failure: one eye's reading copied into the other's blanks.
const eye = { distance_vision: { sph: -1.25, cyl: -0.5, axis: 180, va: '6/6' }, near_vision: null };
const mirrored = reviewFlags({ right_eye: eye, left_eye: structuredClone(eye) }, NOW);
assert.deepEqual(reasonsAt(mirrored, 'left_eye'), ['mirrored_value']);

// Genuinely different eyes are left alone.
const asymmetric = reviewFlags(
  {
    right_eye: eye,
    left_eye: { distance_vision: { sph: -0.75, cyl: null, axis: null, va: '6/9' }, near_vision: null },
  },
  NOW,
);
assert.deepEqual(reasonsAt(asymmetric, 'left_eye'), []);

// The model's own per-item doubt is surfaced, not averaged away.
const unsure = reviewFlags({ medications: [{ medication_name: 'X', confidence: 'low' }] }, NOW);
assert.ok(reasonsAt(unsure, 'medications.0.confidence').includes('low_model_confidence'));

// ── modelConfidenceOf ──────────────────────────────────────────────────────

assert.equal(modelConfidenceOf({ overall_confidence: 'high' }), 'high');
// Anything missing or unrecognised lands on the pessimistic end.
assert.equal(modelConfidenceOf({}), 'low');
assert.equal(modelConfidenceOf({ overall_confidence: 'very sure' }), 'low');

console.log('pipeline self-check passed');
