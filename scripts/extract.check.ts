/**
 * Self-check for parseJsonObject — the one place a weaker model's formatting
 * quirks turn into a failed extraction. Run it with:
 *
 *   node --experimental-strip-types scripts/extract.check.ts
 */

import assert from 'node:assert/strict';
import { parseJsonObject } from '../api/extract.ts';

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

console.log('extract.ts self-check passed');
