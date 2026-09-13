/**
 * Document types, per-type JSON Schemas, and the provider contract.
 *
 * Split out of `api/extract.ts` so a provider implementation (see
 * `lib/glm.ts`) can reach the schemas without importing the handler back —
 * that would be a cycle. Nothing here does I/O.
 */

// ============================================================================
// Document types + JSON Schemas
//
// These are plain, portable JSON Schema objects (lowercase `type` strings),
// reused verbatim by both providers below:
//  - Gemini's `GenerateContentConfig.responseSchema` field is typed
//    `SchemaUnion = Schema | unknown` in the current @google/genai SDK, so
//    these plain JSON Schema objects are assignable with no cast needed.
//  - Claude's `Tool.InputSchema` is `{ type: 'object'; properties?: unknown;
//    required?: string[]; [k: string]: unknown }` — structurally open enough
//    to accept these objects directly, modulo a top-level cast.
//
// Every extracted field carries its own "confidence" (high/medium/low)
// because source documents are often handwritten, partially illegible, or
// mixed Somali/English — the patient-review UI needs to know which fields
// to flag, not just get a single blended confidence.
// ============================================================================

export type DocumentType =
  | 'lab_result'
  | 'prescription'
  | 'diagnosis'
  | 'imaging_report'
  | 'vaccination_card'
  | 'optical_prescription'
  | 'unclear';

const CONFIDENCE_ENUM = ['high', 'medium', 'low'];

/** Schema for `provider.classify()`. Shared by both providers. */
export const classifySchema = {
  type: 'object',
  properties: {
    document_type: {
      type: 'string',
      enum: [
        'lab_result',
        'prescription',
        'diagnosis',
        'imaging_report',
        'vaccination_card',
        'optical_prescription',
        'unclear',
      ],
      description:
        'Use "optical_prescription" for eyewear/refraction slips (sph/cyl/axis per eye), not "prescription". ' +
        'Use "unclear" whenever the document does not clearly belong to one of the other categories — ' +
        'a wrong category is worse than none, because the patient then re-types the whole document.',
    },
    readable: {
      type: 'boolean',
      description: 'False if the document is too blurry, dark, cropped, or incomplete to extract data from.',
    },
    reason_if_unreadable: {
      type: 'string',
      description: 'Short reason the document could not be read, only present when readable is false.',
    },
  },
  required: ['document_type', 'readable'],
};

const labResultSchema = {
  type: 'object',
  properties: {
    test_date: { type: 'string', description: 'ISO 8601 date if determinable' },
    results: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          test_name: { type: 'string' },
          value: { type: 'string' },
          unit: { type: 'string' },
          reference_range: { type: 'string' },
          confidence: { type: 'string', enum: CONFIDENCE_ENUM },
        },
        required: ['test_name', 'value', 'confidence'],
      },
    },
    overall_confidence: { type: 'string', enum: CONFIDENCE_ENUM },
  },
  required: ['results', 'overall_confidence'],
};

const prescriptionSchema = {
  type: 'object',
  properties: {
    prescriber_name: { type: 'string', description: 'Name of the prescribing clinician, if legible' },
    prescription_date: { type: 'string', description: 'ISO 8601 date if determinable' },
    medications: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          medication_name: { type: 'string' },
          dosage: { type: 'string', description: 'e.g. "500mg"' },
          frequency: { type: 'string', description: 'e.g. "twice daily"' },
          route: { type: 'string', description: 'e.g. "oral", "topical"' },
          duration: { type: 'string', description: 'e.g. "7 days"' },
          quantity: { type: 'string' },
          refills: { type: 'string' },
          confidence: { type: 'string', enum: CONFIDENCE_ENUM },
        },
        required: ['medication_name', 'confidence'],
      },
    },
    overall_confidence: { type: 'string', enum: CONFIDENCE_ENUM },
  },
  required: ['medications', 'overall_confidence'],
};

const diagnosisSchema = {
  type: 'object',
  properties: {
    diagnosis_date: { type: 'string', description: 'ISO 8601 date if determinable' },
    clinician_name: { type: 'string' },
    diagnoses: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          condition_name: { type: 'string' },
          icd10_code: {
            type: 'string',
            description:
              'Only include if an ICD-10 code is literally printed on the document. ' +
              'Never infer, guess, or generate a code that is not explicitly written.',
          },
          notes: { type: 'string' },
          confidence: { type: 'string', enum: CONFIDENCE_ENUM },
        },
        required: ['condition_name', 'confidence'],
      },
    },
    overall_confidence: { type: 'string', enum: CONFIDENCE_ENUM },
  },
  required: ['diagnoses', 'overall_confidence'],
};

const imagingReportSchema = {
  type: 'object',
  properties: {
    study_date: { type: 'string', description: 'ISO 8601 date if determinable' },
    modality: { type: 'string', description: 'e.g. "X-ray", "MRI", "CT", "Ultrasound"' },
    body_region: { type: 'string', description: 'e.g. "chest", "left knee"' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          description: { type: 'string' },
          confidence: { type: 'string', enum: CONFIDENCE_ENUM },
        },
        required: ['description', 'confidence'],
      },
    },
    impression: {
      type: 'string',
      description: "The radiologist's summary impression, transcribed verbatim if present",
    },
    radiologist_name: { type: 'string' },
    overall_confidence: { type: 'string', enum: CONFIDENCE_ENUM },
  },
  required: ['findings', 'overall_confidence'],
};

const vaccinationCardSchema = {
  type: 'object',
  properties: {
    vaccinations: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          vaccine_name: { type: 'string' },
          dose_number: { type: 'string', description: 'e.g. "1", "2", "booster"' },
          date_administered: { type: 'string', description: 'ISO 8601 date if determinable' },
          lot_number: { type: 'string' },
          administering_facility: { type: 'string' },
          confidence: { type: 'string', enum: CONFIDENCE_ENUM },
        },
        required: ['vaccine_name', 'confidence'],
      },
    },
    overall_confidence: { type: 'string', enum: CONFIDENCE_ENUM },
  },
  required: ['vaccinations', 'overall_confidence'],
};

/**
 * Eyewear / refraction slip.
 *
 * Two things this schema does deliberately differently:
 *
 *  - Every reading is `["number", "null"]`, and each eye/distance block is
 *    its own object. The model has been observed copying one eye's reading
 *    into the other eye's *blank* field, so the shape has to make "blank"
 *    expressible per side rather than implying symmetry. `lib/review.ts`
 *    additionally flags right/left blocks that come back byte-identical.
 *  - `date` and `age` are extracted but never trusted: the same slip that
 *    produced correct sph/cyl values returned 2016 for a 2026 date. See the
 *    date plausibility check in `lib/review.ts`.
 */
const eyeReadingSchema = {
  type: 'object',
  description:
    'One eye at one viewing distance. Every field is independent — if a cell is blank on the ' +
    'document, return null for that cell. Never copy a value from the other eye or the other row.',
  properties: {
    sph: { type: ['number', 'null'], description: 'Sphere, e.g. -1.25. null if blank.' },
    cyl: { type: ['number', 'null'], description: 'Cylinder, e.g. -0.50. null if blank.' },
    axis: { type: ['number', 'null'], description: 'Axis in degrees, 0-180. null if blank.' },
    va: { type: ['string', 'null'], description: 'Visual acuity as written, e.g. "6/6". null if blank.' },
  },
  required: ['sph', 'cyl', 'axis', 'va'],
};

const eyeSchema = {
  type: 'object',
  properties: {
    distance_vision: eyeReadingSchema,
    near_vision: eyeReadingSchema,
  },
  required: ['distance_vision', 'near_vision'],
};

const opticalPrescriptionSchema = {
  type: 'object',
  properties: {
    clinic_name: { type: ['string', 'null'] },
    prescription_date: { type: ['string', 'null'], description: 'ISO 8601 date exactly as printed. null if absent.' },
    patient_name: { type: ['string', 'null'] },
    age: { type: ['string', 'null'], description: 'Age exactly as printed. null if absent.' },
    right_eye: eyeSchema,
    left_eye: eyeSchema,
    wear_type: { type: ['string', 'null'], description: 'e.g. "constant", "reading only"' },
    lens_design: { type: ['string', 'null'], description: 'e.g. "single vision", "bifocal", "progressive"' },
    lens_materials: { type: 'array', items: { type: 'string' } },
    prescriber_name: { type: ['string', 'null'] },
    overall_confidence: { type: 'string', enum: CONFIDENCE_ENUM },
  },
  required: ['right_eye', 'left_eye', 'overall_confidence'],
};

/** Extraction schema keyed by document type. `unclear` has no schema — it never reaches extract(). */
const EXTRACTION_SCHEMAS: Partial<Record<DocumentType, Record<string, unknown>>> = {
  lab_result: labResultSchema,
  prescription: prescriptionSchema,
  diagnosis: diagnosisSchema,
  imaging_report: imagingReportSchema,
  vaccination_card: vaccinationCardSchema,
  optical_prescription: opticalPrescriptionSchema,
};

/** Verbatim system instruction for `explain()` — never let either provider stray from this. */
export const EXPLAIN_SYSTEM_INSTRUCTION =
  'You are explaining a medical document to a patient in plain language. Describe what the ' +
  'values or contents generally mean. You must NEVER provide a diagnosis, tell the patient ' +
  'whether a result is dangerous, or recommend a treatment. If a value looks abnormal, say ' +
  'only that it falls outside the typical reference range and that they should discuss it ' +
  'with their doctor. Always end with a reminder to consult their healthcare provider. Keep ' +
  'it under 150 words and avoid medical jargon.';

// ============================================================================
// Provider abstraction
// ============================================================================

export interface ClassifyResult {
  document_type: DocumentType;
  readable: boolean;
  reason_if_unreadable?: string;
}

export interface DocPipelineProvider {
  classify(fileBase64: string, mimeType: string): Promise<ClassifyResult>;
  extract(fileBase64: string, mimeType: string, documentType: string): Promise<Record<string, unknown>>;
  explain(fileBase64: string, mimeType: string, extractedData: Record<string, unknown>): Promise<string>;
}

export function schemaFor(documentType: string): Record<string, unknown> {
  const schema = EXTRACTION_SCHEMAS[documentType as DocumentType];
  if (!schema) {
    throw new Error(`No extraction schema registered for document type "${documentType}".`);
  }
  return schema;
}

/**
 * Pulls the first JSON object out of a model response.
 *
 * Exported for the self-check. Models that honour `response_format` return
 * bare JSON, but weaker ones (the free tier especially) wrap it in a ```json
 * fence or bracket it with prose, and a hard JSON.parse would throw on both.
 */
export function parseJsonObject(text: string): Record<string, unknown> {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const body = (fenced ? fenced[1] : text).trim();

  // Slice to the outermost braces so leading/trailing commentary is dropped.
  const start = body.indexOf('{');
  const end = body.lastIndexOf('}');
  if (start === -1 || end === -1 || end < start) {
    throw new Error(`Model did not return a JSON object. Got: ${text.slice(0, 200)}`);
  }

  // The slice always starts with { and ends with }, so JSON.parse either
  // throws or yields an object — no further shape check is reachable. A model
  // that wrapped its answer in an array lands here as the inner object, which
  // is the recovery we want.
  return JSON.parse(body.slice(start, end + 1)) as Record<string, unknown>;
}

/**
 * Restates the JSON Schema in the prompt.
 *
 * `response_format` is only a hint here — OpenRouter routes to whichever
 * provider is cheapest/available, and some reject it outright. Putting the
 * schema in the text means the model still knows the exact shape when the hint
 * had to be dropped.
 */
export function schemaPrompt(schema: Record<string, unknown>): string {
  return (
    'Reply with a single JSON object and nothing else - no prose, no markdown ' +
    `fences. It must match this JSON Schema:\n\n${JSON.stringify(schema)}`
  );
}
