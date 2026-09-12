/**
 * AI document-extraction webhook.
 *
 * Medplum posts a `DocumentReference` here (via a Subscription with
 * `channel.payload = application/fhir+json`) whenever a patient uploads a
 * document. This handler runs a three-stage pipeline — classify, extract,
 * explain — against the underlying file, then writes a draft `Task` back to
 * Medplum so the patient can review the AI's reading before anything
 * clinical is trusted.
 *
 * See the numbered STEP comments in `handler()` for the request lifecycle.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { timingSafeEqual } from 'node:crypto';
import { MedplumClient } from '@medplum/core';
import type { DocumentReference, Task, TaskOutput } from '@medplum/fhirtypes';
import { GoogleGenAI } from '@google/genai';
import Anthropic from '@anthropic-ai/sdk';

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
  | 'unclear';

const CONFIDENCE_ENUM = ['high', 'medium', 'low'];

/** Schema for `provider.classify()`. Shared by both providers. */
const classifySchema = {
  type: 'object',
  properties: {
    document_type: {
      type: 'string',
      enum: ['lab_result', 'prescription', 'diagnosis', 'imaging_report', 'vaccination_card', 'unclear'],
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

/** Extraction schema keyed by document type. `unclear` has no schema — it never reaches extract(). */
const EXTRACTION_SCHEMAS: Partial<Record<DocumentType, Record<string, unknown>>> = {
  lab_result: labResultSchema,
  prescription: prescriptionSchema,
  diagnosis: diagnosisSchema,
  imaging_report: imagingReportSchema,
  vaccination_card: vaccinationCardSchema,
};

/** Verbatim system instruction for `explain()` — never let either provider stray from this. */
const EXPLAIN_SYSTEM_INSTRUCTION =
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

function schemaFor(documentType: string): Record<string, unknown> {
  const schema = EXTRACTION_SCHEMAS[documentType as DocumentType];
  if (!schema) {
    throw new Error(`No extraction schema registered for document type "${documentType}".`);
  }
  return schema;
}

// ── Gemini ───────────────────────────────────────────────────────────────

class GeminiProvider implements DocPipelineProvider {
  private readonly ai: GoogleGenAI;

  constructor(apiKey: string) {
    this.ai = new GoogleGenAI({ apiKey });
  }

  private filePart(fileBase64: string, mimeType: string) {
    return { inlineData: { mimeType, data: fileBase64 } };
  }

  /** `response.text` is a getter (not a call) in @google/genai, and can be `undefined`. */
  private textOf(response: { text?: string }): string {
    if (!response.text) {
      throw new Error('Gemini returned an empty response.');
    }
    return response.text;
  }

  async classify(fileBase64: string, mimeType: string): Promise<ClassifyResult> {
    const response = await this.ai.models.generateContent({
      model: 'gemini-3.5-flash-lite',
      contents: [
        this.filePart(fileBase64, mimeType),
        {
          text:
            'Classify this medical document. Determine which category it belongs to and whether ' +
            'it is clear and complete enough to extract data from.',
        },
      ],
      config: {
        responseMimeType: 'application/json',
        responseSchema: classifySchema,
      },
    });
    return JSON.parse(this.textOf(response)) as ClassifyResult;
  }

  async extract(fileBase64: string, mimeType: string, documentType: string): Promise<Record<string, unknown>> {
    const schema = schemaFor(documentType);
    const response = await this.ai.models.generateContent({
      model: 'gemini-3.5-flash',
      contents: [
        this.filePart(fileBase64, mimeType),
        { text: `Extract the structured data from this ${documentType.replace(/_/g, ' ')} document.` },
      ],
      config: {
        responseMimeType: 'application/json',
        responseSchema: schema,
      },
    });
    return JSON.parse(this.textOf(response)) as Record<string, unknown>;
  }

  async explain(fileBase64: string, mimeType: string, extractedData: Record<string, unknown>): Promise<string> {
    const response = await this.ai.models.generateContent({
      model: 'gemini-3.5-flash',
      contents: [
        this.filePart(fileBase64, mimeType),
        {
          text:
            `Here is the structured data extracted from this document:\n\n${JSON.stringify(extractedData, null, 2)}\n\n` +
            'Explain this to the patient.',
        },
      ],
      config: {
        systemInstruction: EXPLAIN_SYSTEM_INSTRUCTION,
      },
    });
    return this.textOf(response);
  }
}

// ── Claude ───────────────────────────────────────────────────────────────

class ClaudeProvider implements DocPipelineProvider {
  private readonly client: Anthropic;

  constructor(apiKey: string) {
    this.client = new Anthropic({ apiKey });
  }

  /** `document` blocks for PDFs, `image` blocks for everything else. */
  private fileBlock(fileBase64: string, mimeType: string): Anthropic.DocumentBlockParam | Anthropic.ImageBlockParam {
    if (mimeType === 'application/pdf') {
      return {
        type: 'document',
        source: { type: 'base64', media_type: 'application/pdf', data: fileBase64 },
      };
    }
    return {
      type: 'image',
      source: {
        type: 'base64',
        media_type: mimeType as Anthropic.Base64ImageSource['media_type'],
        data: fileBase64,
      },
    };
  }

  /** Pulls the `input` of a forced tool call out of a Messages API response. */
  private toolInput(message: Anthropic.Message, toolName: string): Record<string, unknown> {
    const block = message.content.find(
      (b): b is Anthropic.ToolUseBlock => b.type === 'tool_use' && b.name === toolName,
    );
    if (!block) {
      throw new Error(`Claude did not return a "${toolName}" tool call.`);
    }
    return block.input as Record<string, unknown>;
  }

  async classify(fileBase64: string, mimeType: string): Promise<ClassifyResult> {
    const message = await this.client.messages.create({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 1024,
      tool_choice: { type: 'tool', name: 'classify_document' },
      tools: [
        {
          name: 'classify_document',
          description: 'Classify a medical document (image or PDF) by type and readability.',
          input_schema: classifySchema as unknown as Anthropic.Tool.InputSchema,
        },
      ],
      messages: [
        {
          role: 'user',
          content: [
            this.fileBlock(fileBase64, mimeType),
            {
              type: 'text',
              text:
                'Classify this medical document. Determine which category it belongs to and ' +
                'whether it is clear and complete enough to extract data from.',
            },
          ],
        },
      ],
    });
    return this.toolInput(message, 'classify_document') as unknown as ClassifyResult;
  }

  async extract(fileBase64: string, mimeType: string, documentType: string): Promise<Record<string, unknown>> {
    const schema = schemaFor(documentType);
    const toolName = `extract_${documentType}`;
    const readableType = documentType.replace(/_/g, ' ');
    const message = await this.client.messages.create({
      model: 'claude-sonnet-5',
      max_tokens: 4096,
      tool_choice: { type: 'tool', name: toolName },
      tools: [
        {
          name: toolName,
          description: `Extract structured data from a ${readableType} document.`,
          input_schema: schema as unknown as Anthropic.Tool.InputSchema,
        },
      ],
      messages: [
        {
          role: 'user',
          content: [
            this.fileBlock(fileBase64, mimeType),
            { type: 'text', text: `Extract the structured data from this ${readableType} document.` },
          ],
        },
      ],
    });
    return this.toolInput(message, toolName);
  }

  async explain(fileBase64: string, mimeType: string, extractedData: Record<string, unknown>): Promise<string> {
    const message = await this.client.messages.create({
      model: 'claude-sonnet-5',
      max_tokens: 512,
      system: EXPLAIN_SYSTEM_INSTRUCTION,
      messages: [
        {
          role: 'user',
          content: [
            this.fileBlock(fileBase64, mimeType),
            {
              type: 'text',
              text:
                `Here is the structured data extracted from this document:\n\n${JSON.stringify(extractedData, null, 2)}\n\n` +
                'Explain this to the patient.',
            },
          ],
        },
      ],
    });
    const textBlock = message.content.find((b): b is Anthropic.TextBlock => b.type === 'text');
    if (!textBlock) {
      throw new Error('Claude did not return a text explanation.');
    }
    return textBlock.text;
  }
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
function schemaPrompt(schema: Record<string, unknown>): string {
  return (
    'Reply with a single JSON object and nothing else - no prose, no markdown ' +
    `fences. It must match this JSON Schema:\n\n${JSON.stringify(schema)}`
  );
}

/**
 * OpenRouter, via its OpenAI-compatible chat-completions endpoint.
 *
 * Uses plain `fetch` rather than the OpenAI SDK — three calls do not justify
 * another dependency. The model is not hardcoded: pick one that accepts images
 * at https://openrouter.ai/models?modality=text+image-%3Etext and set
 * OPENROUTER_MODEL, because model ids get retired (which is how we got here).
 */
class OpenRouterProvider implements DocPipelineProvider {
  private readonly apiKey: string;
  private readonly model: string;

  constructor(apiKey: string, model: string) {
    this.apiKey = apiKey;
    this.model = model;
  }

  /** Images go in as data URIs; PDFs use OpenRouter's file-parser plugin. */
  private filePart(fileBase64: string, mimeType: string): Record<string, unknown> {
    if (mimeType === 'application/pdf') {
      return {
        type: 'file',
        file: { filename: 'document.pdf', file_data: `data:application/pdf;base64,${fileBase64}` },
      };
    }
    return { type: 'image_url', image_url: { url: `data:${mimeType};base64,${fileBase64}` } };
  }

  private post(body: Record<string, unknown>): Promise<Response> {
    return fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        // Optional attribution headers OpenRouter uses for its dashboards.
        'HTTP-Referer': 'https://hersihealth.so',
        'X-Title': 'HersiHealth',
      },
      body: JSON.stringify(body),
    });
  }

  private async chat(params: {
    fileBase64: string;
    mimeType: string;
    text: string;
    system?: string;
    schema?: { name: string; schema: Record<string, unknown> };
    maxTokens: number;
  }): Promise<string> {
    const body: Record<string, unknown> = {
      model: this.model,
      max_tokens: params.maxTokens,
      messages: [
        ...(params.system ? [{ role: 'system', content: params.system }] : []),
        {
          role: 'user',
          content: [
            this.filePart(params.fileBase64, params.mimeType),
            { type: 'text', text: params.text },
          ],
        },
      ],
    };

    if (params.schema) {
      // strict:false — the pipeline's schemas do not carry the
      // additionalProperties:false / fully-required shape strict mode demands,
      // and parseJsonObject() covers models that ignore the field entirely.
      body.response_format = {
        type: 'json_schema',
        json_schema: { name: params.schema.name, strict: false, schema: params.schema.schema },
      };
    }

    if (params.mimeType === 'application/pdf') {
      body.plugins = [{ id: 'file-parser', pdf: { engine: 'pdf-text' } }];
    }

    let response = await this.post(body);

    // OpenRouter picks a provider per request, and not all of them accept
    // response_format (Novita, for one, answers 400 "structured outputs not
    // support"). Rather than pinning routing to the few that do, drop the hint
    // and retry: the schema is in the prompt and parseJsonObject() copes with
    // whatever wrapping comes back.
    if (!response.ok && body.response_format) {
      const detail = await response.text();
      if (!/structured output|response_format|json_schema/i.test(detail)) {
        throw new Error(`OpenRouter ${response.status}: ${detail.slice(0, 400)}`);
      }
      delete body.response_format;
      response = await this.post(body);
    }

    if (!response.ok) {
      throw new Error(`OpenRouter ${response.status}: ${(await response.text()).slice(0, 400)}`);
    }

    const json = (await response.json()) as {
      choices?: { message?: { content?: string } }[];
      error?: { message?: string };
    };

    // A 200 can still carry a provider-level error (rate limit, no capacity).
    if (json.error) {
      throw new Error(`OpenRouter: ${json.error.message ?? 'unknown error'}`);
    }

    const content = json.choices?.[0]?.message?.content;
    if (typeof content !== 'string' || !content.trim()) {
      throw new Error('OpenRouter returned an empty response.');
    }
    return content;
  }

  async classify(fileBase64: string, mimeType: string): Promise<ClassifyResult> {
    const text = await this.chat({
      fileBase64,
      mimeType,
      maxTokens: 1024,
      schema: { name: 'classify_document', schema: classifySchema },
      text:
        'Classify this medical document. Determine which category it belongs to and whether ' +
        'it is clear and complete enough to extract data from.' +
        '\n\n' +
        schemaPrompt(classifySchema),
    });
    return parseJsonObject(text) as unknown as ClassifyResult;
  }

  async extract(
    fileBase64: string,
    mimeType: string,
    documentType: string,
  ): Promise<Record<string, unknown>> {
    const schema = schemaFor(documentType);
    const text = await this.chat({
      fileBase64,
      mimeType,
      maxTokens: 4096,
      schema: { name: `extract_${documentType}`, schema },
      text:
        `Extract the structured data from this ${documentType.replace(/_/g, ' ')} document.` +
        '\n\n' +
        schemaPrompt(schema),
    });
    return parseJsonObject(text);
  }

  async explain(
    fileBase64: string,
    mimeType: string,
    extractedData: Record<string, unknown>,
  ): Promise<string> {
    return this.chat({
      fileBase64,
      mimeType,
      maxTokens: 512,
      system: EXPLAIN_SYSTEM_INSTRUCTION,
      text:
        `Here is the structured data extracted from this document:\n\n` +
        `${JSON.stringify(extractedData, null, 2)}\n\nExplain this to the patient.`,
    });
  }
}

/**
 * Picks the provider from `AI_PROVIDER` (default "gemini"). Called once, at
 * the top of the handler — NOT at module scope — so a missing
 * `ANTHROPIC_API_KEY` only throws if "claude" is actually selected at
 * runtime, never merely on import.
 */
function getProvider(): DocPipelineProvider {
  const selected = (process.env.AI_PROVIDER || 'gemini').toLowerCase();

  if (selected === 'gemini') {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error('AI_PROVIDER is "gemini" but GEMINI_API_KEY is not set.');
    }
    return new GeminiProvider(apiKey);
  }

  if (selected === 'claude') {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      throw new Error('AI_PROVIDER is "claude" but ANTHROPIC_API_KEY is not set.');
    }
    return new ClaudeProvider(apiKey);
  }

  if (selected === 'openrouter') {
    const apiKey = process.env.OPENROUTER_API_KEY;
    if (!apiKey) {
      throw new Error('AI_PROVIDER is "openrouter" but OPENROUTER_API_KEY is not set.');
    }
    const model = process.env.OPENROUTER_MODEL;
    if (!model) {
      throw new Error(
        'AI_PROVIDER is "openrouter" but OPENROUTER_MODEL is not set. Pick a model that ' +
          'accepts images from https://openrouter.ai/models?modality=text+image-%3Etext',
      );
    }
    return new OpenRouterProvider(apiKey, model);
  }

  throw new Error(
    `Unknown AI_PROVIDER "${selected}". Expected "gemini", "claude" or "openrouter".`,
  );
}

// ============================================================================
// Small helpers
// ============================================================================

const MAX_FILE_BYTES = 20 * 1024 * 1024; // 20MB

/**
 * Constant-time check that `Authorization` is exactly `Bearer
 * <WEBHOOK_SHARED_SECRET>`. Avoids leaking the secret's length/prefix via
 * response-time timing.
 */
function isAuthorized(authHeader: string | undefined): boolean {
  const secret = process.env.WEBHOOK_SHARED_SECRET;
  if (!secret || !authHeader) {
    return false;
  }
  const expected = Buffer.from(`Bearer ${secret}`);
  const actual = Buffer.from(authHeader);
  if (expected.length !== actual.length) {
    return false;
  }
  // Cast: Buffer satisfies NodeJS.ArrayBufferView at runtime; the mismatch
  // here is a TS lib/@types/node typed-array generic quirk, not a real one.
  return timingSafeEqual(expected as unknown as NodeJS.ArrayBufferView, actual as unknown as NodeJS.ArrayBufferView);
}

function taskOutput(text: string, valueString: string): TaskOutput {
  return { type: { text }, valueString };
}

/** Builds and creates the review/failure Task described in STEP 7 / STEP 8. */
async function createReviewTask(
  medplum: MedplumClient,
  params: {
    documentReferenceId: string;
    subjectReference: string;
    status: Task['status'];
    codeText: string;
    output?: TaskOutput[];
  },
): Promise<void> {
  const task: Task = {
    resourceType: 'Task',
    status: params.status,
    intent: 'order',
    code: { text: params.codeText },
    focus: { reference: `DocumentReference/${params.documentReferenceId}` },
    for: { reference: params.subjectReference },
    ...(params.output ? { output: params.output } : {}),
  };
  await medplum.createResource(task);
}

/** PATCHes `DocumentReference.docStatus` (STEP 7 / STEP 8). Uses `add`, not `replace`, since docStatus may not already be set. */
async function patchDocStatus(
  medplum: MedplumClient,
  documentReferenceId: string,
  docStatus: DocumentReference['docStatus'],
): Promise<void> {
  await medplum.patchResource('DocumentReference', documentReferenceId, [
    { op: 'add', path: '/docStatus', value: docStatus },
  ]);
}
async function readRawBody(req: VercelRequest): Promise<string> {
  // Uint8Array[], not Buffer[] — Buffer.concat's parameter is typed against
  // Uint8Array<ArrayBuffer> and Buffer's ArrayBufferLike does not narrow to it.
  const chunks: Uint8Array[] = [];
  for await (const chunk of req) {
    chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}
// ============================================================================
// Handler
// ============================================================================

export default async function handler(req: VercelRequest, res: VercelResponse): Promise<void> {
  // ── STEP 1: verify the request, before touching the body ──────────────────
  const authHeader = req.headers.authorization;
  if (!isAuthorized(typeof authHeader === 'string' ? authHeader : undefined)) {
    res.status(401).json({ error: 'Unauthorized' });
    return;
  }

  // ── STEP 2: parse the payload ───────────────────────────────────────────
 let payload: unknown = req.body;
if (payload === undefined || payload === null) {
  try {
    const raw = await readRawBody(req);
    payload = raw ? JSON.parse(raw) : undefined;
  } catch {
    payload = undefined;
  }
} else if (typeof payload === 'string') {
  try { payload = JSON.parse(payload); } catch { payload = undefined; }
} else if (Buffer.isBuffer(payload)) {
  try { payload = JSON.parse(payload.toString('utf8')); } catch { payload = undefined; }
}
console.log('[extract] payload type:', typeof req.body, 'resourceType:', (payload as any)?.resourceType);
  if (
    !payload ||
    typeof payload !== 'object' ||
    (payload as { resourceType?: unknown }).resourceType !== 'DocumentReference'
  ) {
    // Not our resource — no-op with 200 so Medplum doesn't retry this forever.
    res.status(200).json({ ok: true, skipped: true, reason: 'Not a DocumentReference' });
    return;
  }

  const docRef = payload as DocumentReference;
  const documentReferenceId = docRef.id;
  const subjectReference = docRef.subject?.reference;
  const attachment = docRef.content?.[0]?.attachment;
  const binaryUrl = attachment?.url;
  const contentType = attachment?.contentType ?? 'application/octet-stream';

  if (!documentReferenceId || !subjectReference || !binaryUrl) {
    // Malformed DocumentReference — same reasoning as above: ack, don't retry.
    console.error('[extract] DocumentReference missing id/subject/attachment.url', {
      documentReferenceId,
      subjectReference,
      binaryUrl,
    });
    res.status(200).json({ ok: true, skipped: true, reason: 'Missing id, subject, or attachment.url' });
    return;
  }

  // ── STEP 3: authenticate to Medplum (client-credentials) ──────────────────
  const medplum = new MedplumClient({ baseUrl: process.env.MEDPLUM_BASE_URL });
  await medplum.startClientLogin(
    process.env.MEDPLUM_CLIENT_ID as string,
    process.env.MEDPLUM_CLIENT_SECRET as string,
  );

  // ── STEP 4: download the file ───────────────────────────────────────────
  const blob = await medplum.download(binaryUrl);
  const arrayBuffer = await blob.arrayBuffer();
  const fileBytes = Buffer.from(arrayBuffer);

  if (fileBytes.byteLength > MAX_FILE_BYTES) {
    await createReviewTask(medplum, {
      documentReferenceId,
      subjectReference,
      status: 'failed',
      codeText: 'File too large to process',
    });
    res.status(200).json({ ok: true, tooLarge: true });
    return;
  }

  const fileBase64 = fileBytes.toString('base64');

  // ── STEPS 5–7, wrapped so any failure becomes a failed Task, not a 500 ────
  try {
    const provider = getProvider();

    // STEP 5: classify + quality gate.
    const classification = await provider.classify(fileBase64, contentType);

    if (!classification.readable || classification.document_type === 'unclear') {
      await createReviewTask(medplum, {
        documentReferenceId,
        subjectReference,
        status: 'ready',
        codeText: 'Retake photo',
        output: [
          taskOutput('documentType', classification.document_type),
          ...(classification.reason_if_unreadable
            ? [taskOutput('reason', classification.reason_if_unreadable)]
            : []),
        ],
      });
      res.status(200).json({ ok: true, unreadable: true });
      return;
    }

    // STEP 6: extract + explain (only ever runs on readable, classified input).
    const extractedData = await provider.extract(fileBase64, contentType, classification.document_type);
    const explanation = await provider.explain(fileBase64, contentType, extractedData);
    const overallConfidence =
      typeof extractedData.overall_confidence === 'string' ? extractedData.overall_confidence : 'low';

    // STEP 7: write the draft Task, then mark the DocumentReference preliminary.
    await createReviewTask(medplum, {
      documentReferenceId,
      subjectReference,
      status: 'ready',
      codeText: 'Review AI-extracted document data',
      output: [
        taskOutput('documentType', classification.document_type),
        taskOutput('extractedData', JSON.stringify(extractedData)),
        taskOutput('explanation', explanation),
        taskOutput('confidence', overallConfidence),
      ],
    });
    await patchDocStatus(medplum, documentReferenceId, 'preliminary');

    res.status(200).json({ ok: true });
  } catch (err) {
    // ── STEP 8: never let a real failure become a 500 (Medplum would retry
    //    the whole webhook and double-process an already-failed document) ────
    const message = err instanceof Error ? err.message : String(err);
    console.error('[extract] Pipeline failed for DocumentReference/' + documentReferenceId, err);

    try {
      await createReviewTask(medplum, {
        documentReferenceId,
        subjectReference,
        status: 'failed',
        codeText: `Extraction failed: ${message}`,
      });
      await patchDocStatus(medplum, documentReferenceId, 'entered-in-error');
    } catch (cleanupErr) {
      // Best-effort: we already logged the root cause above; a failure here
      // just means the Task/docStatus bookkeeping didn't land.
      console.error('[extract] Failed to record the failure Task/docStatus', cleanupErr);
    }

    res.status(200).json({ ok: true, failed: true });
  }
}
