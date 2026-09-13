/**
 * Z.ai GLM vision provider.
 *
 * Z.ai speaks the OpenAI chat-completions shape, so this is the `openai`
 * client pointed at `https://api.z.ai/api/paas/v4/`.
 *
 * Two things are deliberate:
 *
 *  - `maxRetries: 0`. The SDK's own retry would multiply against
 *    `withRetry()` below (2 x 3 = 6 calls per stage, 18 per document) and
 *    blow the function's 60s budget. The backoff here is the only one.
 *  - No `response_format`. The free vision tier does not reliably honour it,
 *    so the schema goes in the prompt and `parseJsonObject()` recovers the
 *    object from fences or surrounding prose — the same treatment the
 *    OpenRouter provider already needs.
 */

import OpenAI from 'openai';
import {
  classifySchema,
  parseJsonObject,
  schemaFor,
  schemaPrompt,
  EXPLAIN_SYSTEM_INSTRUCTION,
  type ClassifyResult,
  type DocPipelineProvider,
  type DocumentType,
} from './schemas.ts';

export const ZAI_BASE_URL = 'https://api.z.ai/api/paas/v4/';

/**
 * Free tier as of 2026-09. Re-check https://z.ai pricing before shipping —
 * "free" is a tier decision Z.ai can change, and this pipeline sends every
 * patient upload through it.
 */
export const DEFAULT_ZAI_MODEL = 'glm-4.6v-flash';

/** Attempt 1 is immediate; these are the waits before attempts 2 and 3. */
const RETRY_DELAYS_MS = [1_000, 3_000];

const DOCUMENT_TYPES: readonly string[] = (classifySchema.properties as {
  document_type: { enum: string[] };
}).document_type.enum;

/**
 * True for failures that a later attempt can plausibly survive.
 *
 * The one this exists for: Z.ai's shared free tier answers `429` with body
 * code `1305` ("temporarily overloaded") several times a day. That is
 * capacity, not credentials and not balance — a `401`/`403` must fall
 * straight through so a misconfigured key fails fast instead of after 4s.
 */
export function isTransientZaiError(err: unknown): boolean {
  const status = (err as { status?: unknown })?.status;
  if (status === 429 || (typeof status === 'number' && status >= 500)) {
    return true;
  }
  // Some Z.ai overload responses arrive as a 200 body or an undecorated
  // fetch failure; the 1305 code is the reliable marker either way.
  const message = err instanceof Error ? err.message : String(err ?? '');
  return /\b1305\b/.test(message) || /fetch failed|ECONNRESET|ETIMEDOUT/i.test(message);
}

async function withRetry<T>(label: string, fn: () => Promise<T>): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (attempt >= RETRY_DELAYS_MS.length || !isTransientZaiError(err)) {
        throw err;
      }
      const wait = RETRY_DELAYS_MS[attempt];
      console.warn(
        `[glm] ${label} attempt ${attempt + 1}/${RETRY_DELAYS_MS.length + 1} failed ` +
          `(${err instanceof Error ? err.message : String(err)}); retrying in ${wait}ms`,
      );
      await new Promise((resolve) => setTimeout(resolve, wait));
    }
  }
}

/**
 * Narrows a parsed classification. Anything the model invents — a type not in
 * the enum, a missing `readable` — degrades to "unclear", which routes the
 * document to manual categorisation instead of forcing a wrong schema onto it.
 */
export function coerceClassification(parsed: Record<string, unknown>): ClassifyResult {
  const type = parsed.document_type;
  const documentType: DocumentType =
    typeof type === 'string' && DOCUMENT_TYPES.includes(type) ? (type as DocumentType) : 'unclear';
  return {
    document_type: documentType,
    // Only an explicit `false` means unreadable; a missing field is not a
    // licence to reject the document.
    readable: parsed.readable !== false,
    ...(typeof parsed.reason_if_unreadable === 'string'
      ? { reason_if_unreadable: parsed.reason_if_unreadable }
      : {}),
  };
}

export class GlmProvider implements DocPipelineProvider {
  private readonly client: OpenAI;
  private readonly model: string;
  /** Raw text of every call this instance made, newest last. Audit only — see `api/extract.ts`. */
  readonly transcript: { stage: string; response: string }[] = [];

  constructor(apiKey: string, model: string = DEFAULT_ZAI_MODEL) {
    this.client = new OpenAI({ apiKey, baseURL: ZAI_BASE_URL, maxRetries: 0 });
    this.model = model;
  }

  private async chat(params: {
    stage: string;
    fileBase64: string;
    mimeType: string;
    text: string;
    system?: string;
    maxTokens: number;
  }): Promise<string> {
    const completion = await withRetry(params.stage, () =>
      this.client.chat.completions.create({
        model: this.model,
        max_tokens: params.maxTokens,
        // Extraction is transcription, not composition.
        temperature: 0,
        messages: [
          ...(params.system ? [{ role: 'system' as const, content: params.system }] : []),
          {
            role: 'user' as const,
            content: [
              {
                type: 'image_url' as const,
                image_url: { url: `data:${params.mimeType};base64,${params.fileBase64}` },
              },
              { type: 'text' as const, text: params.text },
            ],
          },
        ],
      }),
    );

    const content = completion.choices?.[0]?.message?.content;
    if (typeof content !== 'string' || !content.trim()) {
      throw new Error(`GLM returned an empty response for ${params.stage}.`);
    }
    this.transcript.push({ stage: params.stage, response: content });
    return content;
  }

  async classify(fileBase64: string, mimeType: string): Promise<ClassifyResult> {
    // GLM's vision endpoint takes images only. A PDF is not a failure — it is
    // a document we cannot read, which is exactly the manual-categorisation
    // path, so say so here rather than throwing three stages later.
    if (!mimeType.startsWith('image/')) {
      return {
        document_type: 'unclear',
        readable: true,
        reason_if_unreadable:
          'This file is not a photo, and the current reader only handles photos. ' +
          'Please categorise it and enter any details yourself.',
      };
    }

    // Stage 1 is deliberately cheap: no field extraction, no schema restated
    // beyond the classification one, small token ceiling.
    const text = await this.chat({
      stage: 'classify',
      fileBase64,
      mimeType,
      maxTokens: 300,
      text:
        'Classify this medical document. Decide which category it belongs to, and whether the ' +
        'photo is clear and complete enough to read data off. Do not extract any values yet.' +
        '\n\n' +
        schemaPrompt(classifySchema),
    });
    return coerceClassification(parseJsonObject(text));
  }

  async extract(fileBase64: string, mimeType: string, documentType: string): Promise<Record<string, unknown>> {
    const schema = schemaFor(documentType);
    const readableType = documentType.replace(/_/g, ' ');
    const text = await this.chat({
      stage: `extract:${documentType}`,
      fileBase64,
      mimeType,
      maxTokens: 4096,
      text:
        `Extract the structured data from this ${readableType} document.\n` +
        'Transcribe only what is printed or written on the page. A blank cell is null — never ' +
        'fill it from a neighbouring row, column, or the other eye. Copy dates digit by digit.' +
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
      stage: 'explain',
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
