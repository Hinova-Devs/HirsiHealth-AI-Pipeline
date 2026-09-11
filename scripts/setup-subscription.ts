/**
 * Creates (or updates) the Medplum Subscription that fires the extraction
 * webhook. Without this the pipeline is never called — nothing in Medplum
 * knows the endpoint exists.
 *
 * Run it once per Medplum project, and again whenever WEBHOOK_URL or
 * WEBHOOK_SHARED_SECRET changes:
 *
 *   node --env-file=.env --experimental-strip-types scripts/setup-subscription.ts
 *
 * Re-running is safe: the Subscription is matched by its endpoint and updated
 * in place rather than duplicated.
 */

import { MedplumClient } from '@medplum/core';
import type { Subscription } from '@medplum/fhirtypes';

/**
 * Medplum extension limiting the Subscription to `create`.
 *
 * This is not a nicety — it prevents an infinite loop. `handler()` PATCHes
 * `DocumentReference.docStatus` when it finishes, and an update-triggered
 * Subscription would fire on that PATCH, re-run the whole pipeline, PATCH
 * again, and never stop.
 */
const SUPPORTED_INTERACTION =
  'https://medplum.com/fhir/StructureDefinition/subscription-supported-interaction';

function required(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`${name} is not set. Copy .env.example to .env and fill it in.`);
  }
  return value;
}

async function main(): Promise<void> {
  const baseUrl = process.env.MEDPLUM_BASE_URL;
  const clientId = required('MEDPLUM_CLIENT_ID');
  const clientSecret = required('MEDPLUM_CLIENT_SECRET');
  const endpoint = required('WEBHOOK_URL');
  const secret = required('WEBHOOK_SHARED_SECRET');

  const medplum = new MedplumClient({ baseUrl });
  await medplum.startClientLogin(clientId, clientSecret);

  const desired: Subscription = {
    resourceType: 'Subscription',
    status: 'active',
    reason: 'HersiHealth AI document extraction',
    criteria: 'DocumentReference',
    channel: {
      type: 'rest-hook',
      endpoint,
      // The handler parses the DocumentReference straight out of the body.
      payload: 'application/fhir+json',
      // Matches the constant-time check in isAuthorized().
      header: [`Authorization: Bearer ${secret}`],
    },
    extension: [{ url: SUPPORTED_INTERACTION, valueCode: 'create' }],
  };

  const existing = await medplum.searchOne('Subscription', { url: endpoint });

  if (existing) {
    const updated = await medplum.updateResource({ ...desired, id: existing.id });
    console.log(`Updated Subscription/${updated.id} → ${endpoint}`);
  } else {
    const created = await medplum.createResource(desired);
    console.log(`Created Subscription/${created.id} → ${endpoint}`);
  }

  console.log('Fires on: DocumentReference create (not update — see SUPPORTED_INTERACTION).');
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
