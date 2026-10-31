// Decode server output with the dashboard's real runtime decoders.
//
// Input (stdin): {"messages": [<raw WebSocket text>...], "rest": [[<decoder name>, <body>]...]}
// Output (stdout): one result per item, {"ok": true} or {"ok": false, "detail": "..."}.
// Run by tests/api/test_dashboard_decoder.py; needs Node >= 22.18 (native TypeScript).
import { readFileSync } from 'node:fs';

import { parseServerMessage, restDecoders } from '../../dashboard/src/lib/protocol.ts';

const input = JSON.parse(readFileSync(0, 'utf8'));
const results = [];
for (const raw of input.messages) {
  const parsed = parseServerMessage(raw);
  results.push(
    parsed.ok
      ? { ok: true, type: parsed.message.type }
      : { ok: parsed.reason === 'unknown_type', detail: `${parsed.reason}: ${parsed.detail}` },
  );
}
for (const [name, body] of input.rest) {
  try {
    restDecoders[name](body);
    results.push({ ok: true, rest: name });
  } catch (error) {
    results.push({ ok: false, rest: name, detail: String(error) });
  }
}
process.stdout.write(JSON.stringify(results));
