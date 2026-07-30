// Tests for the bulk-import CSV parser. Run with `npm test` (node:test, built
// into Node 22 — no jest/vitest dependency for what is a pure-function module).
//
// This is the one piece of frontend logic with real edge cases: a sales person
// uploads whatever their CRM exported, so quoting, BOMs, missing columns and
// bad emails all have to fail in a way that tells them which row to fix.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { parseCSV, toLeads, MAX_ROWS } from './csv.js'

const HEADER = 'Name,Company,Email'

test('parses a plain row', () => {
  const rows = parseCSV('a,b,c\n1,2,3')
  assert.deepEqual(rows, [['a', 'b', 'c'], ['1', '2', '3']])
})

test('handles quoted fields with commas, newlines and "" escapes', () => {
  const rows = parseCSV('a,"b,still b","line1\nline2","say ""hi"""')
  assert.deepEqual(rows, [['a', 'b,still b', 'line1\nline2', 'say "hi"']])
})

test('tolerates CRLF and a trailing newline', () => {
  assert.deepEqual(parseCSV('a,b\r\n1,2\r\n'), [['a', 'b'], ['1', '2']])
})

test('drops blank lines rather than emitting empty rows', () => {
  assert.deepEqual(parseCSV('a,b\n\n1,2\n   \n'), [['a', 'b'], ['1', '2']])
})

test('maps both spaced and underscored header spellings', () => {
  const { leads } = toLeads(
    'Name,Company,Email,Job Title,Use Case\nA,Co,a@co.com,VP Eng,Scaling outbound')
  assert.equal(leads[0].job_title, 'VP Eng')
  assert.equal(leads[0].use_case, 'Scaling outbound')
})

test('ignores unknown columns so an exported file round-trips', () => {
  const { leads, errors } = toLeads(`${HEADER},Score,Email Draft\nA,Co,a@co.com,87,hi`)
  assert.deepEqual(errors, [])
  assert.equal(leads.length, 1)
  assert.equal(leads[0].score, undefined)
})

test('strips the BOM the CSV export writes', () => {
  // \uFEFF as an escape, not a literal — an invisible BOM in source is both
  // unreadable and an eslint no-irregular-whitespace error.
  const { leads, errors } = toLeads(`\uFEFF${HEADER}\nA,Co,a@co.com`)
  assert.deepEqual(errors, [])
  assert.equal(leads.length, 1)
})

test('is case- and whitespace-insensitive about headers', () => {
  const { leads } = toLeads('  NAME , Company ,  EMAIL \nA,Co,a@co.com')
  assert.equal(leads.length, 1)
})

test('names the missing column instead of failing vaguely', () => {
  const { leads, errors } = toLeads('Name,Company\nA,Co')
  assert.equal(leads.length, 0)
  assert.match(errors[0], /Missing required column: "email"/)
})

test('reports an empty file', () => {
  assert.deepEqual(toLeads('').errors, ['That file is empty.'])
})

test('rejects a bad email and points at the right line', () => {
  const { leads, errors } = toLeads(`${HEADER}\nA,Co,not-an-email`)
  assert.equal(leads.length, 0)
  assert.match(errors[0], /^Row 2: "not-an-email" is not a valid email\.$/)
})

test('line numbers account for the header row', () => {
  const { errors } = toLeads(`${HEADER}\nA,Co,a@co.com\nB,Co,bad`)
  assert.match(errors[0], /^Row 3:/)
})

test('requires all three of name, company and email', () => {
  const { leads, errors } = toLeads(`${HEADER}\n,Co,a@co.com\nB,,b@co.com\nC,Co,`)
  assert.equal(leads.length, 0)
  assert.equal(errors.length, 3)
  for (const e of errors) assert.match(e, /name, company and email are all required/)
})

test('keeps good rows and reports only the bad ones', () => {
  const { leads, errors } = toLeads(`${HEADER}\nA,Co,a@co.com\nB,Co,bad\nC,Co,c@co.com`)
  assert.deepEqual(leads.map(l => l.name), ['A', 'C'])
  assert.equal(errors.length, 1)
})

test('defaults source to Website but keeps an explicit one', () => {
  const { leads } = toLeads(
    `${HEADER},Source\nA,Co,a@co.com,\nB,Co,b@co.com,Conference`)
  assert.equal(leads[0].source, 'Website')
  assert.equal(leads[1].source, 'Conference')
})

test('trims surrounding whitespace out of values', () => {
  const { leads } = toLeads(`${HEADER}\n  A  ,  Co  ,  a@co.com  `)
  assert.deepEqual(
    { name: leads[0].name, company: leads[0].company, email: leads[0].email },
    { name: 'A', company: 'Co', email: 'a@co.com' })
})

test(`accepts exactly ${MAX_ROWS} rows and rejects one more`, () => {
  const rows = n => `${HEADER}\n` +
    Array.from({ length: n }, (_, i) => `N${i},Co,u${i}@co.com`).join('\n')

  const ok = toLeads(rows(MAX_ROWS))
  assert.equal(ok.leads.length, MAX_ROWS)
  assert.deepEqual(ok.errors, [])

  const over = toLeads(rows(MAX_ROWS + 1))
  assert.match(over.errors[0], new RegExp(`Too many rows \\(${MAX_ROWS + 1}\\)`))
})

test('a quoted field containing a newline does not shift row numbering', () => {
  const { leads, errors } = toLeads(
    `${HEADER},Use Case\nA,Co,a@co.com,"multi\nline"\nB,Co,bad,x`)
  assert.equal(leads[0].use_case, 'multi\nline')
  assert.match(errors[0], /^Row 3:/)
})
