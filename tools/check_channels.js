const { DatabaseSync } = require('node:sqlite')
const db = new DatabaseSync('/koishi/data/koishi.db', { readOnly: true })
const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all().map((r) => r.name)
console.log('tables:', tables.join(', '))
try {
  const ch = db.prepare('SELECT * FROM channel').all()
  console.log(JSON.stringify(ch, null, 1))
} catch (e) {
  console.log('channel err', e.message)
}
