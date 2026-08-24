const { DatabaseSync } = require('node:sqlite')
const db = new DatabaseSync('/koishi/data/koishi.db', { readOnly: true })
const total = db.prepare('SELECT COUNT(*) n FROM kairos_messages').get()
const unsynced = db.prepare('SELECT COUNT(*) n FROM kairos_messages WHERE synced = 0').get()
const byChannel = db
  .prepare('SELECT channelId, COUNT(*) n FROM kairos_messages GROUP BY channelId ORDER BY n DESC')
  .all()
console.log(JSON.stringify({ total: total.n, unsynced: unsynced.n, byChannel }, null, 1))
