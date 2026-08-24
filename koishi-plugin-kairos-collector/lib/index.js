'use strict'

const { Schema } = require('koishi')

exports.name = 'kairos-collector'
exports.inject = ['database']

exports.Config = Schema.object({
  groups: Schema.array(Schema.string()).default([]).description('采集白名单群号'),
  kairosEndpoint: Schema.string().default('http://kairos:8095').description('凯伊服务地址（容器内可达）'),
  apiToken: Schema.string().default('').description('KAIROS_API_TOKEN，服务端未启用则留空'),
  flushCount: Schema.number().min(4).max(500).step(1).default(40).description('单群攒满多少条触发推送'),
  flushInterval: Schema.number().min(30).max(86400).step(1).default(300).description('定时推送间隔（秒）'),
  batchSize: Schema.number().min(10).max(1000).step(1).default(200).description('单次推送最大消息条数'),
})

exports.apply = (ctx, cfg) => {
  ctx.model.extend('kairos_messages', {
    id: 'unsigned',
    platform: 'string',
    channelId: 'string',
    userId: 'string',
    nickname: 'string',
    content: 'text',
    time: 'timestamp',
    synced: 'boolean',
  }, { autoInc: true })

  const logger = ctx.logger('kairos-collector')
  const headers = {}
  if (cfg.apiToken) headers['X-Token'] = cfg.apiToken

  let flushing = false
  let failStreak = 0
  const pending = new Map()

  const postJson = async (path, data) =>
    ctx.http.post(cfg.kairosEndpoint + path, data, { headers, timeout: 120000 })

  // ---- collection -------------------------------------------------------
  ctx.on('message', async (session) => {
    try {
      if (!cfg.groups.length) return
      if (!cfg.groups.includes(session.channelId)) return
      if (session.selfId && session.userId === session.selfId) return
      const content = (session.content || '').trim()
      if (!content) return
      logger.info('capturing %s user %s: %s', session.channelId, session.userId, content.slice(0, 40))
      await ctx.database.create('kairos_messages', {
        platform: session.platform || '',
        channelId: session.channelId,
        userId: session.userId || '',
        nickname:
          (session.author && (session.author.nick || session.author.name)) ||
          session.username ||
          session.userId ||
          '',
        content,
        time: session.timestamp ? new Date(session.timestamp) : new Date(),
        synced: false,
      })
      const n = (pending.get(session.channelId) || 0) + 1
      pending.set(session.channelId, n)
      if (n >= cfg.flushCount) {
        pending.set(session.channelId, 0)
        setTimeout(() => { flushAll() }, 3000)
      }
    } catch (err) {
      logger.warn('save failed: %s', err.message)
    }
  })

  // ---- flushing ---------------------------------------------------------
  async function flushChannel(channelId) {
    let total = 0
    for (;;) {
      const rows = await ctx.database.get(
        'kairos_messages',
        { channelId, synced: false },
        { limit: cfg.batchSize, sort: { id: 'asc' } },
      )
      if (rows.length < 2) break // a window needs at least two messages
      const payload = rows.map((r) => ({
        user_id: r.userId,
        nickname: r.nickname,
        time: r.time ? new Date(r.time).toISOString().replace('T', ' ').slice(0, 19) : '',
        text: r.content,
      }))
      await postJson('/api/knowledge/ingest', {
        source: 'qq',
        channel_id: channelId,
        title: '',
        messages: payload,
      })
      await ctx.database.set('kairos_messages', { id: rows.map((r) => r.id) }, { synced: true })
      total += rows.length
      if (rows.length < cfg.batchSize) break
    }
    return total
  }

  async function flushAll() {
    if (flushing) return
    flushing = true
    let pushedAny = false
    try {
      for (const ch of cfg.groups) {
        try {
          const n = await flushChannel(ch)
          if (n > 0) {
            pushedAny = true
            logger.info('pushed %d messages of group %s to kairos', n, ch)
          }
        } catch (err) {
          failStreak += 1
          if (failStreak <= 3 || failStreak % 20 === 0) {
            logger.warn('flush failed for group %s (%d in a row): %s', ch, failStreak, err.message)
          }
        }
      }
      if (pushedAny) failStreak = 0
    } finally {
      flushing = false
    }
  }

  ctx.setInterval(() => { flushAll() }, cfg.flushInterval * 1000)

  // ---- commands ----------------------------------------------------------
  ctx.command('kairos.query <text:text>', '查询凯伊知识库/图谱')
    .alias('凯伊查询')
    .action(async ({ session }, text) => {
      if (!text || !text.trim()) return '用法：kairos.query <关键词或问题>'
      try {
        const res = await ctx.http.get(cfg.kairosEndpoint + '/api/knowledge/query', {
          params: { q: text.trim(), limit: 6 },
          headers,
          timeout: 30000,
        })
        const ans = (res && res.answer) || '没有结果。'
        return ans.length > 1200 ? ans.slice(0, 1200) + '\n…' : ans
      } catch (err) {
        logger.warn('query failed: %s', err.message)
        return '查询失败：' + err.message
      }
    })

  ctx.command('kairos.status', '查看凯伊知识图谱统计')
    .action(async () => {
      try {
        const s = await ctx.http.get(cfg.kairosEndpoint + '/api/knowledge/stats', { headers, timeout: 15000 })
        return `文档 ${s.documents} · 实体 ${s.entities} · 关系 ${s.relations}`
      } catch (err) {
        return '获取失败：' + err.message
      }
    })

  ctx.command('kairos.flush', '立即把未同步的聊天记录推给凯伊')
    .userFields(['authority'])
    .before(({ session }) => session.user && session.user.authority >= 2 ? true : '权限不足')
    .action(async () => {
      await flushAll()
      return '已执行推送。'
    })
}
