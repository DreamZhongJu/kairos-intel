const { Schema, Logger } = require('koishi')
const { tool } = require('@langchain/core/tools')
const { z } = require('zod')

const logger = new Logger('kairos-graph')

exports.name = 'kairos-graph'
exports.using = ['http']
exports.inject = { required: ['chatluna'] }

exports.Config = Schema.object({
  kairosEndpoint: Schema.string().role('link').default('http://kairos:8095')
    .description('Kairós 知识服务地址。'),
  apiToken: Schema.string().default('').description('KAIROS_API_TOKEN，未设置则留空。'),
  whitelistGroups: Schema.array(Schema.string()).default(['830070676'])
    .description('允许使用图谱检索的群号（内测灰度白名单）。'),
  timeoutMs: Schema.natural().default(8000).description('请求超时（毫秒）。'),
})

function createGraphTool(ctx, config) {
  return tool(async ({ query }, runConfig) => {
    const session = runConfig?.configurable?.session
    if (!session) return '当前没有可用的会话。'
    const channelId = String(session.channelId || '')
    if (!config.whitelistGroups.includes(channelId)) {
      return '知识图谱检索功能内测中，仅限指定群使用。'
    }
    const q = String(query || '').trim()
    if (!q) return '请提供要检索的问题。'
    try {
      const url = `${config.kairosEndpoint}/api/knowledge/query?q=${encodeURIComponent(q)}`
      const headers = config.apiToken ? { 'X-Token': config.apiToken } : {}
      // Node native fetch on purpose: it bypasses both the env proxy and the
      // global koishi proxy-agent, either of which would route this intranet
      // request through clash and break it.
      const response = await fetch(url, {
        headers,
        signal: AbortSignal.timeout(config.timeoutMs),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      const evidence = (data.hyper && data.hyper.evidence) || []
      if (!evidence.length) {
        const graphLines = (data.graph && data.graph.lines) || []
        if (!graphLines.length) return '图谱中没有找到相关记忆。'
        return `【图谱关联】\n${graphLines.slice(0, 8).join('\n')}`
      }
      const linked = ((data.hyper && data.hyper.linked) || []).map((e) => e.name).join('、')
      return `【推理链证据】（关联：${linked}）\n${evidence.slice(0, 6).join('\n')}`
    } catch (error) {
      logger.warn('图谱检索失败：%s', error.message)
      return `图谱检索失败：${error.message}`
    }
  }, {
    name: 'search_kairos_graph',
    description: [
      '检索群聊知识图谱，获取关于群友和往事的推理链证据。',
      '当需要回答以下问题时必须主动调用：',
      '某位群友的学校/家乡/所在地/喜好/在玩什么等个人信息；',
      '群友之间的关系（谁认识谁、怎么认识的）；',
      '群里过去发生的事、大家聊过的话题。',
      '输入用自然语言问题即可，例如"康哥在哪上学"。不要编造群友信息，查不到就说不知道。',
    ].join(''),
    schema: z.object({
      query: z.string().describe('要检索的自然语言问题或关键词'),
    }),
  })
}

exports.apply = (ctx, config) => {
  ctx.inject(['chatluna'], (agentCtx) => {
    const dispose = agentCtx.chatluna.platform.registerTool('search_kairos_graph', {
      description: '检索群聊知识图谱中的群友信息与往事记忆。',
      createTool: () => createGraphTool(agentCtx, config),
      selector: () => true,
      meta: {
        source: 'plugin',
        group: 'memory',
        tags: ['knowledge', 'graph', 'memory'],
        defaultAvailability: {
          enabled: true,
          main: true,
          chatluna: true,
          characterScope: 'all',
        },
      },
    })
    agentCtx.on('dispose', dispose)
    logger.info('Kairós graph tool registered (whitelist: %s)', config.whitelistGroups.join(','))
  })
}
