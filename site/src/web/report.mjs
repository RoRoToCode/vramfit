/**
 * 把 core 算出来的结果翻译成「人话 + HTML」。
 *
 * 这里刻意不复用命令行的 render —— 那是 ANSI 文本，塞进网页就是一坨转义码。
 * 但判据必须完全一致，所以结论、档位、速度全部取自 core，这里只做排版。
 */

import {
  GB, FILE_TYPE, quantFromName, paramCountOf, shardInfo, totalWeightsBytes,
  analyzeTemplate, attentionReport, stripControl, baseName, fmtSize,
  headlineOf, TUNING, ASSUMPTIONS
} from '../core/index.mjs'

const esc = (s) => String(s ?? '')
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&#39;')

/**
 * 外部来的名字先洗控制字符。
 *
 * 这里只洗不转义：转义必须发生在拼 HTML 的那一刻，而且只能发生一次。
 * 之前的 safe() 同时做了两件事，buildView 洗完一遍、renderView 又 esc 一遍，
 * 文件名里有个 & 就变成 &amp;amp; 显示在页面上。
 */
const clean = (s) => stripControl(String(s ?? ''))

/** 直接进 HTML 时用这个：先洗再转义，一步到位 */
const safe = (s) => esc(clean(s))

const fmtGB = (b) => `${(b / GB).toFixed(2)} GB`
const fmtCtx = (n) => (n >= 1024 ? `${Math.round(n / 1024)}K` : String(n))
const fmtParams = (n) => {
  if (!Number.isFinite(n) || n <= 0) return '—'
  const b = n / 1e9
  return b >= 10 ? `${b.toFixed(0)}B` : b >= 1 ? `${b.toFixed(1)}B` : `${b.toFixed(2)}B`
}

/** 显存里各项占多少，画成一条堆叠条 */
function breakdown(result) {
  const kv = result.best ? result.best.kv : 0
  // 进度条必须按「权重 + KV + 运行时开销」算 —— 只算权重会比标题数字小一大截，
  // 用户看到 7.13 GB / 7.52 GB 配一条 62% 的条，只会以为工具算错了。
  const total = result.weights + kv + result.overhead
  return {
    weights: result.weights,
    kv,
    overhead: result.overhead,
    total,
    limit: result.limit,
    ratio: Math.min(1, total / (result.limit || 1))
  }
}

export function buildView(meta, result, ctx = {}) {
  const quant = quantFromName(meta.file) ?? (meta.fileType !== undefined ? FILE_TYPE[meta.fileType] : undefined) ?? '未知'
  const shard = shardInfo(meta.file)
  const weights = totalWeightsBytes(meta)
  const speed = result.speed?.onVram
  const sp = Number.isFinite(speed) && speed > 0 ? Math.round(speed) : null
  const att = attentionReport(meta)
  const tpl = analyzeTemplate(meta.chatTemplate)
  const bd = breakdown(result)

  return {
    // 这几个字段来自远端文件名，只洗控制字符，转义留给拼 HTML 的地方做一次
    name: clean(meta.name || baseName(meta.file)),
    file: clean(baseName(meta.file)),
    arch: meta.arch ? clean(meta.arch) : null,
    quant: clean(quant),
    size: fmtGB(weights),
    params: fmtParams(paramCountOf(meta.file, meta.paramCount)),
    layers: meta.blockCount ?? null,
    headKv: meta.headKv ?? null,
    native: meta.contextLength ?? null,
    shard: shard && shard.total > 1 ? `${shard.index}/${shard.total}` : null,
    verdict: headlineOf(meta, result),
    bestCtx: result.best?.ctx ?? null,
    bestCtxText: result.best ? fmtCtx(result.best.ctx) : '—',
    speed: sp,
    speedOnCpu: Number.isFinite(result.speed?.onRam) && result.speed.onRam > 0
      ? Math.round(result.speed.onRam)
      : null,
    bandwidth: result.speed?.bandwidth ?? null,
    gpuName: result.gpu ? clean(result.gpu.name) : null,
    estimated: result.estimated !== false,
    limit: bd.limit,
    limitText: fmtGB(result.limit),
    totalBytes: bd.total,
    totalText: fmtGB(bd.total),
    weightsBytes: bd.weights,
    kvBytes: bd.kv,
    overheadBytes: bd.overhead,
    ratio: bd.ratio,
    onCpu: result.onCpu,
    tiers: result.rows.map((r) => ({
      ctx: r.ctx,
      ctxText: fmtCtx(r.ctx),
      need: r.need,
      needText: r.need === null ? '算不出' : fmtGB(r.need),
      fits: r.fits,
      tight: r.tight,
      beyondNative: r.beyondNative,
      best: result.best?.ctx === r.ctx
    })),
    template: { level: tpl.level, notes: tpl.notes },
    attention: att.swa
      ? `${att.kvLayers} 层带 KV，其中 ${att.swa.fullLayers} 层全注意力、滑窗窗口 ${fmtCtx(att.swa.window)}`
      : att.intervalInferred
        ? `${att.kvLayers} 层带 KV（混合线性注意力，按 ${att.interval}:1 推断）`
        : `${att.kvLayers} 层带 KV`,
    moe: meta.expertCount ? `${meta.expertCount} 专家，每 token 激活 ${meta.expertUsedCount ?? '?'} 个` : null,
    kvType: ctx.kvType ?? 'f16'
  }
}

const LEVEL_TEXT = { strong: '支持工具调用', partial: '支持工具调用（部分）', weak: '格式不全', none: '不支持' }

export function renderView(v) {
  const meter = Math.max(2, Math.min(100, Math.round(v.ratio * 100)))
  // CPU 推理时 onVram 是 null，只算过 onRam；以前这里只看 onVram，
  // 于是没插独显的机器上一律显示「带宽未知，不给估计」，而 CPU 速度其实算得出来。
  const speedText = v.onCpu
    ? (v.speedOnCpu ? `CPU 约 ${v.speedOnCpu} tok/s` : '带宽未知，不给估计')
    : v.speed ? `约 ${v.speed} tok/s` : '带宽未知，不给估计'

  const tierRows = v.tiers.map((t) => {
    const cls = t.beyondNative ? 'beyond' : t.fits ? (t.tight ? 'tight' : 'ok') : 'no'
    const mark = t.beyondNative ? '超出原生' : t.fits ? (t.tight ? '勉强' : '够') : '不够'
    return `<div class="tier ${cls}${t.best ? ' is-best' : ''}">
      <span class="t-ctx">${esc(t.ctxText)}</span>
      <span class="t-bar"><i style="width:${Math.min(100, Math.round(((t.need || 0) / (v.limit || 1)) * 100))}%"></i></span>
      <span class="t-need">${esc(t.needText)}</span>
      <span class="t-mark">${mark}</span>
    </div>`
  }).join('')

  const infoRows = [
    ['架构', v.arch ?? '未知', '读到'],
    ['参数量', v.params, '读到'],
    ['层数', v.layers ? `${v.layers} 层 · ${v.attention}` : '未知', '读到'],
    ['原生上下文', v.native ? fmtCtx(v.native) : '未知', '读到'],
    ['对话模板', LEVEL_TEXT[v.template.level] ?? v.template.level, '读到'],
    ['KV 精度', v.kvType === 'f16' ? 'f16（默认）' : v.kvType, '设定']
  ]
  if (v.moe) infoRows.push(['MoE', v.moe, '读到'])
  if (v.shard) infoRows.push(['分片', `第 ${v.shard} 片，体积按整模估算`, '算出'])

  const info = infoRows.map(([k, val, src]) =>
    `<div class="kv-row"><span class="k">${esc(k)}</span><span class="v">${esc(val)}</span><span class="src">${esc(src)}</span></div>`
  ).join('')

  // notes 是 {ok, text} 结构（core 层统一改造过）：以前直接 esc(字符串)，
  // 现在直接把对象塞进模板会渲染成 [object Object]。
  const tplNotes = v.template.notes
    .map((n) => `<li>${n.ok ? '<i class="ok">✓</i> ' : ''}${esc(n.text ?? n)}</li>`)
    .join('')

  // 算不出 KV 时不要画那条进度条：它的分母只有权重，进度条会显示成「几乎占满」，
  // 和「算不出来」的结论自相矛盾。
  const budget = v.estimated
    ? `<div class="meter"><i style="width:${meter}%"></i></div>
      <div class="b-legend">
        <span><i class="dot w"></i>权重 ${fmtGB(v.weightsBytes)}</span>
        <span><i class="dot k"></i>KV ${v.kvBytes ? fmtGB(v.kvBytes) : '—'}</span>
        <span><i class="dot o"></i>开销 ${fmtSize(v.overheadBytes)}</span>
        <span class="spd">${esc(speedText)}</span>
      </div>`
    : `<div class="b-legend"><span><i class="dot w"></i>权重 ${fmtGB(v.weightsBytes)}</span>
        <span class="spd">${esc(speedText)}</span></div>
      <p class="note">缺元数据，KV cache 无法推算，所以只有权重体积可信。</p>`

  return `<article class="card">
    <header class="verdict" data-level="${v.verdict.level}">
      <div class="v-main">
        <h2>${esc(v.verdict.title)}</h2>
        <p>${esc(v.verdict.why)}</p>
      </div>
      <div class="v-side">
        <div class="v-num">${v.bestCtx ? esc(v.bestCtxText) : '—'}</div>
        <div class="v-cap">推荐上下文</div>
      </div>
    </header>

    <div class="head">
      <div class="h-name">${esc(v.name)}</div>
      <div class="h-meta">
        <span class="chip">${esc(v.quant)}</span>
        <span class="chip">${esc(v.size)}</span>
        ${v.gpuName ? `<span class="chip ghost">${esc(v.gpuName)}</span>` : ''}
      </div>
    </div>

    <section class="budget">
      <div class="b-head">
        <span>显存占用</span>
        <span class="b-num">${esc(v.totalText)} / ${esc(v.limitText)}</span>
      </div>
      ${budget}
      ${v.onCpu ? '<p class="note">没有可用显卡，按 CPU 推理估算</p>' : ''}
    </section>

    <section class="tiers">
      <h3>不同上下文要多少显存</h3>
      ${tierRows}
    </section>

    <section class="info">
      <h3>模型信息</h3>
      ${info}
    </section>

    <details class="tpl">
      <summary>工具调用能力（${esc(LEVEL_TEXT[v.template.level] ?? v.template.level)}）</summary>
      <ul>${tplNotes}</ul>
    </details>

    <footer class="disclaimer">
      <strong>以上全部是静态推算，不是实测。</strong>
      权重按文件体积、KV 按层数×head、速度按带宽÷权重算出来的，误差通常在 ±${TUNING.speedErrorPct}% 以内，
      但要看实际驱动和参数设置。模板里有 tools 分支也不等于模型真会调工具。
      结论只用来排掉「根本不可能」的选项，不能当合格证 —— 请实际加载跑一次再决定。
    </footer>
  </article>`
}

/** 量化对比表：一个仓库的全部变体 */
export function renderRepoTable(repo, rows, hwName) {
  const sorted = [...rows].sort((a, b) => {
    // 读失败的行永远沉底：它们没有量化信息，混在中间会让人以为也是候选
    if (!!a.error !== !!b.error) return a.error ? 1 : -1
    if (a.error && b.error) return 0
    const qa = a.quant ?? '', qb = b.quant ?? ''
    const na = Number((/Q(\d+)/i.exec(qa) ?? [])[1] ?? 99)
    const nb = Number((/Q(\d+)/i.exec(qb) ?? [])[1] ?? 99)
    return nb - na || (a.weights ?? 0) - (b.weights ?? 0)
  })

  const body = sorted.map((r) => {
    // 行得能点。以前表格底下写着「点任意一行可以看完整报告」，但 tr 上根本没有
    // data-url，事件绑不上去 —— 文案在承诺一个不存在的行为。
    const url = r.url ? ` data-url="${safe(r.url)}"` : ''
    if (r.error) {
      return `<tr class="err"${url}><td>${safe(r.file ?? '?')}</td><td colspan="4">${safe(r.error)}</td></tr>`
    }
    const v = r.view ?? r
    const cls = v.verdict.level
    return `<tr class="${cls}"${url}>
      <td class="q">${esc(v.quant)}</td>
      <td class="num">${esc(v.size)}</td>
      <td class="num">${v.bestCtx ? esc(v.bestCtxText) : '—'}</td>
      <td class="num">${v.speed ? `~${v.speed} tok/s` : '—'}</td>
      <td class="res">${esc(v.verdict.short ?? v.verdict.title)}</td>
    </tr>`
  }).join('')

  const clickable = sorted.some((r) => r.url)

  return `<article class="card">
    <div class="head">
      <div class="h-name">${safe(repo)}</div>
      <div class="h-meta"><span class="chip">${rows.length} 个变体</span>${hwName ? `<span class="chip ghost">${safe(hwName)}</span>` : ''}</div>
    </div>
    <table class="repo">
      <thead><tr><th>量化</th><th class="num">体积</th><th class="num">上下文</th><th class="num">速度</th><th class="res">结论</th></tr></thead>
      <tbody>${body}</tbody>
    </table>
    <footer class="disclaimer">
      静态推算，仅供参考；体积、上下文、速度都是估算值，下载后仍要实际加载验证。
      ${clickable ? '点任意一行可以看这个量化的完整报告。' : ''}
    </footer>
  </article>`
}

/**
 * 「参数与假设」面板。
 *
 * 网页上不写死文案，直接渲染 core/assumptions.mjs 里的清单 ——
 * 和 assess() 用的是同一份常量，改一处三处同步，不会出现「文档说 200 MiB、
 * 代码算 400 MiB」这种最难查的问题。
 */
export function renderAssumptions() {
  const groups = ASSUMPTIONS.map((g) => `
    <div class="asm-group">
      <h4>${esc(g.group)}</h4>
      <dl>${g.items.map((it) => `
        <dt>${esc(it.k)}</dt>
        <dd><b>${esc(it.v)}</b>${it.note ? `<span>${esc(it.note)}</span>` : ''}</dd>`).join('')}
      </dl>
    </div>`).join('')

  return `<details class="assumptions">
    <summary>这些数字是怎么算出来的（参数与假设）</summary>
    <p class="asm-lead">
      本工具不加载模型、不跑推理。下面每一个数字都写死在
      <code>core/assumptions.mjs</code> 里，报告、网页、README 显示的是同一份。
      改动它们等于改结论，所以列出完整清单，而不是只给一句「仅供参考」。
    </p>
    ${groups}
    <div class="asm-group">
      <h4>怎么读这份报告</h4>
      <dl>
        <dt>读到</dt><dd><b>直接从 GGUF 元数据里取出的值</b><span>架构、层数、head 数、原生上下文</span></dd>
        <dt>算出</dt><dd><b>用上面的参数推出来的值</b><span>KV cache、显存占用、推荐上下文</span></dd>
        <dt>设定</dt><dd><b>你在这页面上选的</b><span>显卡、显存、已占用、KV 精度</span></dd>
      </dl>
    </div>
  </details>`
}
