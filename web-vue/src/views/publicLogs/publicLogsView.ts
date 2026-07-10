import type {
  PublicLogEvent,
  PublicLogGroup,
  PublicLogStatus,
  PublicStats,
} from '@/types/api'

export type PublicLogMetricItem = {
  key: string
  label: string
  value: string | number
  valueClass?: string
  valueStyle?: Record<string, string>
}

const BADGE_BASE_CLASS = 'rounded-md px-2 py-0.5 text-[11px] font-semibold'

export function publicLogStatusLabel(status: PublicLogStatus) {
  if (status === 'success') return '成功'
  if (status === 'error') return '失败'
  if (status === 'timeout') return '超时'
  return '进行中'
}

export function publicLogStatusBadgeClass(status: PublicLogStatus) {
  if (status === 'success') return `${BADGE_BASE_CLASS} bg-emerald-100 text-emerald-700`
  if (status === 'error') return `${BADGE_BASE_CLASS} bg-rose-100 text-rose-700`
  if (status === 'timeout') return `${BADGE_BASE_CLASS} bg-amber-100 text-amber-700`
  return `${BADGE_BASE_CLASS} bg-amber-100 text-amber-700`
}

export function publicLogEventLabel(event: PublicLogEvent) {
  if (event.type === 'start') return '开始对话'
  if (event.type === 'select') return '选择'
  if (event.type === 'retry') return '重试'
  if (event.type === 'switch') return '切换'
  if (event.type === 'complete') {
    if (event.status === 'success') return '完成'
    if (event.status === 'error') return '失败'
    if (event.status === 'timeout') return '超时'
    return '完成'
  }
  return '事件'
}

export function publicLogEventBadgeClass(event: PublicLogEvent) {
  if (event.type === 'start') return `${BADGE_BASE_CLASS} bg-blue-100 text-blue-700`
  if (event.type === 'select') return `${BADGE_BASE_CLASS} bg-violet-100 text-violet-700`
  if (event.type === 'retry') return `${BADGE_BASE_CLASS} bg-amber-100 text-amber-700`
  if (event.type === 'switch') return `${BADGE_BASE_CLASS} bg-cyan-100 text-cyan-700`
  if (event.type === 'complete') {
    if (event.status === 'success') return `${BADGE_BASE_CLASS} bg-emerald-100 text-emerald-700`
    if (event.status === 'error') return `${BADGE_BASE_CLASS} bg-rose-100 text-rose-700`
    if (event.status === 'timeout') return `${BADGE_BASE_CLASS} bg-amber-100 text-amber-700`
  }
  return `${BADGE_BASE_CLASS} bg-slate-100 text-slate-600`
}

export function countPublicLogsByStatus(logs: readonly PublicLogGroup[], status: PublicLogStatus) {
  return logs.filter(log => log.status === status).length
}

export function averagePublicLogResponseTime(logs: readonly PublicLogGroup[]) {
  let total = 0
  let count = 0

  logs.forEach((log) => {
    if (log.status !== 'success') return
    log.events.forEach((event) => {
      if (event.type !== 'complete') return
      const match = event.content.match(/([0-9]+(?:\.[0-9]+)?)\s*s/)
      if (!match) return
      total += Number(match[1])
      count += 1
    })
  })

  if (count === 0) return '-'
  return `${(total / count).toFixed(1)}s`
}

export function publicLogSuccessRate(successCount: number, errorCount: number) {
  const completed = successCount + errorCount
  if (completed === 0) return '-'
  return `${((successCount / completed) * 100).toFixed(1)}%`
}

export function buildPublicLogMetricItems(
  logs: readonly PublicLogGroup[],
  stats: PublicStats | null | undefined,
  lastUpdated: string,
): PublicLogMetricItem[] {
  const successCount = countPublicLogsByStatus(logs, 'success')
  const errorCount = countPublicLogsByStatus(logs, 'error')

  return [
    { key: 'visitors', label: '总访客', value: stats?.total_visitors ?? 0 },
    {
      key: 'rpm',
      label: '每分钟请求',
      value: stats?.requests_per_minute ?? 0,
      valueStyle: stats?.load_color ? { color: stats.load_color } : undefined,
    },
    { key: 'avg', label: '平均响应', value: averagePublicLogResponseTime(logs) },
    { key: 'success-rate', label: '成功率', value: publicLogSuccessRate(successCount, errorCount), valueClass: 'text-emerald-600' },
    { key: 'total', label: '对话次数', value: logs.length },
    { key: 'success', label: '成功', value: successCount, valueClass: 'text-emerald-600' },
    { key: 'error', label: '失败', value: errorCount, valueClass: 'text-rose-600' },
    { key: 'updated', label: '更新时间', value: lastUpdated, valueClass: 'text-muted-foreground' },
  ]
}
