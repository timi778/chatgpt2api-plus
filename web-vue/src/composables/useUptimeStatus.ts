import { computed, ref } from 'vue'
import { monitorApi } from '@/api/monitor'
import type { UptimeHeartbeat, UptimeResponse, UptimeService } from '@/types/api'
import { usePageQuery, useSerialVisibilityPolling } from '@/composables/usePageQuery'
import { usePageRuntime } from '@/composables/usePageRuntime'
import { useToast } from '@/composables/useToast'

type ServiceView = {
  key: string
  name: string
  statusLabel: string
  statusClass: string
  uptime: number
  total: number
  success: number
  beats: Array<{ className: string; tooltip: string | null }>
}

const slowThresholdMs = 40000
const maxBeats = 60
const AUTO_REFRESH_INTERVAL_MS = 15000
const UPTIME_REQUEST_KEY = 'uptime:status'
const UPTIME_REFRESH_TIMER_KEY = 'uptime:auto-refresh'

const mapStatusLabel = (statusValue: UptimeService['status']) => {
  if (statusValue === 'up') return '正常'
  if (statusValue === 'warn') return '注意'
  if (statusValue === 'down') return '异常'
  return '未知'
}

const mapStatusClass = (statusValue: UptimeService['status']) => {
  if (statusValue === 'up') return 'monitor-badge--up'
  if (statusValue === 'warn') return 'monitor-badge--warn'
  if (statusValue === 'down') return 'monitor-badge--down'
  return 'monitor-badge--unknown'
}

const buildBeats = (heartbeats: UptimeHeartbeat[] = []) => {
  const beats: Array<{ className: string; tooltip: string | null }> = []
  for (let i = 0; i < maxBeats; i += 1) {
    if (i < heartbeats.length) {
      const beat = heartbeats[i]
      const latencyMs = beat.latency_ms ?? null
      const isSlow = beat.success && latencyMs !== null && latencyMs > slowThresholdMs
      const level = beat.level ?? (isSlow ? 'warn' : (beat.success ? 'up' : 'down'))
      const className = level === 'warn'
        ? 'monitor-beat--warn'
        : (level === 'up' ? 'monitor-beat--up' : 'monitor-beat--down')
      const latencyText = latencyMs !== null
        ? ` · 首响 ${(Math.max(latencyMs, 0) / 1000).toFixed(1)}s`
        : ''
      const statusCodeText = beat.status_code ? ` · HTTP ${beat.status_code}` : ''
      const statusText = level === 'warn' ? '警告' : (beat.success ? '成功' : '失败')

      beats.push({
        className,
        tooltip: `${beat.time} · ${statusText}${statusCodeText}${latencyText}`,
      })
    } else {
      beats.push({ className: 'monitor-beat--empty', tooltip: null })
    }
  }
  return beats
}

export function useUptimeStatus() {
  const toast = useToast()
  const pageRuntime = usePageRuntime('uptime')
  const status = ref<UptimeResponse | null>(null)
  const errorMessage = ref('')
  const isLoading = ref(false)
  const uptimeQuery = usePageQuery({
    runtime: pageRuntime,
    key: UPTIME_REQUEST_KEY,
    loading: isLoading,
    error: errorMessage,
    errorMessage: '监控数据获取失败',
  })
  const updatedAt = computed(() => status.value?.updated_at ?? '')

  const services = computed<ServiceView[]>(() => {
    if (!status.value) return []

    const serviceMap = status.value.services && typeof status.value.services === 'object'
      ? status.value.services
      : {}

    return Object.entries(serviceMap).map(([key, service]) => ({
      key,
      name: service.name || key,
      statusLabel: mapStatusLabel(service.status),
      statusClass: mapStatusClass(service.status),
      uptime: Number(service.uptime || 0),
      total: Number(service.total || 0),
      success: Number(service.success || 0),
      beats: buildBeats(Array.isArray(service.heartbeats) ? service.heartbeats : []),
    }))
  })

  const refreshStatus = async () => {
    if (isLoading.value) return
    await uptimeQuery.run(
      () => monitorApi.uptime(),
      {
        apply: (response) => {
          status.value = response
        },
        onError: (message) => {
          errorMessage.value = ''
          toast.error(message)
        },
      },
    )
  }

  const uptimePolling = useSerialVisibilityPolling({
    runtime: pageRuntime,
    key: UPTIME_REFRESH_TIMER_KEY,
    intervalMs: AUTO_REFRESH_INTERVAL_MS,
    action: refreshStatus,
  })

  pageRuntime.onActivate(() => {
    void refreshStatus()
    uptimePolling.start()
  })

  pageRuntime.onShow(() => {
    void refreshStatus()
    uptimePolling.start()
  })

  pageRuntime.onDeactivate(() => {
    uptimeQuery.invalidate()
    uptimePolling.stop()
  })

  pageRuntime.onHide(() => {
    uptimeQuery.invalidate()
    uptimePolling.stop()
  })

  return {
    services,
    updatedAt,
    errorMessage,
    isLoading,
    refreshStatus,
  }
}
