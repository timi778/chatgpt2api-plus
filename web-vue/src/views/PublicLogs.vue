<template>
  <div class="min-h-screen overflow-x-hidden bg-card/70 text-foreground backdrop-blur">
    <div class="mx-auto w-full max-w-6xl min-w-0 px-4 py-8">
      <PagePanel>
        <div class="flex flex-wrap items-start justify-between gap-4">
          <div class="flex items-center gap-3">
            <img :src="logoUrl" alt="ChatGPT2API" class="h-8 w-8 object-contain" />
            <div>
              <p class="ui-section-title">公开日志</p>
            </div>
          </div>
          <div class="flex items-center gap-2 text-xs text-muted-foreground">
            <span>自动刷新：3s</span>
          </div>
        </div>

        <InfoCard class="mt-4" tone="muted" density="compact">
          <div class="flex flex-wrap items-center justify-between gap-3">
            <div class="text-xs text-muted-foreground">
              展示最近 <span class="font-semibold text-foreground">{{ limit }}</span> 条会话日志
            </div>
            <Button
              v-if="chatUrl"
              tag="a"
              :href="chatUrl"
              target="_blank"
              size="sm"
              variant="outline"
            >
              开始对话
            </Button>
            <span v-else class="text-xs text-muted-foreground">开始对话</span>
          </div>
        </InfoCard>

        <MetricStrip
          class="mt-4"
          :items="statCards"
          columns-class="grid-cols-2 md:grid-cols-4"
          density="compact"
        />

        <ResultState
          v-if="logs.length === 0"
          class="mt-4"
          title="暂无日志"
          description="当前还没有可展示的公开会话日志。"
        />

        <div
          v-else-if="logs.length > 0"
          ref="logsScrollRef"
          class="mt-4 max-h-[60vh] overflow-y-auto pr-1 scrollbar-slim"
          @scroll="handleLogsScroll"
        >
          <div :style="{ height: `${topSpacerHeight}px` }"></div>
          <RequestLogGroup
            v-for="{ item: log } in windowedLogs"
            :key="log.request_id"
            :status-label="statusLabel(log.status)"
            :status-badge-class="statusBadgeClass(log.status)"
            :request-id="log.request_id"
            class="public-log-group"
            :collapsed="isCollapsed(log.request_id)"
            :count-text="`${log.events.length} 条事件`"
            @toggle="toggleGroup(log.request_id)"
          >
            <div class="space-y-2">
              <LogEntryRow
                v-for="event in log.events"
                :key="`${log.request_id}-${event.time}-${event.type}`"
                :time="event.time"
                :text="event.content"
                :badge-text="eventLabel(event)"
                :badge-class="eventBadgeClass(event)"
              />
            </div>
          </RequestLogGroup>
          <div :style="{ height: `${bottomSpacerHeight}px` }"></div>
        </div>
      </PagePanel>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { publicDisplayApi } from '@/api/publicDisplay'
import { publicLogsApi } from '@/api/publicLogs'
import { publicStatsApi } from '@/api/publicStats'
import { Button, ResultState } from 'nanocat-ui'
import { usePageQuery, useSerialVisibilityPolling } from '@/composables/usePageQuery'
import { usePageRuntime } from '@/composables/usePageRuntime'
import { useWindowedList } from '@/composables/useWindowedList'
import { useToast } from '@/composables/useToast'
import InfoCard from '@/components/ai/InfoCard.vue'
import LogEntryRow from '@/components/ai/LogEntryRow.vue'
import MetricStrip from '@/components/ai/MetricStrip.vue'
import PagePanel from '@/components/ai/PagePanel.vue'
import RequestLogGroup from '@/components/ai/RequestLogGroup.vue'
import { getJsonPreference, preferenceKeys, setJsonPreference } from '@/lib/preferences'
import {
  buildPublicLogMetricItems,
  publicLogEventBadgeClass,
  publicLogEventLabel,
  publicLogStatusBadgeClass,
  publicLogStatusLabel,
} from './publicLogs/publicLogsView'
import type {
  PublicDisplay,
  PublicLogGroup,
  PublicStats,
} from '@/types/api'

const logs = ref<PublicLogGroup[]>([])
const stats = ref<PublicStats | null>(null)
const display = ref<PublicDisplay | null>(null)
const toast = useToast()
const pageRuntime = usePageRuntime('public-logs')
const lastUpdated = ref('--:--')
const collapsedState = ref<Record<string, boolean>>({})
const limit = 1000
const renderLimit = 1000
const refreshIntervalMs = 3000
const PUBLIC_LOGS_REQUEST_KEY = 'public-logs:data'
const PUBLIC_DISPLAY_REQUEST_KEY = 'public-logs:display'
const PUBLIC_LOGS_REFRESH_TIMER_KEY = 'public-logs:auto-refresh'
const publicLogsLoading = ref(false)
const logsScrollRef = ref<HTMLElement | null>(null)

const publicLogsQuery = usePageQuery({
  runtime: pageRuntime,
  key: PUBLIC_LOGS_REQUEST_KEY,
  loading: publicLogsLoading,
  errorMessage: '日志加载失败',
})
const publicDisplayQuery = usePageQuery({
  runtime: pageRuntime,
  key: PUBLIC_DISPLAY_REQUEST_KEY,
})
const logoUrl = computed(() => {
  const url = display.value?.logo_url?.trim()
  return url || '/logo.svg'
})
const chatUrl = computed(() => display.value?.chat_url?.trim() || '')

const visibleLogs = computed(() => {
  if (logs.value.length > renderLimit) {
    return logs.value.slice(-renderLimit)
  }
  return logs.value
})
const windowedLogsDisabled = computed(() => visibleLogs.value.length <= 120)
const {
  visibleItems: windowedLogs,
  topSpacerHeight,
  bottomSpacerHeight,
  handleScroll: handleLogsScroll,
  setScrollElement: setLogsScrollElement,
} = useWindowedList({
  items: visibleLogs,
  itemHeight: 92,
  overscan: 8,
  disabled: windowedLogsDisabled,
})
const statCards = computed(() => buildPublicLogMetricItems(logs.value, stats.value, lastUpdated.value))
const statusLabel = publicLogStatusLabel
const statusBadgeClass = publicLogStatusBadgeClass
const eventLabel = publicLogEventLabel
const eventBadgeClass = publicLogEventBadgeClass

const loadCollapseState = () => {
  collapsedState.value = getJsonPreference<Record<string, boolean>>(preferenceKeys.publicLogFoldState, {})
}

const saveCollapseState = () => {
  setJsonPreference(preferenceKeys.publicLogFoldState, collapsedState.value)
}

const isCollapsed = (requestId: string) => collapsedState.value[requestId] === true

const toggleGroup = (requestId: string) => {
  collapsedState.value[requestId] = !isCollapsed(requestId)
  saveCollapseState()
}

const fetchData = async () => {
  if (publicLogsLoading.value) return
  await publicLogsQuery.run(
    () => Promise.all([
      publicLogsApi.list({ limit }),
      publicStatsApi.overview(),
    ]),
    {
      apply: ([logsResponse, statsResponse]) => {
        logs.value = logsResponse.logs
        stats.value = statsResponse
        lastUpdated.value = new Date().toLocaleTimeString('zh-CN', {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        })
      },
      onError: (message) => {
        toast.error(message)
      },
    },
  )
}

const fetchDisplay = async () => {
  await publicDisplayQuery.run(
    () => publicDisplayApi.overview(),
    {
      apply: (response) => {
        display.value = response
      },
      onError: () => {
        display.value = null
      },
    },
  )
}

const publicLogsPolling = useSerialVisibilityPolling({
  runtime: pageRuntime,
  key: PUBLIC_LOGS_REFRESH_TIMER_KEY,
  intervalMs: refreshIntervalMs,
  action: fetchData,
})

watch(logsScrollRef, (element) => setLogsScrollElement(element), { flush: 'post' })

pageRuntime.onActivate(({ initial }) => {
  setLogsScrollElement(logsScrollRef.value)
  if (!initial) {
    publicLogsPolling.start()
    return
  }
  loadCollapseState()
  fetchDisplay()
  fetchData()
  publicLogsPolling.start()
})

pageRuntime.onDeactivate(() => {
  publicLogsQuery.invalidate()
  publicDisplayQuery.invalidate()
  publicLogsPolling.stop()
})

pageRuntime.onHide(() => {
  publicLogsQuery.invalidate()
  publicDisplayQuery.invalidate()
  publicLogsPolling.stop()
})

pageRuntime.onShow(() => {
  setLogsScrollElement(logsScrollRef.value)
  publicLogsPolling.start()
})
</script>

<style scoped>
.public-log-group + .public-log-group {
  margin-top: 0.75rem;
}
</style>
