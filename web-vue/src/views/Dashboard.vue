<template>
  <div class="space-y-5">
    <PageLoadingState
      v-if="!dashboardDataReady"
      title="正在加载概览"
      description="读取最新账号、调用趋势和模型统计。"
    />

    <template v-else>
    <section class="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
      <StatCard
        v-for="stat in stats"
        :key="stat.label"
        :label="stat.label"
        :value="stat.value"
        :caption="stat.meta"
        :icon="stat.icon"
        :icon-bg="stat.iconBg"
        :icon-color="stat.iconColor"
      />
    </section>

    <section class="grid grid-cols-1 gap-3 md:grid-cols-3 xl:grid-cols-6">
      <div class="rounded-lg border border-border bg-card/80 px-3 py-2.5 shadow-sm md:col-span-2 xl:col-span-2">
        <div class="flex items-start justify-between gap-3">
          <div class="flex min-w-0 items-start gap-2.5">
            <span class="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-sky-50 text-sky-600">
              <Icon :icon="autoRefreshStatus.icon" class="h-4 w-4" :class="{ 'animate-spin': autoRefreshStatus.running }" />
            </span>
            <div class="min-w-0">
              <div class="flex flex-wrap items-center gap-2">
                <p class="text-sm font-medium text-foreground">账号自动刷新</p>
                <StateBadge :tone="autoRefreshStatus.tone" shape="rounded" size="xs" :bordered="false">
                  {{ autoRefreshStatus.label }}
                </StateBadge>
              </div>
              <p class="mt-1 truncate text-xs text-muted-foreground" :title="autoRefreshStatus.timeText">
                {{ autoRefreshStatus.timeText }}
              </p>
            </div>
          </div>
          <p v-if="autoRefreshStatus.nextRunText" class="shrink-0 text-xs text-muted-foreground">
            {{ autoRefreshStatus.nextRunText }}
          </p>
        </div>
        <div class="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
          <span>{{ autoRefreshStatus.summaryText }}</span>
          <span v-if="autoRefreshStatus.detailText" class="min-w-0 truncate" :title="autoRefreshStatus.detailText">
            {{ autoRefreshStatus.detailText }}
          </span>
        </div>
      </div>
    </section>

    <section class="grid grid-cols-1 gap-4">
      <ChartCard title="模型请求分布">
        <template #actions>
          <TimeRangeTabs v-model="timeRangeHourlyRequests" aria-label="模型请求分布时间范围" />
        </template>
        <div ref="hourlyRequestsChartRef" class="h-72 w-full px-2"></div>
      </ChartCard>
    </section>

    <section class="grid grid-cols-1 gap-4">
      <ChartCard title="调用趋势">
        <template #actions>
          <TimeRangeTabs v-model="timeRangeTrend" aria-label="调用趋势时间范围" />
        </template>
        <div ref="trendChartRef" class="h-56 w-full"></div>
      </ChartCard>
    </section>

    <section class="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <ChartCard title="成功率趋势">
        <template #actions>
          <TimeRangeTabs v-model="timeRangeSuccessRate" aria-label="成功率趋势时间范围" />
        </template>
        <div ref="successRateChartRef" class="h-56 w-full"></div>
      </ChartCard>

      <ChartCard title="平均响应时间">
        <template #actions>
          <TimeRangeTabs v-model="timeRangeResponseTime" aria-label="平均响应时间范围" />
        </template>
        <div ref="responseTimeChartRef" class="h-56 w-full"></div>
      </ChartCard>
    </section>

    <section class="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <ChartCard title="模型调用占比">
        <template #actions>
          <TimeRangeTabs v-model="timeRangeModel" aria-label="模型调用占比时间范围" />
        </template>
        <div ref="modelChartRef" class="h-56 w-full"></div>
      </ChartCard>

      <ChartCard title="模型使用排行">
        <template #actions>
          <TimeRangeTabs v-model="timeRangeModelRank" aria-label="模型使用排行时间范围" />
        </template>
        <div ref="modelRankChartRef" class="h-56 w-full"></div>
      </ChartCard>
    </section>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ChartCard, StatCard } from 'nanocat-ui'
import { Icon } from '@iconify/vue'
import PageLoadingState from '@/components/ai/PageLoadingState.vue'
import StateBadge from '@/components/ai/StateBadge.vue'
import TimeRangeTabs from '@/components/ai/TimeRangeTabs.vue'
import { useDashboardPage } from './dashboard/useDashboardPage'

defineOptions({ name: 'Dashboard' })

const {
  stats,
  autoRefreshStatus,
  dashboardDataReady,
  timeRangeHourlyRequests,
  timeRangeTrend,
  timeRangeSuccessRate,
  timeRangeModel,
  timeRangeModelRank,
  timeRangeResponseTime,
  hourlyRequestsChartRef,
  trendChartRef,
  successRateChartRef,
  responseTimeChartRef,
  modelChartRef,
  modelRankChartRef,
} = useDashboardPage()
</script>
