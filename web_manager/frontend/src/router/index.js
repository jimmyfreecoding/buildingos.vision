import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    redirect: '/dashboard'
  },
  {
    path: '/dashboard',
    component: () => import('../views/Dashboard.vue')
  },
  {
    path: '/ai-monitor',
    component: () => import('../views/AIMonitor.vue')
  },
  {
    path: '/cameras',
    component: () => import('../views/Cameras.vue')
  },
  {
    path: '/ai-params',
    component: () => import('../views/AIParams.vue')
  },
  {
    path: '/occupancy-logs',
    component: () => import('../views/OccupancyLogs.vue')
  },
  {
    path: '/network',
    component: () => import('../views/Network.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
