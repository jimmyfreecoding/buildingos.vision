import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('../views/Login.vue')
  },
  {
    path: '/',
    redirect: '/dashboard'
  },
  {
    path: '/dashboard',
    name: 'Dashboard',
    component: () => import('../views/Dashboard.vue')
  },
  {
    path: '/ai-monitor',
    name: 'AIMonitor',
    component: () => import('../views/AIMonitor.vue')
  },
  {
    path: '/cameras',
    name: 'Cameras',
    component: () => import('../views/Cameras.vue')
  },
  {
    path: '/ai-params',
    name: 'AIParams',
    component: () => import('../views/AIParams.vue')
  },
  {
    path: '/occupancy-logs',
    name: 'OccupancyLogs',
    component: () => import('../views/OccupancyLogs.vue')
  },
  {
    path: '/network',
    name: 'Network',
    component: () => import('../views/Network.vue')
  },
  {
    path: '/local-model',
    name: 'LocalModel',
    component: () => import('../views/LocalModel.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// Navigation Guard
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  if (to.name !== 'Login' && !token) {
    next({ name: 'Login' })
  } else if (to.name === 'Login' && token) {
    next({ name: 'Dashboard' })
  } else {
    next()
  }
})

export default router
