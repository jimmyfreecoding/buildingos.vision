import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    redirect: '/cameras'
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
    path: '/network',
    component: () => import('../views/Network.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
