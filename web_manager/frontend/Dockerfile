# 阶段 1：构建 Vue 前端
FROM node:18-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# 阶段 2：通过一个小巧的 alpine 容器，将构建好的文件拷贝到共享卷中
FROM alpine:latest

# 接收构建产物
COPY --from=builder /app/dist /app/dist

# 当容器启动时，将 dist 目录下的文件复制到挂载的目标卷（ZLM 的 www 目录）
CMD cp -r /app/dist/* /target_www/ && echo "Frontend deployed successfully to ZLM."
