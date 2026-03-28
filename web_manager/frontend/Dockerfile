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

# 当容器启动时，先清空目标卷的旧文件，再将 dist 目录下的文件复制到挂载的目标卷
CMD rm -rf /target_www/* && cp -r /app/dist/* /target_www/ && echo "Frontend deployed successfully to Nginx volume."
