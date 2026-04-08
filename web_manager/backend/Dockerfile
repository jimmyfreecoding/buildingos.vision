FROM node:18-alpine

WORKDIR /app

# Install git and docker-cli for OTA updates
RUN apk add --no-cache git docker-cli util-linux

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install --production

# Copy backend source code
COPY . .

EXPOSE 3000

CMD ["node", "server.js"]
