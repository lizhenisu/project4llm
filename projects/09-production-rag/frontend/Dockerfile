FROM node:24-alpine AS build

WORKDIR /app
COPY projects/09-production-rag/frontend/package*.json ./
RUN npm ci
COPY projects/09-production-rag/frontend ./
RUN npm run build

FROM nginx:1.29-alpine
COPY projects/09-production-rag/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
