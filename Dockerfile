FROM node:20-alpine

WORKDIR /app

COPY package.json ./

# We only copy package.json for now since there's no package-lock.json yet
# In a real scenario, you'd run npm install first to generate the lock file.
RUN npm install

COPY . .

# Expose port for Vite development server
EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host"]
