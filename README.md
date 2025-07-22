# Recycling Sorter Project

This is the main repository for the Recycling Sorter project.  
For the cross-platform frontend (web and mobile), see the `app` directory.

---

## Backend & Database Setup

### **1. Start MySQL and Adminer**
We use Docker to run a MySQL database with persistent storage.

1. Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is running.
2. Start the services:
   ```bash
   docker compose up -d
3. To stop the services:
   ```bash
   docker compose down