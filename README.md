Start
=== 
Change directory to the root folder `Ollama_rag` and type `docker-compose up -d --build`  
  
**Make sure you are in the directory containing the `docker-compose.yml` file.**
```
Ollama_rag/ 
├── docker-compose.yml  
└── rag_server/  
    ├── docs/  
    │   └── data.pdf     
    ├── requirements.txt  
    ├── ingest.py  
    └── server.py
```
***
Setup
===
1. Open Browser and search `http://localhost:3000`
2. Go to `Settings` > `Connections`, then click `Add External Connection`  
   - Connection Type: `Externel`
   - URL: `http://rag-backend:8000/v1`
   - API_KEY: `sk-local`
   


