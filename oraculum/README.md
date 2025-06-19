# Oraculum - Plataforma de Gestão de Conhecimento com RAG

Aplicação para processamento e consulta de documentos utilizando Retrieval-Augmented Generation (RAG) com FAISS e
LangChain.

## 🚀 Recursos Principais

- **Upload inteligente** de documentos (PDF, DOCX, TXT, MD)
- **Chatbot contextual** com memória de conversação
- **Armazenamento vetorial** FAISS com persistência
- **Auto-formatação** de documentos usando LLMs
- **Gestão visual** do banco vetorial

## ⚙️ Variáveis de Ambiente

Crie um arquivo `.env` na raiz com:

```env
OPENAI_API_KEY=sk-sua-chave-aqui
MODEL_CHAT=gpt-4o-mini
MODEL_EMBEDDING=text-embedding-3-small
MD_FOLDER=data/md
INDEX_DIR=data/faiss_index
```

## 📦 Pré-requisitos

Docker 20.10+

Docker Compose 2.20+

Chave API da OpenAI

## 🛠️ Instalação

```bash
# 1. Clonar repositório
git clone https://github.com/seu-usuario/oraculum.git
cd oraculum

# 2. Criar arquivo .env
cp .env.example .env

# 3. Construir e executar
docker-compose up --build -d
```

🖥️ Uso
Acesse no navegador: http://localhost:8501

Página Descrição
Upload Envio e conversão de documentos
Chat Interface de conversação com RAG
Gerenciar FAISS Controle do banco vetorial
🐳 Comandos Docker

```bash
# Iniciar serviços
docker-compose start

# Parar serviços
docker-compose stop

# Visualizar logs
docker-compose logs -f

# Reconstruir containers
docker-compose up --build --force-recreate

# Limpar ambiente
docker-compose down -v
```

📌 Notas Importantes
Os dados são persistidos nos diretórios:

data/md: Documentos processados

data/faiss_index: Índices vetoriais

Para desenvolvimento local:

```bash
# Acessar container
docker exec -it oraculum bash

# Instalar dependências manualmente
pip install -r requirements.txt
```

📄 Licença
MIT License - Consulte o arquivo LICENSE para detalhes.

Aviso: Necessário chave válida da OpenAI para funcionamento completo.

```
Este README fornece uma documentação completa e concisa para usuários e desenvolvedores, incluindo todos os pontos críticos do projeto.
```