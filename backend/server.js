import "dotenv/config"
import app from "./src/app.js"
import { loadKnowledgeBase } from "./src/rag.js"

const PORT = process.env.PORT || 5000;

async function startServer() {
    try {
        console.log("Loading knowledge base...");

        const chunkCount = await loadKnowledgeBase();

        console.log(`Loaded ${chunkCount} knowledge chunks.`);

        app.listen(PORT, () => {
        console.log(`Server running on port ${PORT}`);
        });

    } catch (error) {
        console.error("Failed to start server:", error);
        process.exit(1);
    }
}

startServer();

