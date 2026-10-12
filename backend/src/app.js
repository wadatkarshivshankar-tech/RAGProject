import express from "express"
import { answerUserQuery } from "./rag.js";



const app=express();
app.use(express.json());

app.post("/api/chat", async (req, res) => {
    try {
        const { question } = req.body;

        if (typeof question !== "string" || !question.trim()) {
            return res.status(400).json({
                message: "Please provide a valid question"
            });
        }

        const answer = await answerUserQuery(question.trim());

        res.json({ answer });

    } catch (error) {
        console.error("Chat error:", error);
        res.status(500).json({
        message: "Failed to generate answer"
        });
    }
});

export default app;