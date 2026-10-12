import { createHash } from "node:crypto";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Pinecone } from "@pinecone-database/pinecone";
import { getEncoding } from "js-tiktoken";
import OpenAI from "openai";
import { PDFParse } from "pdf-parse";


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// PDFs are stored in backend/src/resources/
const KNOWLEDGE_DIR = path.resolve(__dirname, "resources");

const INDEX_NAME = process.env.PINECONE_INDEX_NAME ?? "shop-support";
const NAMESPACE = process.env.PINECONE_NAMESPACE ?? "policies";

const EMBEDDING_MODEL = "text-embedding-3-small";
const EMBEDDING_DIMENSIONS = 1024;
const CHAT_MODEL = "gpt-5.6-luna";

const CHUNK_SIZE = 300;
const TOP_K = 4;

function requireKey(name) {
  const value = process.env[name];

  if (!value) {
    throw new Error(`Missing environment variable: ${name}`);
  }

  return value;
}

const openai = new OpenAI({
  apiKey: requireKey("OPENAI_API_KEY"),
});

const pinecone = new Pinecone({
  apiKey: requireKey("PINECONE_API_KEY"),
});

const pineconeIndex = pinecone
  .index(INDEX_NAME)
  .namespace(NAMESPACE);

const tokenizer = getEncoding("cl100k_base");

// Split document text into chunks
function splitIntoTokenChunks(text, chunkSize = CHUNK_SIZE) {
  const tokens = tokenizer.encode(text);
  const chunks = [];

  for (let start = 0; start < tokens.length; start += chunkSize) {
    const chunk = tokenizer
      .decode(tokens.slice(start, start + chunkSize))
      .trim();

    if (chunk) {
      chunks.push(chunk);
    }
  }

  return chunks;
}

async function readKnowledgeBase() {
  const fileNames = (await readdir(KNOWLEDGE_DIR))
    .filter((name) => name.toLowerCase().endsWith(".pdf"))
    .sort();

  if (fileNames.length === 0) {
    throw new Error(`No PDF files found in ${KNOWLEDGE_DIR}`);
  }

  const chunks = [];

//   for (const fileName of fileNames) {
//     const filePath = path.join(KNOWLEDGE_DIR, fileName);
//     const buffer = await readFile(filePath);
//     const parsed = await pdf(buffer);

//     splitIntoTokenChunks(parsed.text).forEach((text, chunkNumber) => {
//       chunks.push({
//         text,
//         metadata: {
//           source: fileName,
//           chunk: chunkNumber,
//         },
//       });
//     });
//   }

    
for (const fileName of fileNames) {
  let parser;

  try {
    console.log(`Reading PDF: ${fileName}`);

    const filePath = path.join(KNOWLEDGE_DIR, fileName);
    const buffer = await readFile(filePath);

    // NEW: Use the PDFParse class
    parser = new PDFParse({ data: buffer });

    // NEW: Extract text using getText()
    const parsed = await parser.getText();

    const fileChunks = splitIntoTokenChunks(parsed.text);

    fileChunks.forEach((text, chunkNumber) => {
      chunks.push({
        text,
        metadata: {
          source: fileName,
          chunk: chunkNumber,
        },
      });
    });

    console.log(
      `Successfully processed: ${fileName} (${fileChunks.length} chunks)`
    );
  } catch (error) {
    console.error(`Failed to process PDF: ${fileName}`);
    console.error(error);
    throw error;
  } finally {
    // NEW: Release parser resources
    if (parser) {
      await parser.destroy();
    }
  }
}


  return chunks;
}

// Generate a stable ID for each chunk
function stableId(chunk) {
  return createHash("sha256")
    .update(
      `${chunk.metadata.source}:${chunk.metadata.chunk}:${chunk.text}`
    )
    .digest("hex");
}

// Load PDF chunks into Pinecone
export async function loadKnowledgeBase() {
   const chunks = await readKnowledgeBase();

  if (chunks.length === 0) {
    throw new Error("No text chunks were extracted from the PDFs.");
  }

  console.log(`Total chunks extracted: ${chunks.length}`);

  const batchSize = 100;

  for (let start = 0; start < chunks.length; start += batchSize) {
    const batch = chunks.slice(start, start + batchSize);

    if (batch.length === 0) {
      continue;
    }

    console.log(
      `Generating embeddings for batch: ${start + 1} to ${start + batch.length}`
    );

    const embeddingResponse = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: batch.map((chunk) => chunk.text),
      dimensions: EMBEDDING_DIMENSIONS,
    });

    const records = batch.map((chunk, index) => ({
      id: stableId(chunk),
      values: embeddingResponse.data[index].embedding,
      metadata: {
        ...chunk.metadata,
        text: chunk.text,
      },
    }));

    if (records.length > 0) {
      await pineconeIndex.upsert({ records });
      console.log(`Upserted ${records.length} records into Pinecone.`);
    }
  }

  return chunks.length;
}

// Answer a user's question using retrieved document chunks
export async function answerUserQuery(question) {
  const queryEmbedding = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: question,
    dimensions: EMBEDDING_DIMENSIONS,
  });

  const searchResult = await pineconeIndex.query({
    vector: queryEmbedding.data[0].embedding,
    topK: TOP_K,
    includeMetadata: true,
  });

  const context = searchResult.matches
    .map((match) => match.metadata?.text)
    .filter(Boolean)
    .join("\n\n");

  const instructions = `You are an AI customer support assistant for our e-commerce company.

Answer the customer using ONLY the company information provided below.

If the answer is not available in the provided information, say:
"I don't have that information in the company documents."

COMPANY INFORMATION
${context}`;

  const response = await openai.responses.create({
    model: CHAT_MODEL,
    instructions,
    input: question,
  });

  return response.output_text;
}

