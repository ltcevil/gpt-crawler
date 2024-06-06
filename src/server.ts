import cors from "cors";
import { configDotenv } from "dotenv";
import express, { Express } from "express";
import { readFile } from "fs/promises";
import swaggerUi from "swagger-ui-express";
import { Config, configSchema } from "./config.js";
// @ts-ignore
import { AzureOpenAIEmbeddings } from "@langchain/azure-openai";
import { PathLike } from "fs";
import { DistanceStrategy, FAISS } from 'langchain-community/vectorstores/faiss';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { join } from 'path';
import swaggerDocument from "../swagger-output.json" assert { type: "json" };
import GPTCrawlerCore from "./core.js";

configDotenv();

const app: Express = express();
const port = Number(process.env.API_PORT) || 3000;
const hostname = process.env.API_HOST || "localhost";

app.use(cors());
app.use(express.json());
app.use("/api-docs", swaggerUi.serve, swaggerUi.setup(swaggerDocument));

// Define a POST route to accept config and run the crawler
app.post("/crawl", async (req, res) => {
  const config: Config = req.body;
  try {
    const validatedConfig = configSchema.parse(config);
    const crawler = new GPTCrawlerCore(validatedConfig);
    await crawler.crawl();
    const outputFileName: PathLike = await crawler.write();
    const outputFileContent = await readFile(outputFileName, "utf-8");
    res.contentType("application/json");
    return res.send(outputFileContent);
  } catch (error) {
    return res
      .status(500)
      .json({ message: "Error occurred during crawling", error });
  }
});

app.post("/generate-embeddings", async (req, res) => {
  try {
    const outputFileName: PathLike = req.body.outputFileName;
    const outputFileContent = await readFile(outputFileName, "utf-8");
    const data = JSON.parse(outputFileContent);

    // Inicializáld az Azure OpenAI embeddings modellt
    const embeddings = new AzureOpenAIEmbeddings({
      azureOpenAIEndpoint: process.env.AZURE_API_ENDPOINT,
      azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
      azureOpenAIApiInstanceName: process.env.AZURE_OPENAI_API_INSTANCE_NAME,
      azureOpenAIApiDeploymentName: process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME,
      azureOpenAIApiVersion: process.env.AZURE_OPENAI_API_VERSION,
    });

    // A beágyazások és a FAISS index generálása a core.ts-ben lévő logika alapján
    const docs = data.map((dict: any) => ({
      pageContent: dict.html,
      metadata: { title: dict.title, url: dict.url },
    }));

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 100,
    });
    const allSplits = await textSplitter.splitDocuments(docs);

    const currentDatetime = new Date().toISOString().replace(/[:.]/g, "-");
    const indexFile = join(__dirname, '..', 'faiss', `${currentDatetime}.faiss`);

    const db = await FAISS.fromDocuments(allSplits, embeddings, {
      normalizeL2: true,
      distanceStrategy: DistanceStrategy.COSINE,
    });
    await db.saveLocal(indexFile);

    res.status(200).json({ message: "Embeddings generated successfully" });
  } catch (error) {
    res.status(500).json({ message: "Error generating embeddings", error });
  }
});


app.listen(port, hostname, () => {
  console.log(`API server listening at http://${hostname}:${port}`);
});

export default app;