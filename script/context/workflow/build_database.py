"""Build HotpotQA article database for local search."""

from datasets import load_dataset
from tqdm import tqdm

from camel.storages import QdrantStorage, VectorDBQuery, VectorRecord

from rosetta.context.workflow.embeddings import SGLangEmbedding

# Configuration
DB_PATH = "local/data/qdrant_hotpotqa"
COLLECTION_NAME = "hotpotqa_articles"
BATCH_SIZE = 1024


def main():
    # Setup embedding and storage
    embedding = SGLangEmbedding()
    storage = QdrantStorage(
        vector_dim=embedding.get_output_dim(),
        path=DB_PATH,
        collection_name=COLLECTION_NAME,
    )

    # Load HotpotQA validation set
    print("Loading HotpotQA dataset...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    # Extract unique articles
    print("Extracting articles...")
    seen = set()
    articles = []
    for item in dataset:
        for title, sentences in zip(
            item["context"]["title"], item["context"]["sentences"]
        ):
            if title not in seen:
                seen.add(title)
                content = " ".join(sentences)
                articles.append({"title": title, "content": content})

    print(f"Found {len(articles)} unique articles")

    # Index articles in batches
    print("Indexing articles...")
    for i in tqdm(range(0, len(articles), BATCH_SIZE)):
        batch = articles[i : i + BATCH_SIZE]
        texts = [f"{a['title']}\n{a['content']}" for a in batch]
        vectors = embedding.embed_list(texts)

        records = [
            VectorRecord(vector=vec, payload={"title": a["title"], "text": text})
            for vec, a, text in zip(vectors, batch, texts)
        ]
        storage.add(records)

    # Test search
    print("\n--- Test Search ---")
    query = "Scott Derrickson nationality"
    query_vec = embedding.embed_list([query])[0]
    results = storage.query(VectorDBQuery(query_vector=query_vec, top_k=3))

    for i, r in enumerate(results):
        print(f"\n[{i+1}] Score: {r.similarity:.3f}")
        print(f"    Title: {r.record.payload['title']}")
        print(f"    {r.record.payload['text'][:200]}...")

    print(f"\nDatabase built at: {DB_PATH}")
    print(f"Total articles indexed: {len(articles)}")


if __name__ == "__main__":
    main()

