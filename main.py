import re
from collections import Counter
from typing import List


def parse_document(doc: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", doc).lower().split(" ")


def compute_collection_probabilities(all_documents: List[List[str]]) -> dict:
    collection_counts = Counter()
    total_words = 0
    for doc in all_documents:
        collection_counts.update(doc)
        total_words += len(doc)

    return {word: count / total_words for word, count in collection_counts.items()}


def compute_document_probability(
    query_terms: List[str],
    document: List[str],
    collection_probabilities: dict,
    lambda_param: float = 0.5,
) -> float:
    doc_counts = Counter(document)
    doc_length = len(document)

    log_probability = 0
    for term in query_terms:
        doc_prob = doc_counts[term] / doc_length if doc_length > 0 else 0
        collection_p = collection_probabilities.get(term, 0)

        smoothed_prob = lambda_param * doc_prob + (1 - lambda_param) * collection_p
        if smoothed_prob > 0:
            log_probability += smoothed_prob

    return log_probability


def get_input() -> tuple[List[List[str]], List[str]]:
    n = int(input().strip())
    documents = []
    for _ in range(n):
        doc = parse_document(input().strip())
        documents.append(doc)
    query = parse_document(input().strip())

    return documents, query


def compute_document_scores(
    documents: List[List[str]], query: List[str], collection_probabilities: dict
) -> List[tuple[int, float]]:
    doc_scores = []
    for i, doc in enumerate(documents):
        score = compute_document_probability(query, doc, collection_probabilities)
        doc_scores.append((i, score))

    return doc_scores


def main():
    documents, query = get_input()

    collection_probs = compute_collection_probabilities(documents)

    doc_scores = compute_document_scores(documents, query, collection_probs)

    sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    result = [i for i, _ in sorted_docs]

    print(result)


if __name__ == "__main__":
    main()
