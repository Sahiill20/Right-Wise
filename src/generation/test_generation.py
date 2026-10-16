from llm_client import generate_answer


def main() -> None:

    prompt = """
You are NyayaSaathi, a consumer-rights information assistant for India.

Use only the context below.

LEGAL CONTEXT:
Section 35 says that a complaint relating to goods or services
may be filed with a District Commission by a consumer.

USER QUESTION:
How can a consumer file a complaint?

Answer in simple language and mention Section 35.
"""

    answer = generate_answer(prompt)

    print("\nGenerated answer:")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()