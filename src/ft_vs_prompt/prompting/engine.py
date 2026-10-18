"""Few-shot prompting engine — genuine prompt engineering effort, not a sandbagged baseline."""

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class PromptTemplate:
    task: str
    name: str
    system_prompt: str
    few_shot_examples: list[dict[str, str]]
    user_template: str


def build_classification_prompt(subtask: str) -> PromptTemplate:
    templates = {
        "sentiment": PromptTemplate(
            task="classification", name="sentiment",
            system_prompt="You are an expert text classifier. Analyze the sentiment of the input and respond with exactly one word: positive, negative, or neutral. Do not explain your answer.",
            few_shot_examples=[
                {"input": "The product arrived on time and works perfectly. I'm very satisfied with my purchase.", "output": "positive"},
                {"input": "Terrible experience. The item broke after two days and customer service was unhelpful.", "output": "negative"},
                {"input": "The package was delivered at 3pm as scheduled.", "output": "neutral"},
            ],
            user_template="Classify the sentiment: {input}",
        ),
        "topic": PromptTemplate(
            task="classification", name="topic",
            system_prompt="You are an expert document classifier. Classify the topic of the input into exactly one category: technology, finance, health, sports, or politics. Respond with only the category name.",
            few_shot_examples=[
                {"input": "The new GPU architecture delivers 40% better performance while consuming less power.", "output": "technology"},
                {"input": "The Federal Reserve announced it would maintain current interest rates amid inflation concerns.", "output": "finance"},
                {"input": "A new study shows that 30 minutes of daily walking reduces cardiovascular risk by 25%.", "output": "health"},
            ],
            user_template="Classify the topic: {input}",
        ),
        "intent": PromptTemplate(
            task="classification", name="intent",
            system_prompt="You are a customer service intent classifier. Classify the user's intent into exactly one category: purchase, support, complaint, inquiry, or cancel. Respond with only the intent label.",
            few_shot_examples=[
                {"input": "I'd like to buy the premium plan for my team of 10 people.", "output": "purchase"},
                {"input": "My account has been locked and I can't reset my password. Please help.", "output": "support"},
                {"input": "I've been overcharged for three months and nobody has resolved this.", "output": "complaint"},
            ],
            user_template="Classify the intent: {input}",
        ),
    }
    return templates.get(subtask, templates["sentiment"])


def build_extraction_prompt(subtask: str) -> PromptTemplate:
    templates = {
        "contact": PromptTemplate(
            task="extraction", name="contact_info",
            system_prompt="You are a precise data extraction system. Extract contact information from the input and output valid JSON with keys: name, email, phone. Use null for missing fields. Output ONLY the JSON, no explanation.",
            few_shot_examples=[
                {"input": "Hi, I'm Sarah Chen. You can reach me at sarah.chen@example.com or call (555) 123-4567.", "output": '{"name": "Sarah Chen", "email": "sarah.chen@example.com", "phone": "(555) 123-4567"}'},
                {"input": "Contact: John Smith, john@acme.com", "output": '{"name": "John Smith", "email": "john@acme.com", "phone": null}'},
            ],
            user_template="Extract contact information from: {input}",
        ),
        "order": PromptTemplate(
            task="extraction", name="order_details",
            system_prompt="You are a precise data extraction system. Extract order information from the input and output valid JSON with keys: order_id, product, quantity, price, date. Use null for missing fields. Output ONLY the JSON.",
            few_shot_examples=[
                {"input": "Order #ORD-12345: 3x Wireless Headphones at $79.99 each, placed on 2024-03-15.", "output": '{"order_id": "ORD-12345", "product": "Wireless Headphones", "quantity": 3, "price": 79.99, "date": "2024-03-15"}'},
                {"input": "Invoice for order INV-678: Mechanical Keyboard, $149.99.", "output": '{"order_id": "INV-678", "product": "Mechanical Keyboard", "quantity": null, "price": 149.99, "date": null}'},
            ],
            user_template="Extract order information from: {input}",
        ),
    }
    return templates.get(subtask, templates["contact"])


def build_style_prompt(subtask: str) -> PromptTemplate:
    templates = {
        "formal": PromptTemplate(
            task="style_generation", name="formal_rewrite",
            system_prompt="You are an expert business writer. Rewrite the input in a formal, professional tone suitable for a business report. Preserve all factual information. Do not add or remove facts.",
            few_shot_examples=[
                {"input": "Hey team, just wanted to let you know the project is done and it looks pretty good!", "output": "This memorandum confirms the completion of the project, which has met all specified requirements and quality standards."},
                {"input": "We messed up the numbers in the last report and need to fix them before the client sees.", "output": "A discrepancy has been identified in the previous report's figures, requiring correction prior to client review."},
            ],
            user_template="Rewrite in formal business tone: {input}",
        ),
        "concise": PromptTemplate(
            task="style_generation", name="concise_summary",
            system_prompt="You are an expert summarizer. Condense the input into a single sentence that captures the key point. Be concise but complete. Do not omit essential information.",
            few_shot_examples=[
                {"input": "After reviewing the quarterly results across all divisions, the board has determined that while revenue increased by 12% year-over-year, operating costs rose by 18%, resulting in a net margin decline of 2.3 percentage points.", "output": "Revenue grew 12% but costs rose faster at 18%, shrinking net margins by 2.3 points."},
                {"input": "The new software update includes several features: dark mode support, improved search functionality with fuzzy matching, export to PDF and CSV formats, and integration with third-party authentication providers including Google and Microsoft.", "output": "The update adds dark mode, fuzzy search, PDF/CSV export, and third-party auth integration."},
            ],
            user_template="Condense into one sentence: {input}",
        ),
        "technical": PromptTemplate(
            task="style_generation", name="technical_documentation",
            system_prompt="You are an expert technical writer. Rewrite the input in clear technical documentation style. Use precise terminology. Do not change the meaning.",
            few_shot_examples=[
                {"input": "So basically you click this button and it does the thing where it saves all your stuff to the cloud automatically.", "output": "Clicking the 'Sync' button initiates an automatic upload of local data to the cloud storage backend, ensuring all files are backed up to the configured remote endpoint."},
                {"input": "If the app crashes, just turn it off and on again and it should work.", "output": "In the event of an application failure, terminate the process and restart. The application will reinitialize its state from the persisted configuration."},
            ],
            user_template="Rewrite as technical documentation: {input}",
        ),
    }
    return templates.get(subtask, templates["formal"])


def format_prompt(template: PromptTemplate, input_text: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": template.system_prompt}]

    for ex in template.few_shot_examples:
        messages.append({"role": "user", "content": template.user_template.format(input=ex["input"])})
        messages.append({"role": "assistant", "content": ex["output"]})

    messages.append({"role": "user", "content": template.user_template.format(input=input_text)})

    return messages


def get_prompt_for_task(task: str, subtask: str = "") -> PromptTemplate:
    if task == "classification":
        return build_classification_prompt(subtask)
    elif task == "extraction":
        return build_extraction_prompt(subtask)
    elif task == "style_generation":
        return build_style_prompt(subtask)
    else:
        return build_classification_prompt("sentiment")


def get_zero_shot_prompt(task: str, subtask: str = "") -> PromptTemplate:
    template = get_prompt_for_task(task, subtask)
    template.few_shot_examples = []
    return template
