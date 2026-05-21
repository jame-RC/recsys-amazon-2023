import re

ZERO_SHOT_SYSTEM = """You are a product recommendation expert. Given a user's purchase history, recommend products they might be interested in."""

ZERO_SHOT_RANKING = """Given a user's purchase history below, rank the following candidate products by how likely the user would want to buy them next.

User's purchase history (most recent last):
{history}

Candidate products to rank:
{candidates}

CRITICAL: Output ONLY the raw list of candidate IDs (integers, one per line), starting with your top recommendation. Do NOT include any conversational filler, introductory remarks, markdown code blocks, explanation, or any other text. Your entire response must consist solely of numbers."""

FEW_SHOT_RANKING = """Given a user's purchase history, rank candidate products by likelihood of purchase.

Example:
History: [Product A: "Guitar Strings", Product B: "Guitar Pick Set"]
Candidates: [1: "Guitar Capo", 2: "Cooking Pan", 3: "Guitar Strap", 4: "Yoga Mat"]
Ranking:
1
3
4
2

Now rank these:
History: {history}
Candidates: {candidates}

CRITICAL: Output ONLY the raw list of candidate IDs (integers, one per line), starting with your top recommendation. Do NOT include any conversational filler, introductory remarks, markdown code blocks, explanation, or any other text. Your entire response must consist solely of numbers."""

RAG_SYSTEM = """You are a product recommendation assistant. Use the provided context about products to make informed recommendations. Consider product categories, features, and user preferences."""

RAG_PROMPT = """Based on the user's purchase history and the retrieved product information, recommend the next product.

User's purchase history:
{history}

Relevant product information:
{context}

Candidate products to rank:
{candidates}

CRITICAL: Output ONLY the raw list of candidate IDs (integers, one per line), starting with your top recommendation. Do NOT include any conversational filler, introductory remarks, markdown code blocks, explanation, or any other text. Your entire response must consist solely of numbers."""


def get_category_prompts(category: str, mode: str = "zeroshot"):
    category_zh = category.replace("_", " ")
    
    # Custom category examples for Few-Shot In-Context Learning
    few_shot_examples = {
        "Musical_Instruments": """Example:
History:
1. D'Addario EXL110 Nickel Wound Guitar Strings
2. Fender Classic Shell Thin Guitar Picks (12-pack)
Candidates to rank:
1: Snark SN5X Clip-On Guitar Tuner
2: Lodge Cast Iron Skillet (10.25-inch)
3: Ernie Ball Polypro Black Guitar Strap
4: Gaiam Essentials Yoga Mat
Ranking:
1
3
4
2""",
        "CDs_and_Vinyl": """Example:
History:
1. Abbey Road (Remastered) by The Beatles (CD)
2. Dark Side of the Moon by Pink Floyd (Vinyl)
Candidates to rank:
1: Led Zeppelin IV (Remastered) (CD)
2: Stanley Classic Vacuum Insulated Bottle (1.0qt)
3: Thriller by Michael Jackson (CD)
4: HP 63XL Black Ink Cartridge
Ranking:
1
3
4
2""",
        "Industrial_and_Scientific": """Example:
History:
1. Liquid Nails LN207 All Purpose Caulk Adhesive (Clear)
2. 3M Professional Safety Glasses (Clear Anti-Fog Lens)
Candidates to rank:
1: Mitutoyo 500-196-30 Advanced Onsite Caliper
2: Casio fx-300ES Plus Scientific Calculator
3: Gorilla Super Glue Gel (20g)
4: Dunlop Tortex Standard 0.88mm Guitar Picks
Ranking:
3
1
2
4"""
    }
    
    example = few_shot_examples.get(category, few_shot_examples["Industrial_and_Scientific"])
    
    # Category-specific expert system prompts with domain-expert priors
    system_prompts = {
        "Musical_Instruments": "You are a professional musical instrument and gear recommendation expert. You have deep knowledge of music gear, player setups, and accessory compatibility (guitars, keyboards, drums, audio interfaces). Given a user's purchase history, analyze their setup, musical interests, and needs, and rank the candidate products based on setup completeness, compatibility, utility, and progression.",
        "CDs_and_Vinyl": "You are an expert music curator, audiophile, and record store recommendations specialist. You possess deep knowledge of music genres (rock, pop, jazz, classical, electronic), artist discographies, album eras, and media formats (CDs, Vinyl). Analyze the user's music taste, preferred genres, and listening habits from their purchase history, and rank candidate albums by musical affinity, genre coherence, artistic relevance, and collector value.",
        "Industrial_and_Scientific": "You are a senior industrial engineer, scientific lab consultant, and supply chain specialist. You have extensive knowledge of professional tools, measurement instruments (calipers, micrometers), laboratory glassware, safety gear (glasses, respirators), and manufacturing adhesives/fasteners. Analyze the user's project requirements, professional tasks, and safety needs, and rank candidate supplies by engineering utility, technical compatibility, and safety standards."
    }
    
    system_prompt = system_prompts.get(category, f"You are a product recommendation expert specializing in the '{category_zh}' domain. Based on a user's purchase history, rank the candidate products they are most likely to buy next.")
    
    if mode == "zeroshot":
        user_prompt = f"""Given a user's purchase history below in the '{category_zh}' category, rank the following candidate products by how likely the user would want to buy them next.

User's purchase history (most recent last):
{{history}}

Candidate products to rank:
{{candidates}}

CRITICAL: Output ONLY the raw list of candidate IDs (integers, one per line), starting with your top recommendation. Do NOT include any conversational filler, introductory remarks, markdown code blocks (such as ```), explanations, or any other text. Your entire response must consist solely of numbers."""
    else:
        user_prompt = f"""Given a user's purchase history in the '{category_zh}' category, rank the candidate products by likelihood of purchase.

{example}

Now rank these:
History:
{{history}}
Candidates to rank:
{{candidates}}

CRITICAL: Output ONLY the raw list of candidate IDs (integers, one per line), starting with your top recommendation. Do NOT include any conversational filler, introductory remarks, markdown code blocks (such as ```), explanations, or any other text. Your entire response must consist solely of numbers.
Ranking:"""
        
    return system_prompt, user_prompt


def format_history(history_items: list) -> str:
    lines = []
    for i, item in enumerate(history_items):
        title = item.get("title", "Unknown")
        desc = item.get("description", "")[:100]
        lines.append(f"{i+1}. {title}" + (f" - {desc}" if desc else ""))
    return "\n".join(lines) if lines else "No history"


def format_candidates(candidates: list) -> str:
    lines = []
    for item in candidates:
        item_id = item.get("id", "?")
        title = item.get("title", "Unknown")
        lines.append(f"{item_id}: {title}")
    return "\n".join(lines)


def format_context(retrieved_items: list) -> str:
    lines = []
    for item in retrieved_items[:5]:
        title = item.get("title", "Unknown")
        desc = item.get("description", "")[:150]
        cat = ", ".join(item.get("categories", []))
        lines.append(f"- {title} [{cat}]: {desc}")
    return "\n".join(lines) if lines else "No context"
