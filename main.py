import re
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- Config ----------
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_KEYWORDS = 10

# Enhanced technical keywords and their variations
TECH_SYNONYMS = {
    'javascript': ['js', 'javascript', 'ecmascript'],
    'typescript': ['ts', 'typescript'],
    'react': ['react', 'reactjs', 'react.js'],
    'node': ['node', 'nodejs', 'node.js'],
    'python': ['python', 'py'],
    'css': ['css', 'css3', 'cascading style sheets'],
    'html': ['html', 'html5'],
    'api': ['api', 'apis', 'rest api', 'restful api', 'web api'],
    'rest': ['rest', 'restful', 'rest api', 'rest apis'],
    'git': ['git', 'version control', 'source control'],
    'aws': ['aws', 'amazon web services', 'cloud'],
    'docker': ['docker', 'containerization'],
    'frontend': ['frontend', 'front-end', 'front end', 'client-side'],
    'backend': ['backend', 'back-end', 'back end', 'server-side'],
    'database': ['database', 'db', 'databases'],
    'sql': ['sql', 'mysql', 'postgresql', 'sqlite'],
    'agile': ['agile', 'scrum', 'kanban'],
    'testing': ['testing', 'unit testing', 'test', 'qa'],
    'ci/cd': ['ci/cd', 'continuous integration', 'continuous deployment', 'devops'],
    'redux': ['redux', 'state management'],
    'express': ['express', 'express.js', 'expressjs'],
    'next': ['next', 'next.js', 'nextjs'],
}

# Important technical skills that should be weighted higher
IMPORTANT_SKILLS = ['javascript', 'typescript', 'python', 'react',
                    'node', 'aws', 'docker', 'git', 'api', 'rest', 'css', 'html']

# ----------------------------
app = FastAPI(title="Resume-Job Matcher")  # Fixed syntax

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Fixed syntax
    allow_methods=["*"],  # Fixed syntax
    allow_headers=["*"],  # Fixed syntax
)

# Load model once at startup
model = SentenceTransformer(MODEL_NAME)
# warm-up
model.encode("warmup", convert_to_tensor=True)  # Fixed syntax


def preprocess_text(text: str) -> str:
    """Enhanced text preprocessing for better keyword extraction"""
    text = text.lower()

    # Handle common technical abbreviations and variations
    replacements = {
        r'\bnode\.?js\b': 'nodejs node',
        r'\breact\.?js\b': 'reactjs react',
        r'\bvue\.?js\b': 'vuejs vue',
        r'\bangular\.?js\b': 'angularjs angular',
        r'\bnext\.?js\b': 'nextjs next',
        r'\bexpress\.?js\b': 'expressjs express',
        r'\brest\s*api[s]?\b': 'rest api apis restful',
        r'\bweb\s*api[s]?\b': 'web api apis',
        r'\bfront[-\s]?end\b': 'frontend front-end',
        r'\bback[-\s]?end\b': 'backend back-end',
        r'\bfull[-\s]?stack\b': 'fullstack full-stack',
        r'\bci/cd\b': 'cicd ci/cd continuous integration deployment',
        r'\baws\b': 'aws amazon web services cloud',
        r'\bjavascript\b': 'javascript js',
        r'\btypescript\b': 'typescript ts',
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    return text


def normalize_keyword(keyword: str) -> str:
    """Normalize keywords to their canonical form"""
    keyword_lower = keyword.lower().strip()

    # Map variations to canonical forms
    canonical_mapping = {
        'nodejs': 'node.js',
        'node js': 'node.js',
        'reactjs': 'react',
        'react js': 'react',
        'javascript': 'javascript',
        'js': 'javascript',
        'typescript': 'typescript',
        'ts': 'typescript',
        'mongodb': 'mongodb',
        'mongo db': 'mongodb',
        'mongo': 'mongodb',
    }

    return canonical_mapping.get(keyword_lower, keyword_lower)


def deduplicate_keywords(keywords: List[str]) -> List[str]:
    """Remove duplicate and redundant keywords"""
    normalized = []
    seen_canonical = set()

    for keyword in keywords:
        canonical = normalize_keyword(keyword)
        if canonical not in seen_canonical and len(canonical) > 1:
            normalized.append(canonical)
            seen_canonical.add(canonical)

    # Remove substrings (e.g., if we have "react" and "react mongodb", keep only "react")
    filtered = []
    for i, kw1 in enumerate(normalized):
        is_substring = False
        for j, kw2 in enumerate(normalized):
            if i != j and kw1 in kw2 and len(kw1) < len(kw2):
                is_substring = True
                break
        if not is_substring:
            filtered.append(kw1)

    return filtered


def extract_keywords_from_short_text(text: str, max_keywords: int = 10) -> List[str]:
    """Better keyword extraction for short job descriptions - only extract what's explicitly mentioned"""
    text_lower = text.lower()

    found_keywords = []

    # Define exact patterns for what we're looking for - only explicit mentions
    keyword_patterns = [
        (r'\bnode\.?js\b|\bnode\b(?!\w)', 'node.js'),
        (r'\breact\.?js\b|\breact\b(?!\w)', 'react'),
        (r'\bmongodb\b|\bmongo\b(?!\w)', 'mongodb'),
        (r'\bjavascript\b', 'javascript'),  # Only if explicitly mentioned
        (r'\bjs\b(?!\w)', 'javascript'),   # Only standalone 'js'
        (r'\btypescript\b', 'typescript'),
        (r'\bpython\b', 'python'),
        (r'\bhtml\b', 'html'),
        (r'\bcss\b', 'css'),
        (r'\bvue\.?js\b|\bvue\b(?!\w)', 'vue'),
        (r'\bangular\b', 'angular'),
        (r'\bexpress\.?js\b|\bexpress\b(?!\w)', 'express'),
        (r'\bmysql\b', 'mysql'),
        (r'\bpostgresql\b', 'postgresql'),
        (r'\baws\b', 'aws'),
        (r'\bdocker\b', 'docker'),
        (r'\bgit\b(?!\w)', 'git'),
        (r'\bapi\b', 'api'),
        (r'\brest\b(?!\w)', 'rest'),
    ]

    # Check each pattern - only add if explicitly found in text
    for pattern, keyword in keyword_patterns:
        if re.search(pattern, text_lower):
            if keyword not in found_keywords:
                found_keywords.append(keyword)

    return found_keywords[:max_keywords]


def find_keyword_matches(keyword: str, text: str) -> bool:
    """Enhanced keyword matching with synonym support"""
    keyword_lower = keyword.lower().strip()
    text_lower = text.lower()

    # Direct match (exact keyword)
    if re.search(r'\b' + re.escape(keyword_lower) + r'\b', text_lower):
        return True

    # Check synonyms - look for the original keyword AND all its variations
    # First, find which tech category this keyword belongs to
    for base_term, synonyms in TECH_SYNONYMS.items():
        if keyword_lower == base_term or keyword_lower in synonyms:
            # Check if any synonym appears in the text
            for synonym in synonyms:
                if re.search(r'\b' + re.escape(synonym) + r'\b', text_lower):
                    return True
            break

    # Special handling for common variations
    if keyword_lower == 'node.js':
        # Check for node, nodejs, node.js
        if any(re.search(r'\b' + re.escape(variant) + r'\b', text_lower)
               for variant in ['node', 'nodejs', 'node.js']):
            return True
    elif keyword_lower == 'react':
        # Check for react, reactjs, react.js
        if any(re.search(r'\b' + re.escape(variant) + r'\b', text_lower)
               for variant in ['react', 'reactjs', 'react.js']):
            return True
    elif keyword_lower == 'javascript':
        # Check for javascript, js
        if any(re.search(r'\b' + re.escape(variant) + r'\b', text_lower)
               for variant in ['javascript', 'js']):
            return True
    elif keyword_lower == 'mongodb':
        # Check for mongodb, mongo
        if any(re.search(r'\b' + re.escape(variant) + r'\b', text_lower)
               for variant in ['mongodb', 'mongo']):
            return True

    return False


class MatchRequest(BaseModel):
    resume: str
    job: str
    top_k: Optional[int] = TOP_K_KEYWORDS


class MatchResponse(BaseModel):
    score: float
    percent: int
    top_keywords: List[str]
    matched_keywords: List[str]
    suggested_keywords: List[str]


@app.get("/")
async def root():
    return {"message": "Resume-Job Matcher API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)  # Fixed syntax
async def match(req: MatchRequest):  # Fixed syntax
    resume = req.resume.strip()  # Fixed syntax
    job = req.job.strip()  # Fixed syntax
    top_k = req.top_k or TOP_K_KEYWORDS  # Fixed syntax

    if not resume or not job:
        return MatchResponse(
            score=0.0, percent=0,  # Fixed syntax
            top_keywords=[], matched_keywords=[], suggested_keywords=[]  # Fixed syntax
        )

    # ---- extract job keywords (improved approach) ----
    # Use the improved keyword extraction for short texts
    top_keywords = extract_keywords_from_short_text(job, top_k)

    # ---- check which keywords appear in resume ----
    # Preprocess resume for better matching
    processed_resume = preprocess_text(resume)
    matched = []

    for kw in top_keywords:
        if find_keyword_matches(kw, processed_resume):
            matched.append(kw)

    # Deduplicate and clean up matched keywords
    matched = deduplicate_keywords(matched)

    # ---- embeddings similarity ----
    emb = model.encode([resume, job], convert_to_tensor=True)  # Fixed syntax
    base_sim = util.cos_sim(emb[0], emb[1]).item()  # -1..1

    # Calculate keyword-based boost
    total_keywords = len(top_keywords) if top_keywords else 1
    matched_count = len(matched) if matched else 0
    keyword_score = matched_count / total_keywords

    # Combine semantic similarity with keyword matching (weighted average)
    # 70% semantic, 30% keyword-based
    sim = (0.7 * base_sim) + (0.3 * keyword_score)
    sim = max(0, min(1, sim))  # Ensure it stays in valid range

    # Apply rescaling to boost scores - use a more gentle curve
    # This will make good matches feel more meaningful but not overly inflated
    if sim > 0:
        # Use a gentler exponential curve: score^0.85 (closer to 1 = less aggressive)
        rescaled_sim = pow(sim, 0.85)
        # Add a smaller linear boost
        rescaled_sim = rescaled_sim + (0.08 * sim)  # Reduced from 0.15 to 0.08
        # Ensure we don't exceed 1.0
        rescaled_sim = min(1.0, rescaled_sim)
        percent = max(0, min(100, int(round(rescaled_sim * 100))))
    else:
        percent = 0

    suggested = [kw for kw in top_keywords if kw not in matched]

    return MatchResponse(
        score=round(rescaled_sim if sim > 0 else 0,
                    3),  # Return rescaled score
        percent=percent,
        top_keywords=top_keywords,
        matched_keywords=matched,
        suggested_keywords=suggested,
    )
