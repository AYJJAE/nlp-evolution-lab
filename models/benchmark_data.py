"""
NLP Evolution Lab - Benchmark & Training Data
Module 1: Evolution of NLP (BoW -> Word Embeddings -> Transformers)

Strict separation between Training Data (used to fit TF-IDF & Embedding baselines)
and Evaluation Data (held-out test benchmark).
"""

# Training dataset (Used strictly for fitting the TF-IDF and Embedding baselines)
TRAIN_DATA = [
    # Positive training examples
    ("This film was an absolute masterpiece with wonderful performances.", 1),
    ("I truly loved the cinematography and the soundtrack was stunning.", 1),
    ("The storytelling was heartwarming, engaging, and remarkably well executed.", 1),
    ("An outstanding experience that exceeded all of my highest expectations.", 1),
    ("Brilliant direction and memorable dialogue throughout the entire feature.", 1),
    ("A delightful and charming production that leaves you smiling.", 1),
    ("Exceptional cast with believable emotion and great pacing.", 1),
    ("The visual effects were breathtaking and the story kept me thrilled.", 1),
    ("I highly recommend this brilliant piece of creative filmmaking.", 1),
    ("Superb acting, poignant theme, and a deeply satisfying conclusion.", 1),
    ("A triumph of modern storytelling that deserves all critical acclaim.", 1),
    ("Very entertaining and captivating from the opening scene to the end.", 1),
    ("A wonderful achievement in cinema with unforgettable characters.", 1),
    ("Fresh, energetic, and genuinely uplifting throughout.", 1),
    ("Genuinely moving and beautifully crafted in every single frame.", 1),
    ("The documentary was insightful, informative, and beautifully directed.", 1),
    ("A heartwarming adventure that every family member will adore.", 1),
    ("One of the best cinematic experiences I have had in recent years.", 1),
    ("Terrific plot twists and fantastic lead performances make this a must-watch.", 1),
    ("Impeccable script, stellar acting, and gorgeous musical score.", 1),
    
    # Negative training examples
    ("This was a complete waste of time and an utterly boring film.", 0),
    ("The dialogue felt clunky, unnatural, and painfully forced.", 0),
    ("Horrible pacing and a completely nonsensical plot ruin the experience.", 0),
    ("I was deeply disappointed by the shallow characters and lazy directing.", 0),
    ("A disastrous failure that fails to deliver on every single promise.", 0),
    ("The acting was stiff and the storyline was dull and predictable.", 0),
    ("An uninspired mess of clichés with terrible pacing and awful sound.", 0),
    ("I regretted purchasing tickets for this tedious and agonizing production.", 0),
    ("Poorly written, poorly directed, and without any redeeming qualities.", 0),
    ("The script is full of plot holes and the ending was laughably bad.", 0),
    ("A dreadful experience that I would not recommend to anyone.", 0),
    ("Annoying characters, sloppy editing, and thoroughly unimpressive visuals.", 0),
    ("One of the worst movies I have witnessed in a very long time.", 0),
    ("Completely devoid of charm, humor, or genuine emotional resonance.", 0),
    ("Monotonous, painfully drawn out, and deeply frustrating to sit through.", 0),
    ("Terrible performance from the lead actor and zero chemistry on screen.", 0),
    ("A hollow and forgettable disaster that falls flat on its face.", 0),
    ("Boring, derivative, and completely lacking any originality.", 0),
    ("The movie drags endlessly without a coherent point or purpose.", 0),
    ("An offensive waste of talent and production budget.", 0),
]

# Held-Out Evaluation Dataset (Strictly unseen by training pipelines)
# Categorized to reveal the structural evolution of NLP:
# - Standard positive/negative
# - Negation challenges (where BoW typically fails)
# - Contrastive / Mixed clauses (where syntax & self-attention matter)
EVALUATION_DATA = [
    {
        "id": 1,
        "text": "The cinematography was breathtaking and every scene felt poetic.",
        "label": 1,
        "category": "Standard Positive",
        "rationale": "High density of positive vocabulary ('breathtaking', 'poetic'). Easily captured by all models."
    },
    {
        "id": 2,
        "text": "A thoroughly enjoyable journey filled with heart, humor, and joy.",
        "label": 1,
        "category": "Standard Positive",
        "rationale": "Explicit positive emotion words ('enjoyable', 'heart', 'humor', 'joy')."
    },
    {
        "id": 3,
        "text": "The lead performance was magnificent and deeply inspiring.",
        "label": 1,
        "category": "Standard Positive",
        "rationale": "Strong positive sentiment cues without structural ambiguity."
    },
    {
        "id": 4,
        "text": "A genuine cinematic gem that delivers an unforgettable emotional punch.",
        "label": 1,
        "category": "Standard Positive",
        "rationale": "Metaphorical positive phrasing ('cinematic gem', 'unforgettable')."
    },
    {
        "id": 5,
        "text": "An awful script combined with miserable acting makes this unwatchable.",
        "label": 0,
        "category": "Standard Negative",
        "rationale": "Explicit negative vocabulary ('awful', 'miserable', 'unwatchable')."
    },
    {
        "id": 6,
        "text": "The plot was utterly predictable and the jokes were completely dead.",
        "label": 0,
        "category": "Standard Negative",
        "rationale": "Direct negative descriptors ('predictable', 'dead')."
    },
    {
        "id": 7,
        "text": "A shallow, tiresome disaster that insults the audience's intelligence.",
        "label": 0,
        "category": "Standard Negative",
        "rationale": "Strong negative keywords ('shallow', 'tiresome', 'disaster')."
    },
    {
        "id": 8,
        "text": "I was bored to tears and couldn't wait for the credits to roll.",
        "label": 0,
        "category": "Standard Negative",
        "rationale": "Idiomatic negative sentiment denoting extreme boredom."
    },
    {
        "id": 9,
        "text": "This film was not bad at all, in fact I quite liked it.",
        "label": 1,
        "category": "Negation Challenge",
        "rationale": "Litotes negation ('not bad'). Bag of words detects 'bad' and often misclassifies as Negative; Transformers capture the negation scope."
    },
    {
        "id": 10,
        "text": "I did not hate this movie despite the harsh critical reviews.",
        "label": 1,
        "category": "Negation Challenge",
        "rationale": "'did not hate' inverts the negative keyword 'hate'. BoW baselines struggle without contextual negation tracking."
    },
    {
        "id": 11,
        "text": "It was by no means a disappointment, offering plenty of charm.",
        "label": 1,
        "category": "Negation Challenge",
        "rationale": "'by no means a disappointment' negates negative noun 'disappointment'."
    },
    {
        "id": 12,
        "text": "This was hardly the masterpiece that the trailers promised.",
        "label": 0,
        "category": "Negation Challenge",
        "rationale": "Subtle negation ('hardly the masterpiece'). Bag of words detects 'masterpiece' and falsely predicts Positive."
    },
    {
        "id": 13,
        "text": "I cannot recommend this film under any circumstances.",
        "label": 0,
        "category": "Negation Challenge",
        "rationale": "Negated recommendation ('cannot recommend'). BoW detects 'recommend' and risks misclassification."
    },
    {
        "id": 14,
        "text": "It fails to deliver any real excitement or emotional depth.",
        "label": 0,
        "category": "Negation Challenge",
        "rationale": "Negative operator 'fails to deliver' precedes positive concepts 'excitement' and 'depth'."
    },
    {
        "id": 15,
        "text": "The movie has stunning visuals, but the story is completely hollow.",
        "label": 0,
        "category": "Contrastive Clauses",
        "rationale": "Adversative 'but' flips emphasis to the second clause. Transformers use cross-token attention to weight discourse markers."
    },
    {
        "id": 16,
        "text": "Despite a rocky first act, the ending was remarkably triumphant.",
        "label": 1,
        "category": "Contrastive Clauses",
        "rationale": "Subordinating 'Despite' subordinates the negative clause to the positive final resolution."
    },
    {
        "id": 17,
        "text": "Great acting from the lead, yet the incoherent screenplay ruined it.",
        "label": 0,
        "category": "Contrastive Clauses",
        "rationale": "'yet the incoherent screenplay ruined it' dominates the overall sentiment."
    },
    {
        "id": 18,
        "text": "Not a great production, but it possessed an undeniable endearing charm.",
        "label": 1,
        "category": "Contrastive Clauses",
        "rationale": "Contrastive negation leading into positive sentiment."
    },
    {
        "id": 19,
        "text": "I expected a total disaster, but was pleasantly surprised.",
        "label": 1,
        "category": "Contrastive Clauses",
        "rationale": "'total disaster' is an expectation contrasted by 'pleasantly surprised'."
    },
    {
        "id": 20,
        "text": "The special effects were impressive, though everything else was mediocre.",
        "label": 0,
        "category": "Contrastive Clauses",
        "rationale": "Concessive 'though' restricts positive sentiment to VFX while qualifying the overall work as mediocre."
    }
]

