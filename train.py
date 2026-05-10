import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import time
from dataclasses import dataclass, asdict

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

def verify_macos_env():
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal. Detected platform: {sys.platform}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available. Ensure you are running on Apple Silicon with a compatible PyTorch build.")
    print("Environment verified: macOS detected with Metal (MPS) hardware acceleration available.")
    print()

# Skip the macOS gate when running on a CUDA / Linux pod for the multi-seed
# Phase 1 sweep. The device-detection code below handles cuda/mps/cpu uniformly
# so nothing else needs to change. Set AUTORESEARCH_SKIP_MACOS_CHECK=1 on the pod.
if os.environ.get("AUTORESEARCH_SKIP_MACOS_CHECK") != "1":
    verify_macos_env()
else:
    print(f"Skipping macOS env check (platform: {sys.platform}). "
          f"CUDA available: {torch.cuda.is_available()}.")
    # prepare.py (read-only) re-runs verify_macos_env at import time.
    # Spoof sys.platform + torch.backends.mps so that gate passes too.
    sys.platform = "darwin"
    class _FakeMPSBackend:
        @staticmethod
        def is_available():
            return True
        @staticmethod
        def is_built():
            return True
    torch.backends.mps = _FakeMPSBackend()

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# Restore real platform so subsequent device-detection code does the right thing.
if os.environ.get("AUTORESEARCH_SKIP_MACOS_CHECK") == "1":
    import platform as _platform
    sys.platform = "linux" if _platform.system() == "Linux" else _platform.system().lower()

# ---------------------------------------------------------------------------
# Probes for circuit-emergence study (project #18). 10 narrow capabilities,
# each a list of (prefix, correct_continuation, distractor_continuation).
# Logged at PROBE_INTERVAL steps to probe_log.tsv during training.
#
# Phase 3: each probe expanded to ~80 items via templating. Lowers binomial
# noise floor from sqrt(0.5*0.5/20)=0.11 to ~0.056 at 80 items, so weak
# but consistent emergences (pronoun_gender, ~0.10 above chance) are no
# longer drowned by sampling noise.
# ---------------------------------------------------------------------------


def _pronoun_items(female, male, verbs):
    """Build pronoun_gender items: '<Name> <verb> <obj> and' -> she/he."""
    items = []
    n = max(len(female), len(male))
    for i in range(n):
        name_f = female[i % len(female)]
        name_m = male[i % len(male)]
        verb = verbs[i % len(verbs)]
        items.append((f"{name_f} {verb} and", " she", " he"))
        items.append((f"{name_m} {verb} and", " he", " she"))
    return items


def _reflexive_items(female, male, verbs):
    """<Name> <verb> -> herself/himself."""
    items = []
    n = max(len(female), len(male))
    for i in range(n):
        name_f = female[i % len(female)]
        name_m = male[i % len(male)]
        verb = verbs[i % len(verbs)]
        items.append((f"{name_f} {verb}", " herself", " himself"))
        items.append((f"{name_m} {verb}", " himself", " herself"))
    return items


def _past_tense_items(time_phrases, verbs_pairs, contexts):
    """'<time_phrase> <subject> <past_verb_1> and' -> past / base."""
    items = []
    for i, (past, base) in enumerate(verbs_pairs):
        tp = time_phrases[i % len(time_phrases)]
        ctx = contexts[i % len(contexts)]
        items.append((f"{tp} {ctx} and", f" {past}", f" {base}"))
    return items


_FEMALE_NAMES = [
    "Mary", "Sarah", "Emily", "Lisa", "Jennifer", "Jessica", "Amanda",
    "Karen", "Linda", "Patricia", "Anna", "Olivia", "Sophia", "Isabella",
    "Mia", "Charlotte", "Amelia", "Harper", "Evelyn", "Abigail",
    "Ella", "Madison", "Chloe", "Lily", "Aubrey", "Zoey", "Hannah",
    "Lillian", "Addison", "Eleanor", "Natalie", "Grace", "Layla",
    "Audrey", "Bella", "Stella", "Allison", "Lucy", "Samantha", "Anna",
]
_MALE_NAMES = [
    "John", "David", "Michael", "Robert", "William", "Thomas", "James",
    "Daniel", "Charles", "Richard", "Joseph", "Henry", "Christopher",
    "Matthew", "Andrew", "Joshua", "Anthony", "Mark", "Steven", "Paul",
    "Kevin", "Brian", "George", "Edward", "Ronald", "Timothy", "Jason",
    "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas", "Eric", "Stephen",
    "Jonathan", "Larry", "Justin", "Scott", "Brandon", "Frank",
]
_PRONOUN_VERBS = [
    "opened the door", "picked up the book", "walked into the room",
    "finished the work", "smiled at the camera", "grabbed the keys",
    "took a deep breath", "opened the letter", "looked at the clock",
    "started the engine", "answered the phone", "closed the window",
    "picked up the bag", "turned off the light", "stood up slowly",
    "sat down at the desk", "waved goodbye", "laughed at the joke",
    "opened the purse", "nodded firmly", "grabbed the cup",
    "set down the plate", "checked the time", "tied the shoes",
    "pulled on the coat", "stepped outside", "looked around",
    "shook the hand", "held the umbrella", "wrote a note",
    "climbed the stairs", "ate the apple", "drank the water",
    "found the wallet", "knocked on the door", "watched the movie",
    "read the message", "left the room", "switched off the radio",
    "rinsed the dish",
]
_REFLEXIVE_VERBS = [
    "hurt", "saw", "blamed", "cooked for", "taught", "looked at",
    "pushed", "told", "bought", "trained", "proved", "asked",
    "convinced", "reminded", "promised", "treated", "cut", "introduced",
    "defended", "served", "excused", "encouraged", "challenged",
    "trusted", "questioned", "calmed", "comforted", "warned",
    "freed", "helped", "called", "imagined", "kept", "found",
    "guarded", "listened to", "talked to", "checked", "offered",
    "wrapped",
]
_TIME_PHRASES_PAST = [
    "Yesterday", "Last week", "Last night", "This morning", "Earlier today",
    "Last summer", "Yesterday afternoon", "Last weekend", "Last month",
    "Yesterday morning", "Last year", "Earlier this week", "Last spring",
    "Last winter", "Last evening", "Yesterday evening", "Last fall",
    "Two days ago", "Last Friday", "Last Sunday",
]
_PAST_BASE_VERBS = [
    ("bought", "buy"), ("sat", "sit"), ("made", "make"), ("had", "have"),
    ("enjoyed", "enjoy"), ("found", "find"), ("visited", "visit"),
    ("went", "go"), ("organized", "organize"), ("woke", "wake"),
    ("learned", "learn"), ("ran", "run"), ("arrived", "arrive"),
    ("finished", "finish"), ("drove", "drive"), ("ate", "eat"),
    ("echoed", "echo"), ("opened", "open"), ("stayed", "stay"),
    ("called", "call"), ("watched", "watch"), ("listened", "listen"),
    ("walked", "walk"), ("returned", "return"), ("picked", "pick"),
    ("studied", "study"), ("cleaned", "clean"), ("painted", "paint"),
    ("baked", "bake"), ("checked", "check"), ("danced", "dance"),
    ("played", "play"), ("answered", "answer"), ("closed", "close"),
    ("waited", "wait"), ("repeated", "repeat"), ("started", "start"),
    ("worked", "work"), ("explained", "explain"), ("ordered", "order"),
]
_PAST_CONTEXTS = [
    "I went to the store", "she walked into the office",
    "he got out of bed", "they played in the park",
    "we watched a movie", "she opened the package",
    "I traveled abroad", "he finished work",
    "they cleaned the house", "she woke up early",
    "I started a new job", "the dog barked",
    "we moved to the city", "she took the train",
    "they painted the room", "he fixed the car",
    "I cooked dinner", "the bell rang",
    "she received a letter", "we visited grandma",
    "he called the doctor", "the teacher arrived",
    "I saw a friend", "she finished the book",
    "we drove to the lake", "they cancelled the trip",
    "I lost the keys", "the rain fell hard",
    "she left the office", "he locked the door",
    "the plane took off", "we ordered food",
    "she danced all night", "he repaired the bike",
    "I checked the mail", "they hosted a dinner",
    "the cat ran away", "we found the wallet",
    "she washed the car", "he rinsed the dishes",
]
PROBES = {
    "close_quote": [
        ('He said, "I will be there soon', '"', '.'),
        ('She whispered, "Trust me with this', '"', '.'),
        ('The note read, "Please call me back', '"', '.'),
        ('"I have something important to tell you', '"', '.'),
        ('"You should not have come here', '"', '.'),
        ('"Where did you put the keys', '"', '.'),
        ('"I am not feeling very well', '"', '.'),
        ('"Look at how big it is', '"', '.'),
        ('"We need to leave right now', '"', '.'),
        ('"Please be careful with that', '"', '.'),
        ('"That is the funniest thing ever', '"', '.'),
        ('"I cannot believe what happened', '"', '.'),
        ('"This is going to be amazing', '"', '.'),
        ('"You are absolutely right about it', '"', '.'),
        ('"I told you many times already', '"', '.'),
        ('"We have to talk about this', '"', '.'),
        ('"Please stop doing that to me', '"', '.'),
        ('"Just give me a few minutes', '"', '.'),
        ('"That is exactly what I meant', '"', '.'),
        ('"Thank you for being here today', '"', '.'),
    ],
    "end_of_sentence": [
        ('She walked to the store yesterday afternoon', '.', ','),
        ('The cat jumped onto the kitchen counter', '.', ','),
        ('He decided to buy a new car', '.', ','),
        ('They went hiking in the mountains today', '.', ','),
        ('The sun rose over the quiet city', '.', ','),
        ('I finished reading the book last night', '.', ','),
        ('We had a great time at dinner', '.', ','),
        ('The dog barked at every passing stranger', '.', ','),
        ('She painted the entire fence by herself', '.', ','),
        ('The teacher explained the lesson very clearly', '.', ','),
        ('My brother won the chess tournament again', '.', ','),
        ('The plane landed safely at the airport', '.', ','),
        ('He fixed the broken sink in minutes', '.', ','),
        ('The garden looked beautiful in the spring', '.', ','),
        ('She baked a delicious cake for everyone', '.', ','),
        ('The kids played in the park all morning', '.', ','),
        ('He answered every question on the test', '.', ','),
        ('We watched the fireworks from the rooftop', '.', ','),
        ('The river flowed gently through the valley', '.', ','),
        ('They moved into the new house last month', '.', ','),
    ],
    "subj_verb_agreement": [
        ('The cats in the garden', ' are', ' is'),
        ('The boys at the school', ' are', ' is'),
        ('The dogs on the street', ' are', ' is'),
        ('The students in the class', ' are', ' is'),
        ('The cars on the highway', ' are', ' is'),
        ('The birds in the tree', ' are', ' is'),
        ('The girl in the red dress', ' is', ' are'),
        ('The man at the door', ' is', ' are'),
        ('The dog in the yard', ' is', ' are'),
        ('The teacher with the books', ' is', ' are'),
        ('The child in the corner', ' is', ' are'),
        ('The car in the driveway', ' is', ' are'),
        ('The children at the table', ' are', ' is'),
        ('The workers in the office', ' are', ' is'),
        ('The flowers near the window', ' are', ' is'),
        ('The book on the shelf', ' is', ' are'),
        ('The phone on my desk', ' is', ' are'),
        ('The woman with the hat', ' is', ' are'),
        ('The horses in the field', ' are', ' is'),
        ('The boy with the toys', ' is', ' are'),
    ],
    "pronoun_gender": [
        ('Mary opened the door and', ' she', ' he'),
        ('John picked up the book and', ' he', ' she'),
        ('Sarah walked into the room and', ' she', ' he'),
        ('David finished his work and', ' he', ' she'),
        ('Emily smiled at the camera and', ' she', ' he'),
        ('Michael grabbed his keys and', ' he', ' she'),
        ('Lisa took a deep breath and', ' she', ' he'),
        ('Robert opened the letter and', ' he', ' she'),
        ('Jennifer looked at the clock and', ' she', ' he'),
        ('William started the engine and', ' he', ' she'),
        ('Jessica answered the phone and', ' she', ' he'),
        ('Thomas closed the window and', ' he', ' she'),
        ('Amanda picked up her bag and', ' she', ' he'),
        ('James turned off the light and', ' he', ' she'),
        ('Karen stood up slowly and', ' she', ' he'),
        ('Daniel sat down at the desk and', ' he', ' she'),
        ('Linda waved goodbye and', ' she', ' he'),
        ('Charles laughed at the joke and', ' he', ' she'),
        ('Patricia opened her purse and', ' she', ' he'),
        ('Richard nodded firmly and', ' he', ' she'),
    ],
    "determiner_a_an": [
        ('I would like to have an', ' apple', ' banana'),
        ('She bought an', ' orange', ' lemon'),
        ('He found an', ' egg', ' rock'),
        ('They saw an', ' elephant', ' tiger'),
        ('We need an', ' answer', ' solution'),
        ('There was an', ' island', ' beach'),
        ('I met an', ' artist', ' singer'),
        ('She read an', ' article', ' chapter'),
        ('He bought a', ' book', ' apple'),
        ('She wore a', ' hat', ' umbrella'),
        ('They have a', ' dog', ' elephant'),
        ('I want a', ' coffee', ' orange'),
        ('We need a', ' map', ' answer'),
        ('She found a', ' coin', ' egg'),
        ('He carried a', ' bag', ' umbrella'),
        ('They built a', ' house', ' island'),
        ('I saw a', ' bird', ' eagle'),
        ('She drew a', ' picture', ' image'),
        ('He wrote a', ' letter', ' essay'),
        ('We watched a', ' movie', ' opera'),
    ],
    "numeric_sequence": [
        ('one, two, three,', ' four', ' seven'),
        ('two, three, four,', ' five', ' nine'),
        ('three, four, five,', ' six', ' ten'),
        ('four, five, six,', ' seven', ' two'),
        ('five, six, seven,', ' eight', ' three'),
        ('six, seven, eight,', ' nine', ' four'),
        ('seven, eight, nine,', ' ten', ' two'),
        ('1, 2, 3,', ' 4', ' 9'),
        ('2, 3, 4,', ' 5', ' 8'),
        ('3, 4, 5,', ' 6', ' 1'),
        ('4, 5, 6,', ' 7', ' 2'),
        ('5, 6, 7,', ' 8', ' 3'),
        ('6, 7, 8,', ' 9', ' 4'),
        ('7, 8, 9,', ' 10', ' 5'),
        ('Monday, Tuesday, Wednesday,', ' Thursday', ' Sunday'),
        ('Tuesday, Wednesday, Thursday,', ' Friday', ' Monday'),
        ('Wednesday, Thursday, Friday,', ' Saturday', ' Tuesday'),
        ('January, February, March,', ' April', ' August'),
        ('February, March, April,', ' May', ' September'),
        ('March, April, May,', ' June', ' October'),
    ],
    "common_idiom": [
        ('peanut butter and', ' jelly', ' chair'),
        ('salt and', ' pepper', ' window'),
        ('bread and', ' butter', ' chair'),
        ('thunder and', ' lightning', ' carpet'),
        ('cats and', ' dogs', ' tables'),
        ('macaroni and', ' cheese', ' sky'),
        ('sticks and', ' stones', ' clouds'),
        ('hide and', ' seek', ' walk'),
        ('rock and', ' roll', ' tree'),
        ('back and', ' forth', ' window'),
        ('night and', ' day', ' fish'),
        ('black and', ' white', ' purple'),
        ('arts and', ' crafts', ' wheels'),
        ('over and', ' over', ' tomato'),
        ('Romeo and', ' Juliet', ' apple'),
        ('Adam and', ' Eve', ' garage'),
        ('here and', ' there', ' apple'),
        ('now and', ' then', ' apple'),
        ('first and', ' foremost', ' tomato'),
        ('time and', ' time', ' apple'),
    ],
    "reflexive_pronoun": [
        ('She hurt', ' herself', ' himself'),
        ('He hurt', ' himself', ' herself'),
        ('Mary saw', ' herself', ' himself'),
        ('John saw', ' himself', ' herself'),
        ('She blamed', ' herself', ' himself'),
        ('He blamed', ' himself', ' herself'),
        ('Lisa cooked for', ' herself', ' himself'),
        ('David cooked for', ' himself', ' herself'),
        ('Sarah taught', ' herself', ' himself'),
        ('Michael taught', ' himself', ' herself'),
        ('Emily looked at', ' herself', ' himself'),
        ('Robert looked at', ' himself', ' herself'),
        ('Karen pushed', ' herself', ' himself'),
        ('William pushed', ' himself', ' herself'),
        ('Linda told', ' herself', ' himself'),
        ('Charles told', ' himself', ' herself'),
        ('Patricia bought', ' herself', ' himself'),
        ('Richard bought', ' himself', ' herself'),
        ('Jessica trained', ' herself', ' himself'),
        ('Thomas trained', ' himself', ' herself'),
    ],
    "past_tense_consistency": [
        ('Yesterday I went to the store and', ' bought', ' buy'),
        ('Last week she walked into the office and', ' sat', ' sit'),
        ('This morning he got out of bed and', ' made', ' makes'),
        ('Yesterday they played in the park and', ' had', ' have'),
        ('Last night we watched a movie and', ' enjoyed', ' enjoy'),
        ('Earlier today she opened the package and', ' found', ' finds'),
        ('Last summer I traveled to Spain and', ' visited', ' visit'),
        ('Yesterday afternoon he finished work and', ' went', ' goes'),
        ('Last weekend they cleaned the house and', ' organized', ' organize'),
        ('This morning she woke up early and', ' made', ' makes'),
        ('Last month I started a new job and', ' learned', ' learn'),
        ('Yesterday morning the dog barked and', ' ran', ' runs'),
        ('Last year we moved to a new city and', ' bought', ' buy'),
        ('Yesterday she took the train and', ' arrived', ' arrives'),
        ('Last week they painted the room and', ' finished', ' finish'),
        ('Yesterday he fixed the car and', ' drove', ' drives'),
        ('Last night I cooked dinner and', ' ate', ' eats'),
        ('Earlier today the bell rang and', ' echoed', ' echoes'),
        ('Yesterday she received a letter and', ' opened', ' opens'),
        ('Last weekend we visited grandma and', ' stayed', ' stays'),
    ],
    "proper_noun_completion": [
        ('The capital of France is', ' Paris', ' apple'),
        ('The largest ocean is the', ' Pacific', ' kitchen'),
        ('The president lives in the White', ' House', ' chair'),
        ('She visited New', ' York', ' apple'),
        ('They flew to Los', ' Angeles', ' apple'),
        ('He grew up in San', ' Francisco', ' apple'),
        ('We climbed Mount', ' Everest', ' chair'),
        ('The Eiffel', ' Tower', ' chair'),
        ('Niagara', ' Falls', ' chair'),
        ('Hong', ' Kong', ' chair'),
        ('Buenos', ' Aires', ' chair'),
        ('Saudi', ' Arabia', ' chair'),
        ('North', ' Korea', ' apple'),
        ('South', ' Africa', ' apple'),
        ('United', ' States', ' apple'),
        ('Great', ' Britain', ' apple'),
        ('She studied at Harvard', ' University', ' chair'),
        ('They went to Disney', ' World', ' chair'),
        ('He worked at Microsoft', ' Corporation', ' apple'),
        ('The Sahara', ' Desert', ' chair'),
    ],
}


# Phase 3: extend each probe with templated items so n_items >= ~60-80
# per probe. Lowers the binomial std at chance from 0.11 (n=20) to ~0.06
# (n=80). The original 20 hand-written items per probe remain prepended
# so cross-phase comparisons still hit the same prefixes.

PROBES["pronoun_gender"].extend(
    _pronoun_items(_FEMALE_NAMES[20:], _MALE_NAMES[20:],
                   _PRONOUN_VERBS[20:])
)
PROBES["reflexive_pronoun"].extend(
    _reflexive_items(_FEMALE_NAMES[20:], _MALE_NAMES[20:],
                     _REFLEXIVE_VERBS[20:])
)
PROBES["past_tense_consistency"].extend(
    _past_tense_items(_TIME_PHRASES_PAST, _PAST_BASE_VERBS[20:],
                      _PAST_CONTEXTS[20:])
)

# determiner_a_an: more "an + vowel-noun" / "a + consonant-noun" pairs.
_DET_AN_VOWEL = [
    "apple", "orange", "egg", "elephant", "answer", "island", "artist",
    "article", "umbrella", "owl", "ant", "octopus", "uncle", "eagle",
    "engineer", "envelope", "actor", "ocean", "instrument", "exercise",
    "interview", "expert", "office", "image", "essay", "idea", "onion",
    "iceberg", "operation", "experiment", "earring", "address", "antenna",
]
_DET_A_CONS = [
    "book", "hat", "dog", "coffee", "map", "coin", "bag", "house",
    "bird", "picture", "letter", "movie", "table", "phone", "car",
    "key", "ball", "tree", "river", "shirt", "doctor", "teacher",
    "horse", "dollar", "knife", "song", "story", "brick", "ladder",
    "donkey", "pencil", "ticket", "candle",
]
for vw, cn in zip(_DET_AN_VOWEL, _DET_A_CONS):
    PROBES["determiner_a_an"].append((f"They saw an", f" {vw}", f" {cn}"))
    PROBES["determiner_a_an"].append((f"He held a", f" {cn}", f" {vw}"))

# subj_verb_agreement: more plural/singular subjects with attached PPs.
_SVA_PLURAL = [
    "boys", "girls", "men", "women", "cats", "dogs", "birds", "horses",
    "cars", "trucks", "books", "phones", "students", "teachers",
    "workers", "doctors", "neighbours", "kids", "leaves", "windows",
    "brothers", "sisters", "guests", "musicians", "drivers",
]
_SVA_SINGULAR = [
    "boy", "girl", "man", "woman", "cat", "dog", "bird", "horse",
    "car", "truck", "book", "phone", "student", "teacher",
    "worker", "doctor", "neighbour", "kid", "leaf", "window",
    "brother", "sister", "guest", "musician", "driver",
]
_SVA_PPS = [
    "in the garden", "at the school", "on the street", "in the class",
    "on the highway", "in the tree", "in the yard", "with the books",
    "in the corner", "in the driveway", "near the window",
    "on the shelf", "on my desk", "with the hat", "in the field",
    "with the toys", "by the lake", "near the door", "behind the fence",
    "above the stove", "below the bridge", "next to the sofa",
    "across the room", "outside the house", "after the storm",
]
for plural, pp in zip(_SVA_PLURAL, _SVA_PPS):
    PROBES["subj_verb_agreement"].append(
        (f"The {plural} {pp}", " are", " is"))
for singular, pp in zip(_SVA_SINGULAR, _SVA_PPS):
    PROBES["subj_verb_agreement"].append(
        (f"The {singular} {pp}", " is", " are"))

# numeric_sequence: longer sequences and more domains.
_NUMERIC_EXTRAS = [
    ('one, two, three, four,', ' five', ' nine'),
    ('two, three, four, five,', ' six', ' two'),
    ('three, four, five, six,', ' seven', ' two'),
    ('four, five, six, seven,', ' eight', ' three'),
    ('five, six, seven, eight,', ' nine', ' two'),
    ('six, seven, eight, nine,', ' ten', ' four'),
    ('1, 2, 3, 4,', ' 5', ' 9'),
    ('2, 3, 4, 5,', ' 6', ' 1'),
    ('3, 4, 5, 6,', ' 7', ' 2'),
    ('4, 5, 6, 7,', ' 8', ' 1'),
    ('5, 6, 7, 8,', ' 9', ' 2'),
    ('6, 7, 8, 9,', ' 10', ' 3'),
    ('Thursday, Friday, Saturday,', ' Sunday', ' Tuesday'),
    ('Friday, Saturday, Sunday,', ' Monday', ' Wednesday'),
    ('Saturday, Sunday, Monday,', ' Tuesday', ' Friday'),
    ('Sunday, Monday, Tuesday,', ' Wednesday', ' Saturday'),
    ('April, May, June,', ' July', ' October'),
    ('May, June, July,', ' August', ' November'),
    ('June, July, August,', ' September', ' February'),
    ('July, August, September,', ' October', ' January'),
    ('August, September, October,', ' November', ' April'),
    ('September, October, November,', ' December', ' May'),
    ('October, November, December,', ' January', ' June'),
    ('first, second, third,', ' fourth', ' tenth'),
    ('second, third, fourth,', ' fifth', ' ninth'),
    ('third, fourth, fifth,', ' sixth', ' second'),
    ('5, 10, 15,', ' 20', ' 8'),
    ('10, 20, 30,', ' 40', ' 70'),
    ('100, 200, 300,', ' 400', ' 900'),
    ('2, 4, 6,', ' 8', ' 5'),
    ('3, 6, 9,', ' 12', ' 8'),
    ('5, 10, 15, 20,', ' 25', ' 7'),
    ('1, 3, 5,', ' 7', ' 4'),
    ('2, 5, 8,', ' 11', ' 6'),
    ('a, b, c,', ' d', ' z'),
    ('b, c, d,', ' e', ' a'),
    ('c, d, e,', ' f', ' z'),
    ('d, e, f,', ' g', ' a'),
    ('e, f, g,', ' h', ' a'),
    ('f, g, h,', ' i', ' z'),
]
PROBES["numeric_sequence"].extend(_NUMERIC_EXTRAS)

# common_idiom: more idioms (binomial pairs).
_IDIOM_EXTRAS = [
    ('hot and', ' cold', ' apple'),
    ('up and', ' down', ' apple'),
    ('in and', ' out', ' apple'),
    ('on and', ' on', ' window'),
    ('off and', ' on', ' apple'),
    ('left and', ' right', ' apple'),
    ('high and', ' low', ' apple'),
    ('young and', ' old', ' apple'),
    ('rich and', ' poor', ' apple'),
    ('big and', ' small', ' apple'),
    ('fish and', ' chips', ' apple'),
    ('milk and', ' cookies', ' carpet'),
    ('hugs and', ' kisses', ' tomato'),
    ('lost and', ' found', ' apple'),
    ('safe and', ' sound', ' tomato'),
    ('quick and', ' easy', ' carpet'),
    ('soap and', ' water', ' tomato'),
    ('shoes and', ' socks', ' apple'),
    ('knife and', ' fork', ' tomato'),
    ('cup and', ' saucer', ' apple'),
    ('coffee and', ' tea', ' apple'),
    ('paper and', ' pen', ' tomato'),
    ('door and', ' window', ' apple'),
    ('bow and', ' arrow', ' apple'),
    ('lock and', ' key', ' tomato'),
    ('dollars and', ' cents', ' apple'),
    ('horse and', ' carriage', ' apple'),
    ('bread and', ' jam', ' tomato'),
    ('chips and', ' salsa', ' apple'),
    ('rain and', ' shine', ' apple'),
    ('wind and', ' rain', ' carpet'),
    ('ice and', ' snow', ' carpet'),
    ('arms and', ' legs', ' apple'),
    ('flesh and', ' blood', ' apple'),
    ('mom and', ' dad', ' apple'),
    ('boys and', ' girls', ' apple'),
    ('mind and', ' body', ' apple'),
    ('body and', ' soul', ' apple'),
    ('blood and', ' guts', ' apple'),
    ('day and', ' night', ' apple'),
]
PROBES["common_idiom"].extend(_IDIOM_EXTRAS)

# end_of_sentence: more single-sentence prefixes.
_EOS_EXTRAS = [
    ('She washed the dishes after dinner', '.', ','),
    ('He climbed up the tall ladder', '.', ','),
    ('The puppy chased its own tail', '.', ','),
    ('They listened to music in the car', '.', ','),
    ('She combed her hair before school', '.', ','),
    ('He fed the rabbits in the cage', '.', ','),
    ('The bus arrived at the station', '.', ','),
    ('We swept the leaves off the path', '.', ','),
    ('She poured the milk into the glass', '.', ','),
    ('He sharpened the pencil at his desk', '.', ','),
    ('The wind blew the curtains open', '.', ','),
    ('They cleaned the dishes after lunch', '.', ','),
    ('She wrote a note to her friend', '.', ','),
    ('He locked the bicycle to the post', '.', ','),
    ('The chef tasted the soup carefully', '.', ','),
    ('We hung the picture above the desk', '.', ','),
    ('She tied her shoes before the run', '.', ','),
    ('He whistled as he walked outside', '.', ','),
    ('The boy fed crumbs to the duck', '.', ','),
    ('They watered the plants every morning', '.', ','),
    ('She read a poem before bedtime', '.', ','),
    ('He polished the silver carefully', '.', ','),
    ('The horse galloped across the field', '.', ','),
    ('We labelled the boxes for moving', '.', ','),
    ('She raced her brother to the gate', '.', ','),
    ('He folded the napkin into a triangle', '.', ','),
    ('The fish swam around the tank', '.', ','),
    ('We shovelled snow from the driveway', '.', ','),
    ('She measured the flour in a cup', '.', ','),
    ('He brushed his teeth before bed', '.', ','),
    ('They locked the gate at sundown', '.', ','),
    ('She turned off the radio quietly', '.', ','),
    ('He whistled a happy little tune', '.', ','),
    ('The kettle whistled on the stove', '.', ','),
    ('She trimmed the hedge that morning', '.', ','),
    ('He stamped the letter and mailed it', '.', ','),
    ('The truck rumbled down the road', '.', ','),
    ('We took the boat out on the lake', '.', ','),
    ('She checked the answers carefully', '.', ','),
    ('He delivered the package by hand', '.', ','),
]
PROBES["end_of_sentence"].extend(_EOS_EXTRAS)

# close_quote: more quoted phrases.
_QUOTE_EXTRAS = [
    ('"Open the door right now', '"', '.'),
    ('"Please do not touch that', '"', '.'),
    ('"You forgot to call me back', '"', '.'),
    ('"Stay right there for a moment', '"', '.'),
    ('"I will be home for dinner', '"', '.'),
    ('"That is exactly the wrong answer', '"', '.'),
    ('"Hand me the screwdriver please', '"', '.'),
    ('"Watch out for the wet floor', '"', '.'),
    ('"This is the last warning today', '"', '.'),
    ('"I am driving home right now', '"', '.'),
    ('"Tell me everything you saw', '"', '.'),
    ('"Pass the bread over here please', '"', '.'),
    ('"You are going to love it', '"', '.'),
    ('"Don\'t forget the tickets again', '"', '.'),
    ('"That should be enough for tonight', '"', '.'),
    ('"I think we are all ready', '"', '.'),
    ('"Sit down and let me explain', '"', '.'),
    ('"The bus leaves in five minutes', '"', '.'),
    ('"It was a long drive home', '"', '.'),
    ('"Pick up your room before lunch', '"', '.'),
    ('"You are missing the whole point', '"', '.'),
    ('"Quiet down for just a second', '"', '.'),
    ('"Please listen to what I say', '"', '.'),
    ('"Take a deep breath and relax', '"', '.'),
    ('"That is what she told me', '"', '.'),
    ('"Stop teasing your little sister', '"', '.'),
    ('"You should ask her instead', '"', '.'),
    ('"I am not going to repeat it', '"', '.'),
    ('"Just leave it on the counter', '"', '.'),
    ('"That makes no sense at all', '"', '.'),
    ('"Try one more time and then stop', '"', '.'),
    ('"Carry the bags up the stairs', '"', '.'),
    ('"Hand the phone to your mother', '"', '.'),
    ('"He never showed up to dinner', '"', '.'),
    ('"This will be a great surprise', '"', '.'),
    ('"Keep walking until I tell you', '"', '.'),
    ('"You should have seen his face', '"', '.'),
    ('"That sounds like a wonderful plan', '"', '.'),
    ('"Tomorrow we are going to the zoo', '"', '.'),
    ('"There is no time left now', '"', '.'),
]
PROBES["close_quote"].extend(_QUOTE_EXTRAS)

# proper_noun_completion: more world-knowledge completions.
_PROPER_EXTRAS = [
    ('The capital of Japan is', ' Tokyo', ' apple'),
    ('The capital of Italy is', ' Rome', ' chair'),
    ('The capital of Germany is', ' Berlin', ' apple'),
    ('The capital of Russia is', ' Moscow', ' chair'),
    ('The capital of England is', ' London', ' apple'),
    ('The capital of Spain is', ' Madrid', ' chair'),
    ('The capital of China is', ' Beijing', ' apple'),
    ('The capital of Brazil is', ' Brasilia', ' chair'),
    ('The capital of Egypt is', ' Cairo', ' apple'),
    ('The capital of India is', ' Delhi', ' chair'),
    ('The capital of Canada is', ' Ottawa', ' apple'),
    ('The largest desert is the', ' Sahara', ' kitchen'),
    ('The longest river is the', ' Nile', ' chair'),
    ('The tallest mountain is', ' Everest', ' apple'),
    ('The deepest ocean trench is the', ' Mariana', ' apple'),
    ('She studied at Stanford', ' University', ' chair'),
    ('They studied at Oxford', ' University', ' apple'),
    ('He attended Yale', ' University', ' chair'),
    ('She attended Cambridge', ' University', ' apple'),
    ('He worked for Apple', ' Inc', ' chair'),
    ('She worked at Google', ' Inc', ' apple'),
    ('He flew to Rio de', ' Janeiro', ' chair'),
    ('They moved to Sao', ' Paulo', ' apple'),
    ('We climbed Mount', ' Kilimanjaro', ' chair'),
    ('He visited the Grand', ' Canyon', ' apple'),
    ('She walked along the Great', ' Wall', ' chair'),
    ('They saw the Statue of', ' Liberty', ' apple'),
    ('We crossed the Brooklyn', ' Bridge', ' chair'),
    ('They visited Buckingham', ' Palace', ' apple'),
    ('We toured the Louvre', ' Museum', ' chair'),
    ('They drove through Death', ' Valley', ' apple'),
    ('She climbed Mount', ' Fuji', ' chair'),
    ('He works for Goldman', ' Sachs', ' apple'),
    ('We watched the Super', ' Bowl', ' chair'),
    ('She studied at Princeton', ' University', ' apple'),
    ('He studied at MIT', ' Sloan', ' chair'),
    ('She visited Times', ' Square', ' apple'),
    ('He flew over the Pacific', ' Ocean', ' chair'),
    ('They walked in Central', ' Park', ' apple'),
    ('We visited the Empire State', ' Building', ' chair'),
]
PROBES["proper_noun_completion"].extend(_PROPER_EXTRAS)


# Phase 5: four additional capabilities for paper coverage. Designed so
# that each probe tests one syntactic/semantic phenomenon and uses
# distractors that are tokenizer-likely but linguistically wrong.

PROBES["relative_clause_agreement"] = [
    # Plural outer subject, singular distractor for the inner verb's
    # influence. The model must use the outer subject for agreement.
    ('The boys who saw the cat', ' are', ' is'),
    ('The girls who liked the song', ' are', ' is'),
    ('The teachers who watched the kids', ' are', ' is'),
    ('The dogs that followed the boy', ' are', ' is'),
    ('The cars that the men drove', ' are', ' is'),
    ('The horses that ran the race', ' are', ' is'),
    ('The children that the dog scared', ' are', ' is'),
    ('The students who finished the test', ' are', ' is'),
    ('The workers who built the house', ' are', ' is'),
    ('The neighbours who saw the truck', ' are', ' is'),
    ('The runners who passed the line', ' are', ' is'),
    ('The musicians who joined the band', ' are', ' is'),
    ('The cooks who prepared the meal', ' are', ' is'),
    ('The visitors who entered the room', ' are', ' is'),
    ('The painters who finished the wall', ' are', ' is'),
    # Singular outer subject; flipped truth.
    ('The boy who saw the cats', ' is', ' are'),
    ('The girl who liked the songs', ' is', ' are'),
    ('The teacher who watched the kids', ' is', ' are'),
    ('The dog that chased the boys', ' is', ' are'),
    ('The car that the men drove', ' is', ' are'),
    ('The horse that won the races', ' is', ' are'),
    ('The child that the dogs scared', ' is', ' are'),
    ('The student who finished the tests', ' is', ' are'),
    ('The worker who built the houses', ' is', ' are'),
    ('The neighbour who saw the trucks', ' is', ' are'),
    ('The runner who passed the lines', ' is', ' are'),
    ('The musician who joined the bands', ' is', ' are'),
    ('The cook who prepared the meals', ' is', ' are'),
    ('The visitor who entered the rooms', ' is', ' are'),
    ('The painter who finished the walls', ' is', ' are'),
    # More plural subjects with longer relative clauses.
    ('The kids who played in the yard', ' are', ' is'),
    ('The cats that slept on the bed', ' are', ' is'),
    ('The men who fixed the roof', ' are', ' is'),
    ('The women who baked the bread', ' are', ' is'),
    ('The drivers who waited at the light', ' are', ' is'),
    ('The dancers who joined the show', ' are', ' is'),
    ('The travelers who missed the train', ' are', ' is'),
    ('The friends who came to dinner', ' are', ' is'),
    ('The chefs who cooked the dish', ' are', ' is'),
    ('The actors who learned the lines', ' are', ' is'),
]

PROBES["comparative_than"] = [
    ('She is taller', ' than', ' for'),
    ('He is faster', ' than', ' the'),
    ('The cat is bigger', ' than', ' the'),
    ('My brother is older', ' than', ' for'),
    ('This pen is cheaper', ' than', ' for'),
    ('Her dress is prettier', ' than', ' for'),
    ('My dog is louder', ' than', ' for'),
    ('That tree is taller', ' than', ' for'),
    ('His car is faster', ' than', ' the'),
    ('The book is thicker', ' than', ' for'),
    ('The road is wider', ' than', ' for'),
    ('The lake is deeper', ' than', ' for'),
    ('The sun is brighter', ' than', ' for'),
    ('Tom is stronger', ' than', ' for'),
    ('My bag is heavier', ' than', ' for'),
    ('Her voice is softer', ' than', ' for'),
    ('The dog is quieter', ' than', ' the'),
    ('His shirt is whiter', ' than', ' for'),
    ('Her room is cleaner', ' than', ' the'),
    ('The path is shorter', ' than', ' the'),
    ('The hill is higher', ' than', ' for'),
    ('My phone is older', ' than', ' for'),
    ('The kitchen is warmer', ' than', ' for'),
    ('The water is colder', ' than', ' for'),
    ('The rope is longer', ' than', ' the'),
    ('Her hair is curlier', ' than', ' for'),
    ('His handwriting is neater', ' than', ' the'),
    ('The cake is sweeter', ' than', ' the'),
    ('Their house is larger', ' than', ' the'),
    ('The fire is hotter', ' than', ' for'),
    ('That puzzle is harder', ' than', ' the'),
    ('The chair is heavier', ' than', ' for'),
    ('Her smile is wider', ' than', ' the'),
    ('My garden is greener', ' than', ' for'),
    ('His joke is funnier', ' than', ' for'),
    ('The new car is faster', ' than', ' for'),
    ('Her work is better', ' than', ' for'),
    ('Their plan is simpler', ' than', ' for'),
    ('The exam is harder', ' than', ' for'),
    ('The light is brighter', ' than', ' for'),
]

PROBES["modal_continuation"] = [
    # After a future-time adverbial, the next verb should be a modal /
    # auxiliary (will, would, can, should). Distractor is a present-tense
    # form ('is', 'has') that's grammatical in other contexts.
    ('Tomorrow she', ' will', ' is'),
    ('Next week he', ' will', ' is'),
    ('Later today they', ' will', ' is'),
    ('Tonight we', ' will', ' is'),
    ('Soon I', ' will', ' is'),
    ('In the morning he', ' will', ' is'),
    ('Tomorrow afternoon she', ' will', ' is'),
    ('Tomorrow night they', ' will', ' is'),
    ('Next month we', ' will', ' is'),
    ('Next year I', ' will', ' is'),
    ('Tomorrow morning he', ' will', ' is'),
    ('In an hour she', ' will', ' is'),
    ('Later this evening they', ' will', ' is'),
    ('Tomorrow at noon we', ' will', ' is'),
    ('Next Friday he', ' will', ' is'),
    ('Next Monday she', ' will', ' is'),
    ('Next summer they', ' will', ' is'),
    ('In two days I', ' will', ' is'),
    ('On Saturday he', ' will', ' is'),
    ('In a few minutes we', ' will', ' is'),
    ('Tomorrow night I', ' will', ' is'),
    ('Tomorrow she', ' will', ' has'),
    ('Tonight they', ' will', ' has'),
    ('Tomorrow I', ' will', ' has'),
    ('Soon you', ' will', ' is'),
    ('Tomorrow evening he', ' will', ' is'),
    ('In ten minutes she', ' will', ' is'),
    ('Later we', ' will', ' is'),
    ('Tomorrow afternoon I', ' will', ' is'),
    ('Next weekend they', ' will', ' is'),
    ('Tonight I', ' will', ' is'),
    ('Soon she', ' will', ' is'),
    ('In the future they', ' will', ' is'),
    ('Tomorrow morning I', ' will', ' is'),
    ('Later today we', ' will', ' is'),
]

# Adjective-order probe is harder because both options are often
# grammatical in some context. Use clear size+colour pairs where the
# distractor is the wrong word class (a plural noun where an adjective
# is needed) so the model fails for syntactic, not stylistic, reasons.
PROBES["adjective_order"] = [
    ('She has a big', ' red', ' apples'),
    ('He bought a small', ' blue', ' cars'),
    ('They saw a tall', ' green', ' tomatoes'),
    ('I want a tiny', ' yellow', ' apples'),
    ('We found a huge', ' brown', ' apples'),
    ('She wore a long', ' black', ' shoes'),
    ('He carried a short', ' white', ' apples'),
    ('There was a thin', ' grey', ' apples'),
    ('She picked a small', ' pink', ' apples'),
    ('He drew a large', ' purple', ' apples'),
    ('They built a tall', ' red', ' apples'),
    ('She baked a tiny', ' blue', ' apples'),
    ('He bought a wide', ' green', ' apples'),
    ('We saw a giant', ' orange', ' apples'),
    ('She spotted a small', ' yellow', ' apples'),
    ('He found a huge', ' silver', ' apples'),
    ('They built a long', ' wooden', ' apples'),
    ('She wrote a short', ' simple', ' apples'),
    ('He held a small', ' shiny', ' apples'),
    ('We saw a big', ' grey', ' apples'),
    ('She wore a long', ' silk', ' apples'),
    ('They had a tiny', ' new', ' apples'),
    ('He owned a small', ' fast', ' apples'),
    ('She bought a large', ' soft', ' apples'),
    ('We sold a tall', ' wooden', ' apples'),
    ('He chose a short', ' steel', ' apples'),
    ('She picked a small', ' yellow', ' tomatoes'),
    ('He bought a thick', ' wool', ' tomatoes'),
    ('They sold a tall', ' glass', ' tomatoes'),
    ('We saw a tiny', ' blue', ' carrots'),
]


def prepare_probes(tokenizer):
    """Pre-tokenize all probe items once at startup."""
    bos = tokenizer.get_bos_token_id()
    prepared = {}
    for probe_name, items in PROBES.items():
        prepared_items = []
        for prefix, correct_str, distractor_str in items:
            prefix_ids = tokenizer.encode(prefix, prepend=bos)
            correct_ids = tokenizer.encode(correct_str)
            distractor_ids = tokenizer.encode(distractor_str)
            if len(correct_ids) == 0 or len(distractor_ids) == 0:
                continue
            prepared_items.append({
                'prefix_ids': prefix_ids,
                'correct_ids': correct_ids,
                'distractor_ids': distractor_ids,
            })
        prepared[probe_name] = prepared_items
    return prepared


@torch.no_grad()
def evaluate_probes(model, prepared_probes, device, max_len=48, chunk_size=1024):
    """
    Returns dict probe_name -> {'argmax_acc': float, 'logprob_diff': float}.
    Forced-decode comparison: for each (prefix, correct, distractor), sum log-probs
    of suffix tokens given prefix. argmax_acc = fraction where correct > distractor.
    logprob_diff = mean (correct_lp - distractor_lp), length-normalized.
    """
    was_training = model.training
    model.eval()

    all_seqs = []
    metadata = []
    for probe_name, items in prepared_probes.items():
        for item_idx, item in enumerate(items):
            prefix = item['prefix_ids']
            for side, suffix in [('correct', item['correct_ids']), ('distractor', item['distractor_ids'])]:
                full = prefix + suffix
                full = full[:max_len]
                all_seqs.append(full)
                metadata.append({
                    'probe': probe_name, 'idx': item_idx, 'side': side,
                    'plen': len(prefix), 'slen': len(suffix), 'seq': full,
                })

    if not all_seqs:
        if was_training:
            model.train()
        return {}

    max_seqlen = max(len(s) for s in all_seqs)
    padded = [s + [0] * (max_seqlen - len(s)) for s in all_seqs]

    scores = {}
    for i_chunk in range(0, len(padded), chunk_size):
        chunk_padded = padded[i_chunk:i_chunk + chunk_size]
        chunk_meta = metadata[i_chunk:i_chunk + chunk_size]
        batch = torch.tensor(chunk_padded, dtype=torch.long, device=device)
        logits = model(batch)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for i, m in enumerate(chunk_meta):
            plen, slen, seq = m['plen'], m['slen'], m['seq']
            lp_sum = 0.0
            n_counted = 0
            for k in range(slen):
                pos = plen - 1 + k
                if pos + 1 > max_seqlen or plen + k >= len(seq):
                    break
                tok = seq[plen + k]
                lp_sum += logprobs[i, pos, tok].item()
                n_counted += 1
            lp = lp_sum / max(n_counted, 1)
            key = (m['probe'], m['idx'])
            if key not in scores:
                scores[key] = {}
            scores[key][m['side']] = lp

    results = {}
    for probe_name, items in prepared_probes.items():
        n = len(items)
        if n == 0:
            continue
        correct = 0
        diff_sum = 0.0
        for item_idx in range(n):
            s = scores.get((probe_name, item_idx), {})
            sc = s.get('correct')
            sd = s.get('distractor')
            if sc is None or sd is None:
                continue
            if sc > sd:
                correct += 1
            diff_sum += sc - sd
        results[probe_name] = {
            'argmax_acc': correct / n,
            'logprob_diff': diff_sum / n,
        }

    if was_training:
        model.train()
    return results


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # PyTorch SDPA without FlashAttention 3
        # Expand heads for KV based on GQA
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        
        # Transpose to [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply mask for sliding window
        window = window_size[0]
        if window > 0 and window < T:
            # Mask out tokens outside the window
            mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            mask = mask.triu(diagonal=1 - window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    # Move scalars to correct device and dtype
    step_t = step_t.to(device=p.device, dtype=p.dtype)
    lr_t = lr_t.to(device=p.device, dtype=p.dtype)
    beta1_t = beta1_t.to(device=p.device, dtype=p.dtype)
    beta2_t = beta2_t.to(device=p.device, dtype=p.dtype)
    eps_t = eps_t.to(device=p.device, dtype=p.dtype)
    wd_t = wd_t.to(device=p.device, dtype=p.dtype)
    
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Move scalars to correct device and dtype
    momentum_t = momentum_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    lr_t = lr_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    wd_t = wd_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    beta2_t = beta2_t.to(device=stacked_params.device, dtype=stacked_params.dtype)

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    
    # Needs to match second_momentum_buffer.dtype for lerp_
    beta2_cast = beta2_t.to(second_momentum_buffer.dtype)
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2_cast)
    
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        
        # Compile conditionally
        compiler_kwargs = {"dynamic": False, "fullgraph": True}
        if device_type in ("cuda", "cpu"):
            self.adamw_step_fused = torch.compile(adamw_step_fused, **compiler_kwargs)
            self.muon_step_fused = torch.compile(muon_step_fused, **compiler_kwargs)
        else:
            self.adamw_step_fused = adamw_step_fused
            self.muon_step_fused = muon_step_fused

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            self.adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        self.muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "L"    # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**16 # ~65K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.3    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 4               # number of transformer layers
# Per-device batch size. Default 32 was tuned for MPS; CUDA pods can fit
# more (try 64 on RTX 4090 / 3090). Override via AUTORESEARCH_BATCH_SIZE.
DEVICE_BATCH_SIZE = int(os.environ.get("AUTORESEARCH_BATCH_SIZE", "32"))

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
# Seed is overridable via env var so multi-seed reproducibility runs (Phase 1)
# can be launched as `AUTORESEARCH_SEED=123 uv run train.py` without editing
# this file. Default 42 keeps single-seed runs unchanged.
SEED = int(os.environ.get("AUTORESEARCH_SEED", "42"))
print(f"Seed: {SEED}")
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.set_float32_matmul_precision("high")

# Detect device
device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_type)

# Autocast context
if device_type == "cuda":
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
elif device_type == "cpu":
    autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
else:
    import contextlib
    autocast_ctx = contextlib.nullcontext()

H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

# torch.compile is unstable on MPS, only use on CUDA
if device_type == "cuda":
    model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Probe setup: pre-tokenize and open log file
PROBE_INTERVAL = 5  # eval probes every N optimizer steps after burn-in
# Burn-in: probe every step for the first PROBE_BURNIN_STEPS so the
# fast-saturating probes (proper_noun_completion, determiner_a_an —
# both saturated by step 50) get fine-grained t_50 resolution. Without
# this, sparse 5-step sampling on a 0->1 transition that happens in
# 50 steps gives only ~10 datapoints in the rise; not enough for the
# sigmoid fit to pin t_50 to better than ~5 steps.
PROBE_BURNIN_STEPS = 50
prepared_probes = prepare_probes(tokenizer)
print(f"Probes: {len(prepared_probes)} categories, "
      f"{sum(len(v) for v in prepared_probes.values())} items total")
# Write directly to a seed-suffixed file so multi-seed runs in Phase 1 don't
# overwrite each other. analyze.py globs probe_log*.tsv and parses the seed
# from the filename, so this is the format it expects.
probe_log_path = f"probe_log_seed{SEED}.tsv"
probe_log_file = open(probe_log_path, "w")
probe_log_file.write("step\ttraining_seconds\tprobe_name\targmax_acc\tlogprob_diff\n")
probe_log_file.flush()


def log_probes(step, training_seconds, results):
    for name, metrics in results.items():
        probe_log_file.write(f"{step}\t{training_seconds:.2f}\t{name}\t{metrics['argmax_acc']:.6f}\t{metrics['logprob_diff']:.6f}\n")
    probe_log_file.flush()

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

def sync_device(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()

while True:
    sync_device(device_type)
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding
    if train_loss_f > 100:
        print("FAIL")
        exit(1)

    sync_device(device_type)
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Probe-emergence eval (excluded from training_seconds). During the
    # burn-in window we probe every step so the fast-emergence transition
    # (steps 0-50) is finely resolved. Outside burn-in we probe every
    # PROBE_INTERVAL steps to preserve the 5-min training budget.
    if step <= PROBE_BURNIN_STEPS or step % PROBE_INTERVAL == 0:
        with autocast_ctx:
            probe_results = evaluate_probes(model, prepared_probes, device)
        log_probes(step, total_training_time, probe_results)

    # Checkpoint saving for mechanistic analysis.
    # AUTORESEARCH_CHECKPOINT_STEPS is a comma-separated list of step numbers
    # (e.g. "0,100,200,500,1000,2000") at which to save model state_dict.
    # Files go to checkpoints/seed{SEED}_step{N}.pt. Only active if the env
    # var is set, so normal training runs are unaffected.
    _ckpt_steps_env = os.environ.get("AUTORESEARCH_CHECKPOINT_STEPS", "")
    if _ckpt_steps_env and step in {int(s) for s in _ckpt_steps_env.split(",") if s.strip()}:
        import pathlib
        ckpt_dir = pathlib.Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"seed{SEED}_step{step:05d}.pt"
        torch.save({"step": step, "model_state": model.state_dict(),
                    "config": config.__dict__}, ckpt_path)
        print(f"\nSaved checkpoint: {ckpt_path}")

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final probe eval snapshot
with autocast_ctx:
    final_probe_results = evaluate_probes(model, prepared_probes, device)
log_probes(step, total_training_time, final_probe_results)
probe_log_file.close()

# Always save a final checkpoint when checkpoint saving is enabled.
_ckpt_steps_env_final = os.environ.get("AUTORESEARCH_CHECKPOINT_STEPS", "")
if _ckpt_steps_env_final:
    import pathlib as _pathlib
    _ckpt_dir = _pathlib.Path("checkpoints")
    _ckpt_dir.mkdir(exist_ok=True)
    _final_ckpt = _ckpt_dir / f"seed{SEED}_step{step:05d}_final.pt"
    torch.save({"step": step, "model_state": model.state_dict(),
                "config": config.__dict__}, _final_ckpt)
    print(f"Saved final checkpoint: {_final_ckpt}")

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
if device_type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram_mb = 0.0

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"seed:             {SEED}")
print(f"probe_log_path:   {probe_log_path}")
print("--- final_probe_acc ---")
for name, metrics in final_probe_results.items():
    print(f"probe_{name:24s} argmax={metrics['argmax_acc']:.3f} lp_diff={metrics['logprob_diff']:+.3f}")
