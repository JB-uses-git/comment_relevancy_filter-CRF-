"""
Reddit Comment Scraper using PRAW.
Scrapes real comments from gaming subreddits for relevancy evaluation.
Falls back to synthetic data if Reddit API credentials aren't configured.
Normalizes all datasets to required pipeline schema.
"""

from typing import Optional

import pandas as pd
from tqdm import tqdm

from config import (
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    SCRAPE_CONFIG, DATA_DIR, EVALUATION_QUESTIONS
)


def _reddit_available() -> bool:
    """Check if Reddit API credentials are configured."""
    return bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)


def scrape_reddit_comments(
    subreddits: Optional[list] = None,
    search_queries: Optional[list] = None,
    max_posts: int = 10,
    max_comments: int = 50,
    min_length: int = 15,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Scrape comments from Reddit using PRAW.

    Returns normalized DataFrame with required pipeline columns:
    [comment, upvotes, true_label, topic, ...metadata]
    """
    if not _reddit_available():
        print("⚠️  Reddit API credentials not set. Use environment variables:")
        print("    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET")
        print("    Falling back to synthetic dataset.")
        return _load_synthetic_data()

    import praw

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    subreddits = subreddits or SCRAPE_CONFIG["subreddits"]
    search_queries = search_queries or SCRAPE_CONFIG["search_queries"]
    max_posts = max_posts or SCRAPE_CONFIG["max_posts_per_query"]
    max_comments = max_comments or SCRAPE_CONFIG["max_comments_per_post"]
    min_length = min_length or SCRAPE_CONFIG["min_comment_length"]

    all_comments = []
    seen_ids = set()

    for sub_name in subreddits:
        subreddit = reddit.subreddit(sub_name)
        for query in tqdm(search_queries, desc=f"r/{sub_name}"):
            try:
                posts = list(subreddit.search(query, limit=max_posts, sort="relevance"))
            except Exception as e:
                print(f"  ⚠️ Error searching r/{sub_name} for '{query}': {e}")
                continue

            for post in posts:
                try:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list()[:max_comments]:
                        if comment.id in seen_ids:
                            continue
                        body = comment.body.strip()
                        if len(body) < min_length or body == "[deleted]" or body == "[removed]":
                            continue

                        seen_ids.add(comment.id)
                        all_comments.append({
                            "comment": body,
                            "upvotes": comment.score,
                            "post_title": post.title,
                            "subreddit": sub_name,
                            "post_url": f"https://reddit.com{post.permalink}",
                            "comment_id": comment.id,
                        })
                except Exception as e:
                    continue

    df = pd.DataFrame(all_comments)

    if len(df) == 0:
        print("⚠️  No comments scraped. Falling back to synthetic data.")
        return _load_synthetic_data()

    # Deduplicate by comment text
    df = df.drop_duplicates(subset="comment").reset_index(drop=True)
    df = _normalize_dataset(df, source="reddit_scrape")

    print(f"✅ Scraped {len(df)} unique comments from {len(subreddits)} subreddit(s)")

    # Save raw data
    if output_file is None:
        output_file = DATA_DIR / "reddit_comments_raw.csv"
    df.to_csv(output_file, index=False)
    print(f"   Saved to {output_file}")

    return df


def _load_synthetic_data() -> pd.DataFrame:
    """Load the expanded synthetic dataset as fallback."""
    cached = DATA_DIR / "synthetic_comments.csv"
    if cached.exists():
        return _normalize_dataset(pd.read_csv(cached), source="synthetic_cache")

    df = _generate_synthetic_data()
    df.to_csv(cached, index=False)
    return _normalize_dataset(df, source="synthetic_generated")


def _generate_synthetic_data() -> pd.DataFrame:
    """
    Generate an expanded synthetic dataset with 300 comments across multiple gaming topics.
    This allows testing on multiple questions, not just Elden Beast.
    """
    comments_data = [
        # ═══════════════════════════════════════════════════════════════════
        # ELDEN BEAST RELEVANT (label=1, topic=elden_beast)
        # ═══════════════════════════════════════════════════════════════════
        ("Use a bleed build with Rivers of Blood katana, it melts the Elden Beast health quickly.", 4200, 1, "elden_beast"),
        ("Always clean the Erdtree golden sigils off the ground before attacking or you take huge damage.", 3800, 1, "elden_beast"),
        ("Stock up on Crimson Tears flasks and upgrade them to +12 before attempting this fight.", 3500, 1, "elden_beast"),
        ("The Elden Beast swims away constantly. Use spirit ashes to close the distance faster.", 3300, 1, "elden_beast"),
        ("Faith builds wreck this fight. Use Black Flame incantations, they deal percentage-based damage.", 3100, 1, "elden_beast"),
        ("Target the glowing golden weak spots on its body for extra damage windows.", 2900, 1, "elden_beast"),
        ("Use Mimic Tear ashes for Radagon phase, then Black Knife Tiche for Elden Beast, she avoids AOEs.", 2800, 1, "elden_beast"),
        ("Do not stand too close during the constellation attack, it tracks you. Sprint sideways.", 2700, 1, "elden_beast"),
        ("Bloodhound Step ash of war completely breaks this fight, use it to dodge the sword slam.", 2600, 1, "elden_beast"),
        ("If you are struggling, summon the Mimic Tear. It doubles your DPS and splits boss attention.", 2500, 1, "elden_beast"),
        ("Learn the golden sword slam combo: dodge into the boss, not away. Counter-intuitive but it works.", 2400, 1, "elden_beast"),
        ("Rivers of Blood +10 with Lord of Bloods Exultation talisman is the meta pick for this boss.", 2300, 1, "elden_beast"),
        ("Rogiers Rapture sorcery deals insane damage to the Elden Beast if you are a magic build.", 2200, 1, "elden_beast"),
        ("Godfrey Icon talisman boosts charged spells, great for landing big hits between Elden Beast attacks.", 2100, 1, "elden_beast"),
        ("Spam Blasphemous Blade skill, it heals on kill and the AoE hits the boss consistently.", 2050, 1, "elden_beast"),
        ("For melee builds: Giant Hunt ash of war launches you into the air and deals great poise damage.", 1900, 1, "elden_beast"),
        ("Bring at least 10 flasks going into Radagon and Elden Beast, it is a long back-to-back fight.", 1850, 1, "elden_beast"),
        ("The golden rings it spawns on the ground will explode, roll through them, do not try to run away.", 1800, 1, "elden_beast"),
        ("Grease your weapon with Holy-countering consumables if you are using a physical build.", 1750, 1, "elden_beast"),
        ("Use the Cerulean Hidden Tear in your Wondrous Flask to cast spells for free during Elden Beast.", 1700, 1, "elden_beast"),
        ("Vykes War Spear with bleed and madness is actually decent against Elden Beast, underrated pick.", 1650, 1, "elden_beast"),
        ("Radagon is weak to fire. Bring Fire Grease or fire spells for phase 1 to save flask charges.", 1600, 1, "elden_beast"),
        ("Your Physick flask combo should be Crimsonburst Crystal Tear and Opaline Hardtear for survivability.", 1580, 1, "elden_beast"),
        ("Elden Beast has Holy resistance. Do not use Holy weapons here, go Physical or Fire instead.", 1550, 1, "elden_beast"),
        ("Keep spacing awareness. The Elden Beast tail swipe comes out fast when you are underneath it.", 1480, 1, "elden_beast"),
        ("You can interrupt some Elden Beast spellcasting with a heavy weapon stagger if you are quick.", 1450, 1, "elden_beast"),
        ("Equip Erdtrees Favor talisman for extra HP and Stamina, this is a stamina-heavy fight.", 1420, 1, "elden_beast"),
        ("Try using Swarm of Flies incantation to stack bleed from range when the boss swims away.", 1400, 1, "elden_beast"),
        ("Power stance two Reduvia daggers and you can proc bleed every 2 to 3 hits on the Elden Beast.", 1380, 1, "elden_beast"),
        ("Get your Vigor to at least 40 before this fight. One-shots are common if you are under 35.", 1350, 1, "elden_beast"),
        ("If you die to Radagon, try memorizing his combos first before worrying about Elden Beast.", 1320, 1, "elden_beast"),
        ("Elden Beast holy beam follows you, sprint horizontally the entire time to dodge it.", 1300, 1, "elden_beast"),
        ("Comet Azur with Terra Magica setup deals absurd burst, you can one-phase Elden Beast sometimes.", 1280, 1, "elden_beast"),
        ("Night Comet sorcery is great here since it is hard for the AI to dodge, cast during openings.", 1240, 1, "elden_beast"),
        ("Dodge the Elden Stars meteor shower by sprinting diagonally, never in a straight line.", 1220, 1, "elden_beast"),
        ("The Elden Beast gets more aggressive below 50 percent HP. Save a golden vow incantation for that.", 1180, 1, "elden_beast"),
        ("Turtle Talisman extends stamina recovery. Very useful when dodging the Elden Beast constantly.", 1160, 1, "elden_beast"),
        ("Do not fight near the water edges, the camera can get messed up and cause mistimed rolls.", 1140, 1, "elden_beast"),
        ("If you are below level 130, go grind Farum Azula for runes before attempting this fight again.", 1100, 1, "elden_beast"),
        ("Elden Beast weakness window is right after its big slam attacks, punish those hard.", 1080, 1, "elden_beast"),
        ("The Elden Stars projectiles can be dodged by rolling toward where they came from, not away.", 1060, 1, "elden_beast"),
        ("Bring Preserving Boluses if you accidentally proc Scarlet Rot from environmental hazards.", 1020, 1, "elden_beast"),
        ("Do not rush the Radagon phase, play safe, conserve flasks, Elden Beast is the harder fight.", 1000, 1, "elden_beast"),
        ("Dexterity builds: use Moonveil with its transient moonlight ash of war, incredible DPS.", 980, 1, "elden_beast"),
        ("You can cheese Elden Beast by staying directly under its belly, most attacks miss you there.", 960, 1, "elden_beast"),
        ("Ancestral Spirits Horn talisman restores FP on kill, great for spell users in this long fight.", 940, 1, "elden_beast"),
        ("Rykards Great Rune helps sustain HP during long boss fights if you have it equipped.", 920, 1, "elden_beast"),
        ("Bloodflame Blade incantation on any physical weapon adds bleed buildup, solid on melee builds.", 900, 1, "elden_beast"),
        ("Try jumping attacks during the Elden Beast recovery animations for poise break potential.", 880, 1, "elden_beast"),
        ("If you use spirit ashes, make sure they are at +10, they survive much longer in this fight.", 860, 1, "elden_beast"),

        # ═══════════════════════════════════════════════════════════════════
        # MALENIA RELEVANT (label=1, topic=malenia_strategy)
        # ═══════════════════════════════════════════════════════════════════
        ("Bloodhound Step is mandatory for dodging Waterfowl Dance. Do not try to outrun it.", 3900, 1, "malenia_strategy"),
        ("Malenia heals on every hit, even if you block. Two-hand your weapon and learn to dodge.", 3600, 1, "malenia_strategy"),
        ("Frost and bleed combo destroys Malenia. Use dual Cold Uchigatanas with Seppuku.", 3200, 1, "malenia_strategy"),
        ("In phase 2, Malenia gains a Scarlet Rot dive bomb. Sprint away the moment she flies up.", 2900, 1, "malenia_strategy"),
        ("Mimic Tear +10 is the best summon for Malenia. It splits aggro and matches your build.", 2700, 1, "malenia_strategy"),
        ("Stagger her with jump attacks. She has surprisingly low poise for how aggressive she is.", 2500, 1, "malenia_strategy"),
        ("Black Knife Tiche ash is incredible here. Tiche dodges Waterfowl and deals percentage HP damage.", 2300, 1, "malenia_strategy"),
        ("Bring Preserving Boluses for phase 2 Scarlet Rot. She applies it with the butterfly attacks.", 2100, 1, "malenia_strategy"),
        ("Malenia can be stun-locked with dual Colossal weapons if you time your jump attacks right.", 1900, 1, "malenia_strategy"),
        ("Flame of the Redmanes ash of war staggers her in 2-3 hits. Absolutely broken against her.", 1750, 1, "malenia_strategy"),
        ("Phase 1 tip: stay at mid range. She whiffs most combos and you get free running R2 punishes.", 1600, 1, "malenia_strategy"),
        ("Rivers of Blood works but she dodges the initial slash sometimes. Use it during her recovery.", 1450, 1, "malenia_strategy"),
        ("Waterfowl Dance has 3 flurries. First: run away. Second: dodge into her. Third: dodge again.", 1300, 1, "malenia_strategy"),
        ("She is weak to frost. Hoarfrost Stomp plus a bleed weapon means double status procs.", 1150, 1, "malenia_strategy"),
        ("Do NOT use greatshields. Her lifesteal works on blocked hits, you will never outdamage the healing.", 1000, 1, "malenia_strategy"),

        # ═══════════════════════════════════════════════════════════════════
        # BEST BUILD RELEVANT (label=1, topic=best_build)
        # ═══════════════════════════════════════════════════════════════════
        ("For a first playthrough, go Vagabond class and pump Vigor to 40 before anything else.", 4100, 1, "best_build"),
        ("Quality build (STR/DEX even) lets you try every weapon. Most beginner-friendly approach.", 3700, 1, "best_build"),
        ("Bleed builds are overpowered. Get Arcane to 45 and use Rivers of Blood or Mohgs Spear.", 3400, 1, "best_build"),
        ("Pure strength with a Greatsword or Grafted Blade is simple and effective for new players.", 3100, 1, "best_build"),
        ("Intelligence builds struggle early but become god-tier with Comet Azur and Moonveil late game.", 2800, 1, "best_build"),
        ("Faith builds are versatile. You get heals, damage spells, and buffs all in one build.", 2500, 1, "best_build"),
        ("Do not neglect Endurance. You need stamina for dodging and heavy armor. 25 minimum.", 2200, 1, "best_build"),
        ("Brass Shield from the soldiers at Gatefront is the best medium shield you will find for hours.", 1900, 1, "best_build"),
        ("Prisoner class gives you Intelligence plus Dex for a spellblade build. Very fun and strong.", 1700, 1, "best_build"),
        ("The Uchigatana from the Deathtouched Catacombs is free and carries you through most of the game.", 1500, 1, "best_build"),
        ("Confessor class is the best starting class overall. Good stats, has heal, starts with a shield.", 1300, 1, "best_build"),
        ("Level Mind to at least 20 if you plan on using any magic or ashes of war regularly.", 1100, 1, "best_build"),
        ("Golden Halberd from the Tree Sentinel is the strongest early weapon if you can beat him.", 950, 1, "best_build"),
        ("Dual wielding two of the same weapon type gives you power stance attacks. Massive DPS increase.", 800, 1, "best_build"),
        ("Radagons Soreseal gives you 20 free levels of stats. Best talisman in the game for the first half.", 650, 1, "best_build"),

        # ═══════════════════════════════════════════════════════════════════
        # IRRELEVANT / OFF-TOPIC (label=0, topic=irrelevant)
        # ═══════════════════════════════════════════════════════════════════
        ("Bro just git gud lmaooo", 9800, 0, "irrelevant"),
        ("This game is trash compared to Dark Souls 3. DS3 had better bosses.", 7500, 0, "irrelevant"),
        ("I gave up and watched the ending on YouTube. No regrets.", 6800, 0, "irrelevant"),
        ("Have you tried turning your PC off and on again? Works for everything.", 6200, 0, "irrelevant"),
        ("My cat walked across the keyboard and somehow beat Radagon. True story.", 5900, 0, "irrelevant"),
        ("Just hire a gamer to do it for you lol life is too short", 5600, 0, "irrelevant"),
        ("I paused the game to make a sandwich and came back to a win screen somehow", 5100, 0, "irrelevant"),
        ("Petition to make George R.R. Martin finish Winds of Winter before adding more Elden Ring DLC", 4900, 0, "irrelevant"),
        ("This is why I play Mario Kart instead", 4700, 0, "irrelevant"),
        ("Just wait for the mobile version with auto-battle feature", 4500, 0, "irrelevant"),
        ("Touch grass. Seriously.", 4300, 0, "irrelevant"),
        ("Skill issue.", 4200, 0, "irrelevant"),
        ("My grandma could beat this game blindfolded", 4000, 0, "irrelevant"),
        ("Has anyone tried the GTA VI beta yet? Way better than Elden Ring", 3700, 0, "irrelevant"),
        ("The real final boss was the friends we made along the way", 3600, 0, "irrelevant"),
        ("I just learned Minecraft has a new update and it looks insane", 3400, 0, "irrelevant"),
        ("FromSoftware should add an easy mode already", 3200, 0, "irrelevant"),
        ("Reminder that Cyberpunk 2077 was also broken at launch lol", 3000, 0, "irrelevant"),
        ("Why is nobody talking about how good Baldurs Gate 3 is right now", 2800, 0, "irrelevant"),
        ("Uninstalled this game three times. On my fourth reinstall. Send help.", 2700, 0, "irrelevant"),
        ("I bench pressed my controller trying to win this fight", 2600, 0, "irrelevant"),
        ("Reddit: where you ask for help and get 40 jokes instead of answers", 2500, 0, "irrelevant"),
        ("My dog is better at this game than me and he does not have thumbs", 2400, 0, "irrelevant"),
        ("Every Elden Ring post: git gud. Every response: git gudder.", 2300, 0, "irrelevant"),
        ("Did anyone catch the Super Bowl last night? Insane ending.", 2200, 0, "irrelevant"),
        ("The economy is too rough right now to afford gaming time anyway", 2100, 0, "irrelevant"),
        ("Hot take: Skyrim is still the greatest RPG ever made", 2050, 0, "irrelevant"),
        ("I once speedran brushing my teeth in under 30 seconds. Felt like beating Malenia.", 1950, 0, "irrelevant"),
        ("Elden Ring? More like Elden SUFFERING am I right gamers", 1900, 0, "irrelevant"),
        ("Game devs should be legally required to add checkpoints before every boss", 1800, 0, "irrelevant"),
        ("Found 20 dollars in my old jacket pocket today. Best loot drop of the week.", 1700, 0, "irrelevant"),
        ("I think about quitting Elden Ring and then I remember I paid full price", 1650, 0, "irrelevant"),
        ("The real question is whether pineapple belongs on pizza. Change my mind.", 1600, 0, "irrelevant"),
        ("Miyazaki just tweeted something cryptic again. New game confirmed?", 1580, 0, "irrelevant"),
        ("I had a dream I beat this boss. Woke up. Still dead on first attempt.", 1550, 0, "irrelevant"),
        ("Does anyone else eat entire bags of chips during boss fights?", 1500, 0, "irrelevant"),
        ("Honestly the real final boss is my internet connection", 1480, 0, "irrelevant"),
        ("My wife says I have to choose between Elden Ring and dinner. I chose poorly.", 1450, 0, "irrelevant"),
        ("If you rotate your controller 90 degrees you get the same result but angrier", 1400, 0, "irrelevant"),
        ("Breaking: Area Man Still Cannot Beat Elden Ring Final Boss, Considers Career Change", 1380, 0, "irrelevant"),
        ("I just watched someone beat it with a Dance Dance Revolution pad on Twitch", 1350, 0, "irrelevant"),
        ("Imagine paying 60 dollars to get destroyed by a glowing space whale for 200 hours", 1320, 0, "irrelevant"),
        ("My therapist says I need to stop letting fictional bosses affect my mood", 1300, 0, "irrelevant"),
        ("One day this will all be a funny memory. Today is not that day.", 1280, 0, "irrelevant"),
        ("I showed my mom the boss. She said just do not die. Thanks mom.", 1260, 0, "irrelevant"),
        ("This comment section is the real final boss", 1240, 0, "irrelevant"),
        ("Anyone else GPU get physically hot playing this game or just me", 1220, 0, "irrelevant"),
        ("Just remembered I have not eaten today. The Elden Beast wins again.", 1200, 0, "irrelevant"),
        ("Bro my roommate sneezed during the last phase and I died. He is evicted.", 1180, 0, "irrelevant"),
        ("Going to bed. Will try again at 3am when I am even less coordinated.", 1160, 0, "irrelevant"),
        ("I rage quit and made pasta. The pasta was excellent. 10 out of 10 would recommend.", 1140, 0, "irrelevant"),
        ("Tell me you have never touched grass without telling me you have never touched grass", 1120, 0, "irrelevant"),
        ("Sometimes I think game companies design bosses just to sell more controllers", 1100, 0, "irrelevant"),
        ("I paused to use the bathroom and came back to the death screen. Physics.", 1060, 0, "irrelevant"),
        ("Fellow sufferers: there is a support group meeting Tuesday. Bring snacks.", 1040, 0, "irrelevant"),
        ("My character looks cooler dying than most characters do winning", 1020, 0, "irrelevant"),
        ("Honestly respecting the Elden Beast at this point. It is just built different.", 1000, 0, "irrelevant"),
        ("At what point does losing to a boss become a personality trait", 980, 0, "irrelevant"),
        ("Plot twist: you ARE the final boss and never realized", 960, 0, "irrelevant"),
        ("New achievement unlocked: Talked to Reddit instead of practicing", 940, 0, "irrelevant"),
        ("The Elden Ring movie better be 3 hours of someone struggling with this boss", 920, 0, "irrelevant"),
        ("I beat it once. Cannot replicate. I think it was a government experiment.", 900, 0, "irrelevant"),
        ("Opinion: any boss you cannot beat in 5 tries is badly designed. Change my mind.", 880, 0, "irrelevant"),
        ("Fun fact: the Elden Beast is named after someones ex. Probably.", 860, 0, "irrelevant"),
        ("I play games to relax. Somehow ended up here.", 840, 0, "irrelevant"),
        ("Solidarity fist bump to every Tarnished still stuck on this boss", 820, 0, "irrelevant"),
        ("Turns out punching your monitor does not deal damage to the boss. Confirmed.", 800, 0, "irrelevant"),
        ("Season 3 of The Bear just dropped and you are all in here dying repeatedly", 780, 0, "irrelevant"),
        ("Objectively speaking, have you considered just not fighting the boss", 760, 0, "irrelevant"),
        ("My build is called No Plan Whatsoever and it is going terribly", 740, 0, "irrelevant"),
        ("Is it just me or does the Elden Beast look like a cosmic goldfish", 720, 0, "irrelevant"),
        ("Just started Elden Ring. Where do I find the jump button?", 700, 0, "irrelevant"),
        ("I bet the devs laugh at us struggling while eating fancy sushi in Tokyo", 680, 0, "irrelevant"),
        ("Day 47: Still trapped in the Elden Ring subreddit. Send runes.", 660, 0, "irrelevant"),
        ("The loading screen tips are more helpful than half these comments", 640, 0, "irrelevant"),
        ("I tried to parry the Elden Beast. It did not go well.", 620, 0, "irrelevant"),
        ("WiFi went down mid-fight. Universe confirmed against me.", 600, 0, "irrelevant"),
        ("My controller battery died at the exact moment I had the kill. Pain.", 580, 0, "irrelevant"),
        ("Someone mod this boss into Stardew Valley. I want revenge.", 560, 0, "irrelevant"),
        ("The Elden Ring wiki is written in a language I do not speak: Git-Gud-ese.", 540, 0, "irrelevant"),
    ]

    df = pd.DataFrame(
        comments_data,
        columns=["comment", "upvotes", "true_label", "topic"]
    )

    print(f"✅ Generated synthetic dataset: {len(df)} comments")
    print(f"   Relevant: {df['true_label'].sum()} | Irrelevant: {(df['true_label']==0).sum()}")
    print(f"   Topics: {df['topic'].value_counts().to_dict()}")

    return df


def _infer_topic_and_label(comment: str) -> tuple[str, int, str]:
    """
    Weakly infer (topic, true_label) from comment text.
    This is a fallback when human labels are missing in real Reddit data.
    """
    text = (comment or "").lower()
    if not text:
        return "irrelevant", 0, "weak"

    meme_markers = (
        "git gud", "lmao", "lol", "skill issue", "touch grass",
        "youtube", "trash", "meme", "joke"
    )
    if any(marker in text for marker in meme_markers):
        return "irrelevant", 0, "weak"

    best_topic = "irrelevant"
    best_hits = 0
    for q in EVALUATION_QUESTIONS:
        topic_id = q["id"]
        keywords = q.get("keywords", [])
        hits = sum(1 for kw in keywords if kw and kw.lower() in text)
        if hits > best_hits:
            best_topic = topic_id
            best_hits = hits

    strategy_markers = (
        "use ", "build", "weapon", "incantation", "summon", "dodge",
        "flask", "phase", "attack", "damage", "talisman", "ash of war"
    )
    has_strategy_language = any(marker in text for marker in strategy_markers)

    if best_hits >= 2 or (best_hits >= 1 and has_strategy_language):
        return best_topic, 1, "weak"
    return "irrelevant", 0, "weak"


def _normalize_dataset(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Normalize data into pipeline-required schema.
    Required columns: comment, upvotes, true_label, topic.
    """
    if df is None or len(df) == 0:
        raise ValueError(f"No rows available in dataset source '{source}'.")

    normalized = df.copy()

    # Normalize common column aliases
    if "comment" not in normalized.columns and "body" in normalized.columns:
        normalized["comment"] = normalized["body"]
    if "upvotes" not in normalized.columns and "score" in normalized.columns:
        normalized["upvotes"] = normalized["score"]

    if "comment" not in normalized.columns:
        raise ValueError(
            f"Dataset source '{source}' is missing required text column. "
            "Expected 'comment' (or alias 'body')."
        )

    normalized["comment"] = normalized["comment"].astype(str).str.strip()
    normalized = normalized[normalized["comment"] != ""].copy()

    if "upvotes" not in normalized.columns:
        normalized["upvotes"] = 0
    normalized["upvotes"] = pd.to_numeric(normalized["upvotes"], errors="coerce").fillna(0).astype(int)

    has_labels = "true_label" in normalized.columns and normalized["true_label"].notna().all()
    has_topic = "topic" in normalized.columns and normalized["topic"].notna().all()

    if has_labels and has_topic:
        normalized["true_label"] = pd.to_numeric(normalized["true_label"], errors="coerce").fillna(0).astype(int)
        normalized["topic"] = normalized["topic"].astype(str).str.strip().replace("", "irrelevant")
        normalized["label_source"] = normalized.get("label_source", "human_or_existing")
    else:
        print(
            "⚠️  Real data has no complete human labels (true_label/topic). "
            "Using weak heuristic labels for pipeline compatibility."
        )
        inferred = normalized["comment"].apply(_infer_topic_and_label)
        normalized["topic"] = inferred.map(lambda x: x[0])
        normalized["true_label"] = inferred.map(lambda x: x[1]).astype(int)
        normalized["label_source"] = inferred.map(lambda x: x[2])

    normalized = normalized.drop_duplicates(subset="comment").reset_index(drop=True)
    return normalized


def load_or_scrape_data(force_scrape: bool = False) -> pd.DataFrame:
    """
    Main entry point: load cached data or scrape fresh from Reddit.
    Falls back to synthetic data if Reddit isn't available.
    """
    cached_labeled = DATA_DIR / "reddit_comments_labeled.csv"
    cached_reddit = DATA_DIR / "reddit_comments_raw.csv"
    cached_synthetic = DATA_DIR / "synthetic_comments.csv"

    if not force_scrape and cached_labeled.exists():
        print(f"📂 Loading human-labeled Reddit data from {cached_labeled}")
        return _normalize_dataset(pd.read_csv(cached_labeled), source="reddit_labeled_cache")
    elif not force_scrape and cached_reddit.exists():
        print(f"📂 Loading cached Reddit data from {cached_reddit}")
        return _normalize_dataset(pd.read_csv(cached_reddit), source="reddit_raw_cache")
    elif not force_scrape and cached_synthetic.exists():
        print(f"📂 Loading cached synthetic data from {cached_synthetic}")
        return _normalize_dataset(pd.read_csv(cached_synthetic), source="synthetic_cache")

    if _reddit_available():
        return scrape_reddit_comments()
    else:
        return _load_synthetic_data()


if __name__ == "__main__":
    df = load_or_scrape_data(force_scrape=True)
    print(f"\nSample:\n{df.head(10).to_string()}")
    print(f"\nTotal: {len(df)} comments")
