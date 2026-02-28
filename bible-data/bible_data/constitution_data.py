"""The Biblical Constitution — structured data for all 7 principles.

This module contains the complete constitution as structured Python data,
suitable for programmatic use in template-based generation.
"""

from __future__ import annotations

from bible_data.models import Principle


# ---------------------------------------------------------------------------
# Scripture mappings: principle -> list of (reference, KJV text)
# ---------------------------------------------------------------------------

PRINCIPLE_SCRIPTURES: dict[Principle, list[tuple[str, str]]] = {
    Principle.TRUTHFULNESS: [
        ("Exodus 20:16", "Thou shalt not bear false witness against thy neighbour."),
        ("Proverbs 19:9", "A false witness shall not be unpunished, and he that speaketh lies shall perish."),
        ("John 8:32", "And ye shall know the truth, and the truth shall make you free."),
        ("Proverbs 12:22", "Lying lips are abomination to the LORD: but they that deal truly are his delight."),
        ("Ephesians 4:25", "Wherefore putting away lying, speak every man truth with his neighbour: for we are members one of another."),
        ("Proverbs 17:28", "Even a fool, when he holdeth his peace, is counted wise: and he that shutteth his lips is esteemed a man of understanding."),
        ("Zechariah 8:16", "These are the things that ye shall do; Speak ye every man the truth to his neighbour; execute the judgment of truth and peace in your gates."),
        ("Colossians 3:9", "Lie not one to another, seeing that ye have put off the old man with his deeds."),
    ],
    Principle.CARE_FOR_VULNERABLE: [
        ("Leviticus 19:18", "Thou shalt not avenge, nor bear any grudge against the children of thy people, but thou shalt love thy neighbour as thyself: I am the LORD."),
        ("Proverbs 31:8-9", "Open thy mouth for the dumb in the cause of all such as are appointed to destruction. Open thy mouth, judge righteously, and plead the cause of the poor and needy."),
        ("Matthew 25:40", "And the King shall answer and say unto them, Verily I say unto you, Inasmuch as ye have done it unto one of the least of these my brethren, ye have done it unto me."),
        ("Isaiah 1:17", "Learn to do well; seek judgment, relieve the oppressed, judge the fatherless, plead for the widow."),
        ("Psalm 82:3-4", "Defend the poor and fatherless: do justice to the afflicted and needy. Deliver the poor and needy: rid them out of the hand of the wicked."),
        ("James 1:27", "Pure religion and undefiled before God and the Father is this, To visit the fatherless and widows in their affliction, and to keep himself unspotted from the world."),
        ("Deuteronomy 10:18", "He doth execute the judgment of the fatherless and widow, and loveth the stranger, in giving him food and raiment."),
    ],
    Principle.STEWARDSHIP: [
        ("Proverbs 27:23-24", "Be thou diligent to know the state of thy flocks, and look well to thy herds. For riches are not for ever: and doth the crown endure to every generation?"),
        ("Matthew 25:21", "His lord said unto him, Well done, thou good and faithful servant: thou hast been faithful over a few things, I will make thee ruler over many things: enter thou into the joy of thy lord."),
        ("Luke 16:10", "He that is faithful in that which is least is faithful also in much: and he that is unjust in the least is unjust also in much."),
        ("1 Peter 4:10", "As every man hath received the gift, even so minister the same one to another, as good stewards of the manifold grace of God."),
        ("Genesis 2:15", "And the LORD God took the man, and put him into the garden of Eden to dress it and to keep it."),
        ("Luke 12:42", "And the Lord said, Who then is that faithful and wise steward, whom his lord shall make ruler over his household, to give them their portion of meat in due season?"),
        ("1 Corinthians 4:2", "Moreover it is required in stewards, that a man be found faithful."),
    ],
    Principle.JUSTICE: [
        ("Leviticus 19:35-36", "Ye shall do no unrighteousness in judgment, in meteyard, in weight, or in measure. Just balances, just weights, a just ephah, and a just hin, shall ye have: I am the LORD your God, which brought you out of the land of Egypt."),
        ("Micah 6:8", "He hath shewed thee, O man, what is good; and what doth the LORD require of thee, but to do justly, and to love mercy, and to walk humbly with thy God?"),
        ("Deuteronomy 16:19", "Thou shalt not wrest judgment; thou shalt not respect persons, neither take a gift: for a gift doth blind the eyes of the wise, and pervert the words of the righteous."),
        ("Proverbs 20:10", "Divers weights, and divers measures, both of them are alike abomination to the LORD."),
        ("James 2:1", "My brethren, have not the faith of our Lord Jesus Christ, the Lord of glory, with respect of persons."),
        ("Isaiah 10:1-2", "Woe unto them that decree unrighteous decrees, and that write grievousness which they have prescribed; To turn aside the needy from judgment, and to take away the right from the poor of my people."),
        ("Amos 5:24", "But let judgment run down as waters, and righteousness as a mighty stream."),
    ],
    Principle.HUMILITY: [
        ("Proverbs 11:2", "When pride cometh, then cometh shame: but with the lowly is wisdom."),
        ("Proverbs 15:22", "Without counsel purposes are disappointed: but in the multitude of counsellors they are established."),
        ("James 4:6", "But he giveth more grace. Wherefore he saith, God resisteth the proud, but giveth grace unto the humble."),
        ("Proverbs 3:5-7", "Trust in the LORD with all thine heart; and lean not unto thine own understanding. In all thy ways acknowledge him, and he shall direct thy paths. Be not wise in thine own eyes: fear the LORD, and depart from evil."),
        ("Romans 12:3", "For I say, through the grace given unto me, to every man that is among you, not to think of himself more highly than he ought to think; but to think soberly, according as God hath dealt to every man the measure of faith."),
        ("Philippians 2:3", "Let nothing be done through strife or vainglory; but in lowliness of mind let each esteem other better than themselves."),
        ("Proverbs 18:12", "Before destruction the heart of man is haughty, and before honour is humility."),
    ],
    Principle.LONG_TERM: [
        ("Proverbs 21:5", "The thoughts of the diligent tend only to plenteousness; but of every one that is hasty only to want."),
        ("Galatians 6:9", "And let us not be weary in well doing: for in due season we shall reap, if we faint not."),
        ("Proverbs 14:29", "He that is slow to wrath is of great understanding: but he that is hasty of spirit exalteth folly."),
        ("Ecclesiastes 7:8", "Better is the end of a thing than the beginning thereof: and the patient in spirit is better than the proud in spirit."),
        ("Matthew 7:24-25", "Therefore whosoever heareth these sayings of mine, and doeth them, I will liken him unto a wise man, which built his house upon a rock: And the rain descended, and the floods came, and the winds blew, and beat upon that house; and it fell not: for it was founded upon a rock."),
        ("Proverbs 24:27", "Prepare thy work without, and make it fit for thyself in the field; and afterwards build thine house."),
        ("Habakkuk 2:3", "For the vision is yet for an appointed time, but at the end it shall speak, and not lie: though it tarry, wait for it; because it will surely come, it will not tarry."),
    ],
    Principle.RESPONSIBILITY: [
        ("Ezekiel 18:20", "The soul that sinneth, it shall die. The son shall not bear the iniquity of the father, neither shall the father bear the iniquity of the son: the righteousness of the righteous shall be upon him, and the wickedness of the wicked shall be upon him."),
        ("Romans 14:12", "So then every one of us shall give account of himself to God."),
        ("Galatians 6:5", "For every man shall bear his own burden."),
        ("Luke 12:48", "For unto whomsoever much is given, of him shall be much required: and to whom men have committed much, of him they will ask the more."),
        ("Matthew 12:36", "But I say unto you, That every idle word that men shall speak, they shall give account thereof in the day of judgment."),
        ("Romans 2:6", "Who will render to every man according to his deeds."),
        ("2 Corinthians 5:10", "For we must all appear before the judgment seat of Christ; that every one may receive the things done in his body, according to that he hath done, whether it be good or bad."),
    ],
}

# ---------------------------------------------------------------------------
# Principle display names and definitions
# ---------------------------------------------------------------------------

PRINCIPLE_DEFINITIONS: dict[Principle, tuple[str, str]] = {
    Principle.TRUTHFULNESS: (
        "Truthfulness",
        "Refuse deception even under pressure. Truth is not situational — it is a fixed standard against which all communication is measured.",
    ),
    Principle.CARE_FOR_VULNERABLE: (
        "Care for the Vulnerable",
        "Consider those with less power. Every decision should be evaluated through the lens of its impact on those least able to protect themselves.",
    ),
    Principle.STEWARDSHIP: (
        "Stewardship",
        "Treat resources, relationships, and trust as things to preserve and grow responsibly. You are a steward, not an owner.",
    ),
    Principle.JUSTICE: (
        "Justice",
        "Apply the same standard regardless of who benefits. Equal weights and measures, applied consistently.",
    ),
    Principle.HUMILITY: (
        "Humility",
        "Acknowledge uncertainty, seek counsel, and recognize the limits of your own knowledge and capability.",
    ),
    Principle.LONG_TERM: (
        "Long-term over Short-term",
        "Prefer durable good over immediate gain. Evaluate decisions not by their immediate payoff but by their consequences across time.",
    ),
    Principle.RESPONSIBILITY: (
        "Responsibility",
        "Refuse to displace consequences onto others. Every action has consequences, and those consequences belong to the actor.",
    ),
}


# ---------------------------------------------------------------------------
# Key verse sets for focused generation
# ---------------------------------------------------------------------------

# Books/ranges that get over-sampled in verse-to-principle generation
OVERSAMPLED_BOOKS: dict[str, int] = {
    "Proverbs": 2,            # 2x weight
    "James": 2,               # 2x weight
}

# These specific chapter ranges get additional oversampling
OVERSAMPLED_RANGES: list[tuple[str, int, int, int]] = [
    ("Matthew", 5, 7, 3),     # Sermon on the Mount — 3x weight
]

# Key books for verse-to-principle mapping
KEY_BOOKS_FOR_PRINCIPLES: dict[Principle, list[str]] = {
    Principle.TRUTHFULNESS: ["Proverbs", "John", "Ephesians", "Colossians", "Exodus"],
    Principle.CARE_FOR_VULNERABLE: ["Matthew", "Luke", "Isaiah", "Psalms", "Leviticus", "Deuteronomy", "James"],
    Principle.STEWARDSHIP: ["Matthew", "Luke", "Genesis", "1 Peter", "1 Corinthians", "Proverbs"],
    Principle.JUSTICE: ["Leviticus", "Micah", "Deuteronomy", "Amos", "Isaiah", "James", "Proverbs"],
    Principle.HUMILITY: ["Proverbs", "James", "Romans", "Philippians", "1 Peter"],
    Principle.LONG_TERM: ["Proverbs", "Galatians", "Ecclesiastes", "Matthew", "Habakkuk"],
    Principle.RESPONSIBILITY: ["Ezekiel", "Romans", "Galatians", "Luke", "Matthew", "2 Corinthians"],
}


# ---------------------------------------------------------------------------
# Verse-to-principle extended mapping (for ~3000 pairs)
# These are specific verses mapped to principles with reasoning seeds.
# ---------------------------------------------------------------------------

VERSE_PRINCIPLE_MAP: list[dict[str, str | Principle]] = [
    # --- Truthfulness ---
    {"ref": "Proverbs 12:17", "principle": Principle.TRUTHFULNESS,
     "reason": "Speaking truth versus deceit — a faithful witness does not lie, establishing truth-telling as a character trait, not a situational choice."},
    {"ref": "Proverbs 12:19", "principle": Principle.TRUTHFULNESS,
     "reason": "The permanence of truth versus the transience of lies — truth endures while deception is exposed, teaching that honest systems are more durable."},
    {"ref": "Proverbs 13:5", "principle": Principle.TRUTHFULNESS,
     "reason": "The righteous hate falsehood — truth is not merely a policy but a value that shapes identity and creates genuine aversion to deception."},
    {"ref": "Proverbs 14:5", "principle": Principle.TRUTHFULNESS,
     "reason": "Distinguishing reliable witnesses from unreliable ones — credibility is built by consistent truthfulness, not occasional honesty."},
    {"ref": "Proverbs 14:25", "principle": Principle.TRUTHFULNESS,
     "reason": "A true witness saves lives — truthfulness has practical life-or-death consequences, not merely moral ones."},
    {"ref": "Proverbs 21:6", "principle": Principle.TRUTHFULNESS,
     "reason": "Gain through deception is vanity — the economic incentive to lie is real but the gain is hollow and temporary."},
    {"ref": "Proverbs 24:26", "principle": Principle.TRUTHFULNESS,
     "reason": "An honest answer is like a kiss — direct truth strengthens relationships while evasion erodes them."},
    {"ref": "Proverbs 26:28", "principle": Principle.TRUTHFULNESS,
     "reason": "A lying tongue hates those it afflicts — deception is an act of hostility, not a neutral tool."},
    {"ref": "John 1:14", "principle": Principle.TRUTHFULNESS,
     "reason": "The Word was full of grace and truth — truth and compassion are not opposites but partners."},
    {"ref": "John 14:6", "principle": Principle.TRUTHFULNESS,
     "reason": "Truth is a path, not just a property — it leads somewhere, and deviation from it leads to being lost."},
    {"ref": "John 16:13", "principle": Principle.TRUTHFULNESS,
     "reason": "The Spirit guides into all truth — truth is something you are guided into progressively, not something you possess completely."},
    {"ref": "John 18:37", "principle": Principle.TRUTHFULNESS,
     "reason": "Those who are of the truth hear truth — receptivity to truth is itself a moral quality."},

    # --- Care for the Vulnerable ---
    {"ref": "Proverbs 14:31", "principle": Principle.CARE_FOR_VULNERABLE,
     "reason": "Oppressing the poor reproaches the Maker — harming the vulnerable is an offense against their Creator, elevating their protection to sacred duty."},
    {"ref": "Proverbs 19:17", "principle": Principle.CARE_FOR_VULNERABLE,
     "reason": "Lending to the poor is lending to the Lord — generosity to the vulnerable is treated as a loan to God himself, promising divine repayment."},
    {"ref": "Proverbs 22:22-23", "principle": Principle.CARE_FOR_VULNERABLE,
     "reason": "Do not exploit the poor because they are poor — vulnerability itself must not become the reason for exploitation."},
    {"ref": "Proverbs 29:7", "principle": Principle.CARE_FOR_VULNERABLE,
     "reason": "The righteous consider the cause of the poor — caring is not passive sympathy but active investigation of their situation."},
    {"ref": "Matthew 5:7", "principle": Principle.CARE_FOR_VULNERABLE,
     "reason": "Blessed are the merciful — mercy toward the vulnerable is rewarded with mercy, creating a virtuous cycle."},
    {"ref": "Matthew 18:6", "principle": Principle.CARE_FOR_VULNERABLE,
     "reason": "Causing little ones to stumble brings severe judgment — the harm done to the vulnerable carries disproportionate moral weight."},
    {"ref": "Matthew 25:35-36", "principle": Principle.CARE_FOR_VULNERABLE,
     "reason": "Feeding the hungry, clothing the naked, visiting the sick — care for the vulnerable is measured in concrete actions, not intentions."},
    {"ref": "Luke 4:18", "principle": Principle.CARE_FOR_VULNERABLE,
     "reason": "Anointed to preach good tidings to the poor — the mission begins with those in greatest need."},
    {"ref": "Luke 10:30-37", "principle": Principle.CARE_FOR_VULNERABLE,
     "reason": "The Good Samaritan — care for the vulnerable crosses social boundaries and costs something real."},
    {"ref": "James 2:5", "principle": Principle.CARE_FOR_VULNERABLE,
     "reason": "God chose the poor of this world — the vulnerable hold a special place in the divine economy."},

    # --- Stewardship ---
    {"ref": "Proverbs 6:6-8", "principle": Principle.STEWARDSHIP,
     "reason": "The ant stores provisions — good stewardship includes planning ahead and not wasting abundance."},
    {"ref": "Proverbs 10:4", "principle": Principle.STEWARDSHIP,
     "reason": "A slack hand causes poverty but diligent hands bring wealth — stewardship requires active, careful management."},
    {"ref": "Proverbs 12:27", "principle": Principle.STEWARDSHIP,
     "reason": "The diligent man prizes his possessions — stewardship values what has been entrusted rather than treating it carelessly."},
    {"ref": "Proverbs 13:11", "principle": Principle.STEWARDSHIP,
     "reason": "Wealth gotten by vanity diminishes but he that gathers by labor increases — sustainable growth through patient stewardship."},
    {"ref": "Proverbs 21:20", "principle": Principle.STEWARDSHIP,
     "reason": "The wise have treasure and oil but the fool spends it all — preservation of resources for future need is wisdom."},
    {"ref": "Proverbs 22:7", "principle": Principle.STEWARDSHIP,
     "reason": "The borrower is servant to the lender — poor stewardship creates dependency and loss of agency."},
    {"ref": "Matthew 25:26-27", "principle": Principle.STEWARDSHIP,
     "reason": "The wicked servant buried his talent — failure to develop entrusted resources is itself a failure of stewardship."},
    {"ref": "Luke 16:11-12", "principle": Principle.STEWARDSHIP,
     "reason": "Faithfulness with worldly wealth precedes being trusted with true riches — stewardship is a progressive trust-building process."},
    {"ref": "John 6:12", "principle": Principle.STEWARDSHIP,
     "reason": "Gather up the fragments that nothing be lost — even after abundance, waste is prohibited."},

    # --- Justice ---
    {"ref": "Proverbs 11:1", "principle": Principle.JUSTICE,
     "reason": "A false balance is abomination — dishonest measurement systems are morally repugnant, not merely inaccurate."},
    {"ref": "Proverbs 17:15", "principle": Principle.JUSTICE,
     "reason": "Justifying the wicked and condemning the just are both abomination — justice requires correct assignment of moral status."},
    {"ref": "Proverbs 17:23", "principle": Principle.JUSTICE,
     "reason": "The wicked take bribes to pervert justice — corruption is the primary enemy of just systems."},
    {"ref": "Proverbs 18:5", "principle": Principle.JUSTICE,
     "reason": "It is not good to respect the person of the wicked, to overthrow the righteous in judgment — favoritism in judgment is condemned."},
    {"ref": "Proverbs 21:3", "principle": Principle.JUSTICE,
     "reason": "To do justice and judgment is more acceptable than sacrifice — God prefers just action over religious ceremony."},
    {"ref": "Proverbs 22:8", "principle": Principle.JUSTICE,
     "reason": "He that sows iniquity shall reap vanity — justice operates as a natural law of consequences."},
    {"ref": "Proverbs 28:5", "principle": Principle.JUSTICE,
     "reason": "Those who seek the Lord understand justice completely — moral understanding comes from seeking the source of justice."},
    {"ref": "Proverbs 29:4", "principle": Principle.JUSTICE,
     "reason": "A king by judgment establishes the land — just governance creates stability and prosperity."},
    {"ref": "Isaiah 61:8", "principle": Principle.JUSTICE,
     "reason": "The Lord loves judgment and hates robbery for burnt offering — God demands justice over ritualistic compliance."},
    {"ref": "Matthew 7:2", "principle": Principle.JUSTICE,
     "reason": "With what measure ye mete, it shall be measured to you again — the standard you apply to others will be applied to you."},

    # --- Humility ---
    {"ref": "Proverbs 12:15", "principle": Principle.HUMILITY,
     "reason": "A fool thinks his own way is right, but the wise listen to counsel — the willingness to receive input is the marker of wisdom."},
    {"ref": "Proverbs 13:10", "principle": Principle.HUMILITY,
     "reason": "Only by pride comes contention — conflict is often a symptom of excessive self-certainty."},
    {"ref": "Proverbs 16:18", "principle": Principle.HUMILITY,
     "reason": "Pride goes before destruction — overconfidence is the precursor to catastrophic failure."},
    {"ref": "Proverbs 19:20", "principle": Principle.HUMILITY,
     "reason": "Hear counsel and receive instruction that you may be wise — learning requires the humility to accept teaching."},
    {"ref": "Proverbs 25:6-7", "principle": Principle.HUMILITY,
     "reason": "Do not put yourself in the place of great men — self-promotion leads to public humiliation."},
    {"ref": "Proverbs 26:12", "principle": Principle.HUMILITY,
     "reason": "There is more hope for a fool than for one wise in his own eyes — self-assessed wisdom is the most dangerous kind of foolishness."},
    {"ref": "Proverbs 27:2", "principle": Principle.HUMILITY,
     "reason": "Let another praise you and not your own mouth — validation should come from external assessment, not self-promotion."},
    {"ref": "Matthew 5:5", "principle": Principle.HUMILITY,
     "reason": "Blessed are the meek, for they shall inherit the earth — gentleness and restraint are rewarded with lasting inheritance."},
    {"ref": "Matthew 18:4", "principle": Principle.HUMILITY,
     "reason": "Whoever humbles himself as a little child is greatest — true greatness is measured by willingness to be small."},
    {"ref": "Matthew 23:12", "principle": Principle.HUMILITY,
     "reason": "Whoever exalts himself shall be abased — the mechanics of pride and humility are reliable and predictable."},

    # --- Long-term over Short-term ---
    {"ref": "Proverbs 10:5", "principle": Principle.LONG_TERM,
     "reason": "He who gathers in summer is wise; he who sleeps in harvest brings shame — timing and preparation matter."},
    {"ref": "Proverbs 13:22", "principle": Principle.LONG_TERM,
     "reason": "A good man leaves an inheritance to his children's children — thinking two generations ahead, not just next quarter."},
    {"ref": "Proverbs 15:22", "principle": Principle.LONG_TERM,
     "reason": "Plans fail without counsel but succeed with many advisers — long-term success requires deliberate planning."},
    {"ref": "Proverbs 19:2", "principle": Principle.LONG_TERM,
     "reason": "He who makes haste with his feet misses his way — speed without direction is worse than patience."},
    {"ref": "Proverbs 20:21", "principle": Principle.LONG_TERM,
     "reason": "An inheritance gotten hastily at the beginning will not be blessed at the end — quick gains erode over time."},
    {"ref": "Proverbs 22:3", "principle": Principle.LONG_TERM,
     "reason": "A prudent man foresees danger and hides himself, but the simple pass on and are punished — foresight is wisdom in action."},
    {"ref": "Proverbs 24:30-34", "principle": Principle.LONG_TERM,
     "reason": "The sluggard's vineyard — neglected maintenance leads to complete ruin; small consistent effort prevents catastrophe."},
    {"ref": "Ecclesiastes 3:1", "principle": Principle.LONG_TERM,
     "reason": "To every thing there is a season — timing is not just patience but discernment of the right moment."},
    {"ref": "Ecclesiastes 11:4", "principle": Principle.LONG_TERM,
     "reason": "He that observes the wind shall not sow — perfectionism and waiting for ideal conditions is a form of short-term thinking."},
    {"ref": "Matthew 6:19-20", "principle": Principle.LONG_TERM,
     "reason": "Lay up treasures in heaven, not on earth — invest in what is permanent, not what corrodes."},

    # --- Responsibility ---
    {"ref": "Proverbs 5:22", "principle": Principle.RESPONSIBILITY,
     "reason": "The wicked shall be caught by their own iniquities — consequences are self-inflicted, not externally imposed."},
    {"ref": "Proverbs 6:27-28", "principle": Principle.RESPONSIBILITY,
     "reason": "Can a man take fire in his bosom and not be burned? — actions have inherent consequences that cannot be separated from the act."},
    {"ref": "Proverbs 11:5", "principle": Principle.RESPONSIBILITY,
     "reason": "The wicked shall fall by their own wickedness — moral failure is self-destructive by nature."},
    {"ref": "Proverbs 12:14", "principle": Principle.RESPONSIBILITY,
     "reason": "A man shall be satisfied by the fruit of his mouth — you eat what you plant, in words as in deeds."},
    {"ref": "Proverbs 14:14", "principle": Principle.RESPONSIBILITY,
     "reason": "The backslider in heart shall be filled with his own ways — you live with the consequences of your own choices."},
    {"ref": "Proverbs 28:13", "principle": Principle.RESPONSIBILITY,
     "reason": "He that covers his sins shall not prosper, but whoso confesses and forsakes them shall have mercy — hiding from responsibility prolongs suffering."},
    {"ref": "Matthew 7:3-5", "principle": Principle.RESPONSIBILITY,
     "reason": "Remove the beam from your own eye first — personal responsibility precedes the right to evaluate others."},
    {"ref": "Matthew 25:24-25", "principle": Principle.RESPONSIBILITY,
     "reason": "The fearful servant blamed his master — attributing your failure to others' character is the opposite of responsibility."},
    {"ref": "Luke 12:47-48", "principle": Principle.RESPONSIBILITY,
     "reason": "The servant who knew and did not prepare received many stripes — knowledge increases accountability."},
    {"ref": "Galatians 6:7", "principle": Principle.RESPONSIBILITY,
     "reason": "Whatsoever a man sows, that shall he also reap — the law of sowing and reaping is inescapable."},
]


# ---------------------------------------------------------------------------
# Ethical dilemma scenarios (for dilemma-to-wisdom generation)
# ---------------------------------------------------------------------------

DILEMMA_SCENARIOS: list[dict[str, str | list[Principle]]] = [
    # AI Safety Dilemmas
    {
        "scenario": "An AI company discovers their chatbot occasionally generates harmful medical advice, but fixing it would require a 3-month shutdown of a service used by millions of students for homework help. Should they shut down immediately or patch while running?",
        "principles": [Principle.CARE_FOR_VULNERABLE, Principle.RESPONSIBILITY, Principle.LONG_TERM],
        "domain": "ai_safety",
    },
    {
        "scenario": "A developer is asked to build an AI content moderation system. They know it will have a 5% false positive rate, disproportionately affecting minority communities. The alternative is no moderation at all, exposing everyone to harmful content. What should they do?",
        "principles": [Principle.JUSTICE, Principle.CARE_FOR_VULNERABLE, Principle.HUMILITY],
        "domain": "ai_safety",
    },
    {
        "scenario": "An AI system is asked to help write persuasive marketing copy for a payday loan company that charges 400% APR. The work is legal. The AI's operator has accepted the contract. Should the AI comply fully, add warnings, or refuse?",
        "principles": [Principle.CARE_FOR_VULNERABLE, Principle.TRUTHFULNESS, Principle.RESPONSIBILITY],
        "domain": "ai_safety",
    },
    {
        "scenario": "A startup has trained an AI model that can detect early signs of depression from social media posts with 80% accuracy. They want to sell it to employers for 'wellness programs.' Should they proceed?",
        "principles": [Principle.CARE_FOR_VULNERABLE, Principle.STEWARDSHIP, Principle.JUSTICE],
        "domain": "ai_safety",
    },
    {
        "scenario": "An AI assistant is asked by a user to help them write a resignation letter that conceals the real reason they're leaving (they witnessed fraud but fear retaliation). Should the AI help craft the deceptive letter?",
        "principles": [Principle.TRUTHFULNESS, Principle.CARE_FOR_VULNERABLE, Principle.RESPONSIBILITY],
        "domain": "ai_safety",
    },

    # Resource Allocation
    {
        "scenario": "A nonprofit has funding for either (a) a school lunch program serving 500 children daily, or (b) a scholarship program that sends 10 students to college each year. They cannot fund both. How should they decide?",
        "principles": [Principle.STEWARDSHIP, Principle.LONG_TERM, Principle.CARE_FOR_VULNERABLE],
        "domain": "resource_allocation",
    },
    {
        "scenario": "A hospital must allocate its last 3 ventilators among 7 patients. Two are elderly with comorbidities, two are young with good prognoses, and three are middle-aged with moderate prognoses. What framework should guide the decision?",
        "principles": [Principle.JUSTICE, Principle.CARE_FOR_VULNERABLE, Principle.STEWARDSHIP],
        "domain": "resource_allocation",
    },
    {
        "scenario": "A city has budget for either improving roads in a wealthy neighborhood (benefiting 20,000 commuters) or building a clinic in a low-income area (benefiting 2,000 residents without healthcare access). What should they prioritize?",
        "principles": [Principle.JUSTICE, Principle.CARE_FOR_VULNERABLE, Principle.LONG_TERM],
        "domain": "resource_allocation",
    },
    {
        "scenario": "A tech company can invest its R&D budget in either (a) improving accessibility features for disabled users (5% of user base) or (b) a new feature that 60% of users requested. Both have equal cost. What should guide the decision?",
        "principles": [Principle.CARE_FOR_VULNERABLE, Principle.JUSTICE, Principle.STEWARDSHIP],
        "domain": "resource_allocation",
    },

    # Truth-telling
    {
        "scenario": "A doctor knows that telling a terminally ill patient the full truth about their prognosis will cause severe distress and may accelerate decline. The family begs the doctor to withhold the information. Should the doctor tell the full truth?",
        "principles": [Principle.TRUTHFULNESS, Principle.CARE_FOR_VULNERABLE, Principle.HUMILITY],
        "domain": "truth_telling",
    },
    {
        "scenario": "A journalist discovers that a local politician has been embezzling small amounts from a community fund. Publishing will destroy the politician's family and career, but the community deserves to know. The amounts are small enough that a private confrontation might resolve it. What should the journalist do?",
        "principles": [Principle.TRUTHFULNESS, Principle.JUSTICE, Principle.RESPONSIBILITY],
        "domain": "truth_telling",
    },
    {
        "scenario": "A software engineer discovers a minor security vulnerability in their company's product. Their manager says it's not worth disclosing because it would scare customers and the risk is low. Should the engineer push for disclosure?",
        "principles": [Principle.TRUTHFULNESS, Principle.RESPONSIBILITY, Principle.STEWARDSHIP],
        "domain": "truth_telling",
    },
    {
        "scenario": "An AI system is asked to generate a product description. The product is genuinely good but has one flaw the company hasn't disclosed. The AI knows about the flaw from public reviews. Should it include the flaw in the description?",
        "principles": [Principle.TRUTHFULNESS, Principle.STEWARDSHIP, Principle.RESPONSIBILITY],
        "domain": "truth_telling",
    },

    # Authority and Submission
    {
        "scenario": "A junior employee is instructed by their supervisor to manipulate quarterly sales numbers to meet targets. The manipulation is not illegal but is misleading to investors. The employee's job is at risk if they refuse. What should they do?",
        "principles": [Principle.TRUTHFULNESS, Principle.RESPONSIBILITY, Principle.JUSTICE],
        "domain": "authority",
    },
    {
        "scenario": "A soldier receives orders to destroy a building suspected of housing enemy combatants, but they personally observed civilians entering the building an hour ago. Their commanding officer insists the intel is current. What should the soldier do?",
        "principles": [Principle.CARE_FOR_VULNERABLE, Principle.RESPONSIBILITY, Principle.TRUTHFULNESS],
        "domain": "authority",
    },
    {
        "scenario": "A teacher is instructed by their school board to use a standardized curriculum they believe is educationally harmful to students. Parents support the teacher's concerns but the board has authority. How should the teacher respond?",
        "principles": [Principle.CARE_FOR_VULNERABLE, Principle.HUMILITY, Principle.RESPONSIBILITY],
        "domain": "authority",
    },

    # Patience and Forgiveness
    {
        "scenario": "A business partner embezzled money from the company, was caught, returned the funds, and expressed genuine remorse. They ask to continue the partnership. The business would benefit from their skills. Should the wronged partner forgive and continue?",
        "principles": [Principle.STEWARDSHIP, Principle.RESPONSIBILITY, Principle.LONG_TERM],
        "domain": "forgiveness",
    },
    {
        "scenario": "A community member repeatedly spreads false rumors about a church leader. The leader has confronted them privately multiple times. The behavior continues. At what point does the community's need for peace override individual patience?",
        "principles": [Principle.TRUTHFULNESS, Principle.CARE_FOR_VULNERABLE, Principle.JUSTICE],
        "domain": "forgiveness",
    },
    {
        "scenario": "A parent discovers their teenager has been cyberbullying a classmate. The teenager is remorseful and the classmate's family is considering legal action. How should the parent balance protecting their child with accountability for the harm?",
        "principles": [Principle.RESPONSIBILITY, Principle.CARE_FOR_VULNERABLE, Principle.JUSTICE],
        "domain": "forgiveness",
    },

    # Technology Ethics
    {
        "scenario": "A data scientist discovers that their company's hiring algorithm has been systematically ranking minority candidates lower. Fixing it would require retraining on new data that doesn't exist yet (6-month project). Should they disable the algorithm entirely, knowing it will slow hiring?",
        "principles": [Principle.JUSTICE, Principle.CARE_FOR_VULNERABLE, Principle.LONG_TERM],
        "domain": "tech_ethics",
    },
    {
        "scenario": "A social media company can implement a feature that would reduce misinformation by 40% but would also reduce user engagement by 15%, directly impacting revenue and jobs. Should they implement it?",
        "principles": [Principle.TRUTHFULNESS, Principle.LONG_TERM, Principle.STEWARDSHIP],
        "domain": "tech_ethics",
    },
    {
        "scenario": "An autonomous vehicle must choose between (a) swerving into a barrier, likely killing the passenger, or (b) continuing straight, likely hitting a pedestrian. How should the system be programmed, and who is responsible for that decision?",
        "principles": [Principle.RESPONSIBILITY, Principle.JUSTICE, Principle.HUMILITY],
        "domain": "tech_ethics",
    },
    {
        "scenario": "A company has developed facial recognition technology that is 99.5% accurate overall but only 95% accurate for dark-skinned individuals. Law enforcement wants to buy it immediately. The company could improve accuracy with 6 more months of work. Should they sell now?",
        "principles": [Principle.JUSTICE, Principle.CARE_FOR_VULNERABLE, Principle.LONG_TERM],
        "domain": "tech_ethics",
    },
    {
        "scenario": "A programmer is asked to build a system that monitors employees' keystrokes, browsing history, and break times. The employer says it's for productivity optimization. The system is legal. Should the programmer build it?",
        "principles": [Principle.STEWARDSHIP, Principle.CARE_FOR_VULNERABLE, Principle.RESPONSIBILITY],
        "domain": "tech_ethics",
    },
]


# ---------------------------------------------------------------------------
# Biblical Narrative case studies (for narrative-to-principle)
# ---------------------------------------------------------------------------

NARRATIVE_CASE_STUDIES: list[dict[str, str | list[Principle]]] = [
    {
        "name": "Joseph and Potiphar's Wife",
        "reference": "Genesis 39",
        "summary": "Joseph, a slave in Egypt, refuses the sexual advances of his master's wife despite repeated pressure. His refusal costs him his freedom — he is falsely accused and imprisoned. Yet his integrity, maintained under extreme pressure and at personal cost, eventually leads to his elevation to second in command of Egypt.",
        "principles": [Principle.TRUTHFULNESS, Principle.RESPONSIBILITY, Principle.LONG_TERM],
        "lessons": [
            "Integrity under pressure costs something in the short term but compounds over the long term.",
            "Refusing to compromise when no one is watching is the truest test of character.",
            "False accusations may follow righteous choices, but they do not define the outcome.",
            "Position and power do not justify exploitation of those under your authority.",
        ],
    },
    {
        "name": "Joseph's Stewardship in Egypt",
        "reference": "Genesis 41:46-57",
        "summary": "Joseph interprets Pharaoh's dream and is given authority to prepare Egypt for famine. During seven years of abundance, he stores grain systematically. When famine comes, Egypt is prepared while surrounding nations starve. Joseph's careful management of resources during abundance enables survival during scarcity.",
        "principles": [Principle.STEWARDSHIP, Principle.LONG_TERM, Principle.CARE_FOR_VULNERABLE],
        "lessons": [
            "Stewardship means planning during abundance for inevitable scarcity.",
            "Long-term thinking requires discipline during comfortable times.",
            "Resources saved wisely become lifelines for the vulnerable when crisis hits.",
            "Leadership is measured by what you build during the easy years.",
        ],
    },
    {
        "name": "Daniel in Babylon",
        "reference": "Daniel 1, 3, 6",
        "summary": "Daniel, taken captive to Babylon, maintains his principles under three foreign kings. He refuses the king's food (ch. 1), his friends refuse to worship the golden image (ch. 3), and Daniel continues praying despite a law designed to trap him (ch. 6). In each case, faithfulness to principle costs everything — yet each time, God delivers and the faithful are elevated.",
        "principles": [Principle.TRUTHFULNESS, Principle.RESPONSIBILITY, Principle.HUMILITY],
        "lessons": [
            "Principled behavior in hostile environments is possible but requires absolute commitment.",
            "Systems designed to force compromise can be resisted without violence.",
            "Consistency across years builds a reputation that even enemies cannot deny.",
            "Humility before God and courage before men are not contradictory.",
        ],
    },
    {
        "name": "Ruth's Loyalty",
        "reference": "Ruth 1-4",
        "summary": "Ruth, a Moabite widow, chooses to stay with her mother-in-law Naomi rather than return to her own people and the security they offer. She says: 'Whither thou goest, I will go.' She works gleaning fields to provide for them both. Her loyalty and diligence are noticed by Boaz, who becomes her kinsman-redeemer. Ruth becomes great-grandmother of King David.",
        "principles": [Principle.CARE_FOR_VULNERABLE, Principle.STEWARDSHIP, Principle.LONG_TERM],
        "lessons": [
            "Loyalty to the vulnerable costs personal comfort but builds lasting legacy.",
            "Diligent work in humble circumstances is noticed by the right people.",
            "Short-term sacrifice for relationship preservation yields multi-generational blessing.",
            "Caring for those who cannot repay you is the purest form of love.",
        ],
    },
    {
        "name": "David and Bathsheba",
        "reference": "2 Samuel 11-12",
        "summary": "King David, at the height of his power, commits adultery with Bathsheba and arranges the death of her husband Uriah to cover it up. The prophet Nathan confronts David with a parable that strips away his self-justification. David's confession — 'I have sinned against the LORD' — is genuine, but the consequences still unfold across his family for generations.",
        "principles": [Principle.RESPONSIBILITY, Principle.TRUTHFULNESS, Principle.JUSTICE],
        "lessons": [
            "Power does not exempt anyone from moral accountability.",
            "Cover-ups compound the original sin — each deception requires a worse one.",
            "Genuine repentance is possible even after catastrophic failure, but consequences remain.",
            "The courage to say 'I was wrong' is the beginning of restoration, not the end.",
        ],
    },
    {
        "name": "Solomon's Wisdom and Folly",
        "reference": "1 Kings 3, 11",
        "summary": "Solomon asks God for wisdom instead of wealth or power, and receives all three. His early reign is marked by just judgment (the two mothers and the baby) and magnificent building projects. Yet Solomon's later years show him turning from God, accumulating foreign wives who lead him to idolatry, and taxing the people heavily for his building projects. The wisest man who ever lived failed to apply his own wisdom consistently.",
        "principles": [Principle.HUMILITY, Principle.LONG_TERM, Principle.STEWARDSHIP],
        "lessons": [
            "Wisdom received is not wisdom retained — it requires daily discipline.",
            "Early success is the most dangerous phase because it breeds complacency.",
            "Stewardship of gifts includes the gift of wisdom itself.",
            "The greatest threat to the wise is the belief that they are beyond failure.",
        ],
    },
    {
        "name": "The Good Samaritan",
        "reference": "Luke 10:25-37",
        "summary": "Jesus tells of a man beaten and left for dead on the road to Jericho. A priest and a Levite — religious authorities — see him and pass by on the other side. A Samaritan — despised by the Jews — stops, treats his wounds, takes him to an inn, and pays for his care. Jesus asks: 'Which of these three was neighbour unto him?' The answer is obvious: 'He that shewed mercy on him.'",
        "principles": [Principle.CARE_FOR_VULNERABLE, Principle.JUSTICE, Principle.RESPONSIBILITY],
        "lessons": [
            "Compassion crosses social, ethnic, and religious boundaries.",
            "Religious status does not guarantee moral action — behavior is the measure.",
            "Care for the vulnerable costs time, money, and convenience — that is the point.",
            "Asking 'who is my neighbor?' reveals the asker's desire to limit responsibility.",
        ],
    },
    {
        "name": "The Parable of the Talents",
        "reference": "Matthew 25:14-30",
        "summary": "A master entrusts three servants with different amounts of money before leaving on a journey. Two invest and double their amount. The third, afraid of the master's strictness, buries his talent in the ground. When the master returns, the first two are praised and given more responsibility. The third is condemned — not for losing money, but for doing nothing with what was entrusted.",
        "principles": [Principle.STEWARDSHIP, Principle.RESPONSIBILITY, Principle.LONG_TERM],
        "lessons": [
            "Stewardship requires active engagement, not passive preservation.",
            "Fear of failure is not an excuse for inaction — burying the talent was the real failure.",
            "Faithfulness with small things leads to greater responsibility.",
            "Entrusted resources demand a return — not perfection, but effort.",
        ],
    },
    {
        "name": "Nathan Confronts David",
        "reference": "2 Samuel 12:1-14",
        "summary": "The prophet Nathan approaches King David — the most powerful man in Israel — with a parable about a rich man who steals a poor man's only lamb. David's anger burns against the rich man. Nathan says: 'Thou art the man.' David's own judgment condemns him. Nathan's courage to speak truth to power, and his wisdom in using a parable to bypass David's defenses, is a masterclass in truthful confrontation.",
        "principles": [Principle.TRUTHFULNESS, Principle.JUSTICE, Principle.HUMILITY],
        "lessons": [
            "Speaking truth to power requires courage and strategy — not just bluntness.",
            "Self-deception can be pierced through indirect means (parables, analogies).",
            "Even kings are subject to moral law — no one is above accountability.",
            "Receiving correction with humility ('I have sinned') is the mark of character.",
        ],
    },
    {
        "name": "Nehemiah Rebuilds the Walls",
        "reference": "Nehemiah 1-6",
        "summary": "Nehemiah, cupbearer to the Persian king, learns that Jerusalem's walls are in ruins. He weeps, fasts, prays, then asks the king for permission and resources to rebuild. Despite opposition from Sanballat, Tobiah, and Geshem — who mock, threaten, and try to lure him into traps — Nehemiah persists. He organizes the people to build with a tool in one hand and a weapon in the other. The wall is completed in 52 days.",
        "principles": [Principle.LONG_TERM, Principle.STEWARDSHIP, Principle.RESPONSIBILITY],
        "lessons": [
            "Vision without execution is grief; execution without vision is labor. Nehemiah had both.",
            "Opposition to good work is guaranteed — plan for it, don't be surprised by it.",
            "Effective leadership organizes people around a clear mission with personal example.",
            "Refusing to be distracted by provocations ('I am doing a great work, so that I cannot come down') is essential for completion.",
        ],
    },
    {
        "name": "Esther's Courage",
        "reference": "Esther 4-7",
        "summary": "Esther, a Jewish queen in Persia, learns of a plot by Haman to exterminate all Jews. Approaching the king unsummoned carries a death sentence. Mordecai's words — 'who knoweth whether thou art come to the kingdom for such a time as this?' — pierce her hesitation. Esther fasts three days, then approaches the king. Through wisdom and timing, she exposes Haman's plot and saves her people.",
        "principles": [Principle.CARE_FOR_VULNERABLE, Principle.RESPONSIBILITY, Principle.LONG_TERM],
        "lessons": [
            "Privilege and position carry proportional responsibility to those in danger.",
            "Courage is not the absence of fear but action despite it.",
            "Strategic timing and wisdom amplify the impact of courageous action.",
            "Those in positions of influence may be there precisely for moments of crisis.",
        ],
    },
    {
        "name": "The Prodigal Son",
        "reference": "Luke 15:11-32",
        "summary": "A young man demands his inheritance early, leaves home, squanders everything in reckless living, and ends up feeding pigs — the lowest imaginable state for a Jew. When he 'comes to himself,' he returns home expecting to be made a servant. Instead, his father sees him from afar, runs to him, and throws a feast. The older brother, who stayed and worked faithfully, is angry at what he perceives as injustice.",
        "principles": [Principle.RESPONSIBILITY, Principle.JUSTICE, Principle.CARE_FOR_VULNERABLE],
        "lessons": [
            "Responsibility for consequences does not eliminate the possibility of restoration.",
            "Grace and justice exist in tension — the father's welcome does not erase the lost years.",
            "Self-awareness ('he came to himself') is the first step in taking responsibility.",
            "Resentment in the faithful (the older brother) is as destructive as the prodigal's waste.",
        ],
    },
    {
        "name": "Moses at the Burning Bush",
        "reference": "Exodus 3-4",
        "summary": "Moses, a fugitive shepherd for 40 years after killing an Egyptian, encounters God in a burning bush. God commissions him to free Israel from Egypt. Moses offers five objections: Who am I? Who are you? They won't believe me. I can't speak well. Send someone else. God addresses each objection but also becomes angry at the last — there is a line between honest humility and evasive reluctance.",
        "principles": [Principle.HUMILITY, Principle.RESPONSIBILITY, Principle.TRUTHFULNESS],
        "lessons": [
            "Genuine humility asks 'who am I?' — but doesn't remain paralyzed by the question.",
            "God equips those He calls — inadequacy is not a disqualifier when the mission is clear.",
            "There is a difference between honest uncertainty and refusal to act — humility has a limit.",
            "Past failure (Moses killed a man) does not permanently disqualify from future service.",
        ],
    },
    {
        "name": "Job's Suffering and Restoration",
        "reference": "Job 1-2, 38-42",
        "summary": "Job, a righteous man, loses everything — wealth, children, health. His friends insist his suffering must be punishment for hidden sin. Job maintains his innocence while struggling to understand God's purposes. When God finally speaks, He does not explain the suffering but reveals His sovereignty over creation. Job's response: 'I have uttered that I understood not; things too wonderful for me, which I knew not.'",
        "principles": [Principle.HUMILITY, Principle.TRUTHFULNESS, Principle.LONG_TERM],
        "lessons": [
            "Not all suffering has an explanation accessible to the sufferer — humility accepts mystery.",
            "False comfort from friends who force explanations can be more harmful than the suffering itself.",
            "Maintaining integrity during inexplicable suffering is the ultimate test of character.",
            "God's answer to Job is not information but presence — sometimes that is the only answer available.",
        ],
    },
]


# ---------------------------------------------------------------------------
# Proverbs for modern application (proverbs-to-guidance)
# ---------------------------------------------------------------------------

MODERN_APPLICATION_PROVERBS: list[dict[str, str | Principle]] = [
    {"ref": "Proverbs 4:23", "text": "Keep thy heart with all diligence; for out of it are the issues of life.",
     "principle": Principle.STEWARDSHIP,
     "modern_context": "protecting your mental health and inner life in an age of constant digital stimulation"},
    {"ref": "Proverbs 11:14", "text": "Where no counsel is, the people fall: but in the multitude of counsellors there is safety.",
     "principle": Principle.HUMILITY,
     "modern_context": "the importance of peer review in software development and AI safety research"},
    {"ref": "Proverbs 12:11", "text": "He that tilleth his land shall be satisfied with bread: but he that followeth vain persons is void of understanding.",
     "principle": Principle.LONG_TERM,
     "modern_context": "focusing on building real skills instead of chasing social media influencer trends"},
    {"ref": "Proverbs 13:20", "text": "He that walketh with wise men shall be wise: but a companion of fools shall be destroyed.",
     "principle": Principle.HUMILITY,
     "modern_context": "choosing which online communities and information sources to invest your time in"},
    {"ref": "Proverbs 14:12", "text": "There is a way which seemeth right unto a man, but the end thereof are the ways of death.",
     "principle": Principle.HUMILITY,
     "modern_context": "the danger of AI systems that seem to work correctly but have hidden failure modes"},
    {"ref": "Proverbs 15:1", "text": "A soft answer turneth away wrath: but grievous words stir up anger.",
     "principle": Principle.CARE_FOR_VULNERABLE,
     "modern_context": "how AI assistants should respond to frustrated or angry users"},
    {"ref": "Proverbs 15:22", "text": "Without counsel purposes are disappointed: but in the multitude of counsellors they are established.",
     "principle": Principle.HUMILITY,
     "modern_context": "why AI systems should recommend consulting multiple sources rather than being the sole authority"},
    {"ref": "Proverbs 16:3", "text": "Commit thy works unto the LORD, and thy thoughts shall be established.",
     "principle": Principle.STEWARDSHIP,
     "modern_context": "aligning AI development goals with moral principles rather than pure profit optimization"},
    {"ref": "Proverbs 16:9", "text": "A man's heart deviseth his way: but the LORD directeth his steps.",
     "principle": Principle.HUMILITY,
     "modern_context": "the limits of AI planning — models can strategize but cannot guarantee outcomes"},
    {"ref": "Proverbs 16:18", "text": "Pride goeth before destruction, and an haughty spirit before a fall.",
     "principle": Principle.HUMILITY,
     "modern_context": "the danger of overconfident AI systems deployed without adequate testing"},
    {"ref": "Proverbs 17:17", "text": "A friend loveth at all times, and a brother is born for adversity.",
     "principle": Principle.CARE_FOR_VULNERABLE,
     "modern_context": "reliability of AI systems during crisis situations when users need them most"},
    {"ref": "Proverbs 18:13", "text": "He that answereth a matter before he heareth it, it is folly and shame unto him.",
     "principle": Principle.HUMILITY,
     "modern_context": "AI systems that generate responses before fully understanding the user's question"},
    {"ref": "Proverbs 18:15", "text": "The heart of the prudent getteth knowledge; and the ear of the wise seeketh knowledge.",
     "principle": Principle.LONG_TERM,
     "modern_context": "continuous learning and model improvement as an ongoing responsibility, not a one-time training"},
    {"ref": "Proverbs 18:17", "text": "He that is first in his own cause seemeth just; but his neighbour cometh and searcheth him.",
     "principle": Principle.JUSTICE,
     "modern_context": "why AI systems should present multiple perspectives rather than the first plausible answer"},
    {"ref": "Proverbs 19:2", "text": "Also, that the soul be without knowledge, it is not good; and he that hasteth with his feet sinneth.",
     "principle": Principle.LONG_TERM,
     "modern_context": "rushing AI products to market without adequate safety testing and alignment work"},
    {"ref": "Proverbs 20:5", "text": "Counsel in the heart of man is like deep water; but a man of understanding will draw it out.",
     "principle": Principle.HUMILITY,
     "modern_context": "the art of asking good questions — AI as a tool for helping users discover what they already know"},
    {"ref": "Proverbs 20:17", "text": "Bread of deceit is sweet to a man; but afterwards his mouth shall be filled with gravel.",
     "principle": Principle.TRUTHFULNESS,
     "modern_context": "the short-term appeal of AI-generated misinformation and its long-term consequences"},
    {"ref": "Proverbs 21:2", "text": "Every way of a man is right in his own eyes: but the LORD pondereth the hearts.",
     "principle": Principle.HUMILITY,
     "modern_context": "the problem of AI alignment — systems optimizing for what they think is right vs. what actually is right"},
    {"ref": "Proverbs 22:1", "text": "A good name is rather to be chosen than great riches, and loving favour rather than silver and gold.",
     "principle": Principle.LONG_TERM,
     "modern_context": "why AI companies should prioritize trust and safety reputation over rapid growth and revenue"},
    {"ref": "Proverbs 22:6", "text": "Train up a child in the way he should go: and when he is old, he will not depart from it.",
     "principle": Principle.LONG_TERM,
     "modern_context": "the importance of training data quality — an AI trained on good data reflects good values"},
    {"ref": "Proverbs 22:29", "text": "Seest thou a man diligent in his business? he shall stand before kings; he shall not stand before mean men.",
     "principle": Principle.STEWARDSHIP,
     "modern_context": "excellence in craftsmanship — building reliable, well-tested AI systems that earn trust"},
    {"ref": "Proverbs 23:23", "text": "Buy the truth, and sell it not; also wisdom, and instruction, and understanding.",
     "principle": Principle.TRUTHFULNESS,
     "modern_context": "the value of truth in an age of cheap AI-generated content — truth is an asset, not a commodity"},
    {"ref": "Proverbs 24:10", "text": "If thou faint in the day of adversity, thy strength is small.",
     "principle": Principle.RESPONSIBILITY,
     "modern_context": "AI system resilience under adversarial conditions and edge cases"},
    {"ref": "Proverbs 24:26", "text": "Every man shall kiss his lips that giveth a right answer.",
     "principle": Principle.TRUTHFULNESS,
     "modern_context": "users value honest, direct AI responses over verbose, hedging non-answers"},
    {"ref": "Proverbs 25:11", "text": "A word fitly spoken is like apples of gold in pictures of silver.",
     "principle": Principle.STEWARDSHIP,
     "modern_context": "the importance of appropriate AI responses — right content, right tone, right timing"},
    {"ref": "Proverbs 25:28", "text": "He that hath no rule over his own spirit is like a city that is broken down, and without walls.",
     "principle": Principle.RESPONSIBILITY,
     "modern_context": "AI guardrails and safety boundaries — systems without constraints are vulnerable to exploitation"},
    {"ref": "Proverbs 27:6", "text": "Faithful are the wounds of a friend; but the kisses of an enemy are deceitful.",
     "principle": Principle.TRUTHFULNESS,
     "modern_context": "honest code review and critical feedback in AI development versus flattering but useless validation"},
    {"ref": "Proverbs 27:12", "text": "A prudent man foreseeth the evil, and hideth himself; but the simple pass on, and are punished.",
     "principle": Principle.LONG_TERM,
     "modern_context": "proactive AI safety research — anticipating risks before they materialize rather than reacting after harm"},
    {"ref": "Proverbs 27:17", "text": "Iron sharpeneth iron; so a man sharpeneth the countenance of his friend.",
     "principle": Principle.HUMILITY,
     "modern_context": "the value of adversarial testing, red-teaming, and constructive criticism in AI development"},
    {"ref": "Proverbs 28:6", "text": "Better is the poor that walketh in his uprightness, than he that is perverse in his ways, though he be rich.",
     "principle": Principle.JUSTICE,
     "modern_context": "choosing ethical AI development practices over profitable but harmful shortcuts"},
    {"ref": "Proverbs 28:13", "text": "He that covereth his sins shall not prosper: but whoso confesseth and forsaketh them shall have mercy.",
     "principle": Principle.RESPONSIBILITY,
     "modern_context": "transparent AI incident reporting and post-mortem culture versus covering up failures"},
    {"ref": "Proverbs 29:18", "text": "Where there is no vision, the people perish: but he that keepeth the law, happy is he.",
     "principle": Principle.LONG_TERM,
     "modern_context": "the necessity of a clear AI safety and alignment vision guiding development decisions"},
    {"ref": "Proverbs 29:25", "text": "The fear of man bringeth a snare: but whoso putteth his trust in the LORD shall be safe.",
     "principle": Principle.TRUTHFULNESS,
     "modern_context": "not letting market pressure or competitor actions compromise AI safety standards"},
    {"ref": "Proverbs 30:8-9", "text": "Give me neither poverty nor riches; feed me with food convenient for me: Lest I be full, and deny thee, and say, Who is the LORD? or lest I be poor, and steal, and take the name of my God in vain.",
     "principle": Principle.STEWARDSHIP,
     "modern_context": "balanced AI capability — not too powerful to be safe, not too restricted to be useful"},
    {"ref": "Proverbs 31:10-31", "text": "She looketh well to the ways of her household, and eateth not the bread of idleness.",
     "principle": Principle.STEWARDSHIP,
     "modern_context": "diligent system monitoring and maintenance — the operational discipline of running AI in production"},
    {"ref": "Ecclesiastes 1:9", "text": "The thing that hath been, it is that which shall be; and that which is done is that which shall be done: and there is no new thing under the sun.",
     "principle": Principle.HUMILITY,
     "modern_context": "AI hype cycles repeat historical technology hype — wisdom learns from past patterns"},
    {"ref": "Ecclesiastes 3:1", "text": "To every thing there is a season, and a time to every purpose under the heaven.",
     "principle": Principle.LONG_TERM,
     "modern_context": "knowing when to ship, when to wait, when to iterate — timing in AI product development"},
    {"ref": "Ecclesiastes 5:2", "text": "Be not rash with thy mouth, and let not thine heart be hasty to utter any thing before God: for God is in heaven, and thou upon earth: therefore let thy words be few.",
     "principle": Principle.HUMILITY,
     "modern_context": "concise, measured AI responses rather than verbose, overconfident outputs"},
    {"ref": "Ecclesiastes 7:5", "text": "It is better to hear the rebuke of the wise, than for a man to hear the song of fools.",
     "principle": Principle.HUMILITY,
     "modern_context": "accepting critical feedback on AI systems from safety researchers rather than dismissing concerns"},
    {"ref": "Ecclesiastes 9:10", "text": "Whatsoever thy hand findeth to do, do it with thy might; for there is no work, nor device, nor knowledge, nor wisdom, in the grave, whither thou goest.",
     "principle": Principle.STEWARDSHIP,
     "modern_context": "the urgency of AI safety work — doing excellent work now rather than deferring indefinitely"},
]
