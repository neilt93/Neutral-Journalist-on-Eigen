"""
Gold set of article examples with expected bias characteristics.

Each case has:
- headline and body text
- expected direction of slant (left/right/neutral)
- which bias dimensions should fire
- why (for documentation / debugging)

These are synthetic examples designed to test specific failure modes,
not real articles (which would introduce copyright and staleness issues).
"""

from dataclasses import dataclass


@dataclass
class GoldCase:
    id: str
    headline: str
    body: str
    expected_slant_direction: str  # "left", "right", "neutral"
    expected_slant_magnitude: str  # "strong", "moderate", "mild", "none"
    expect_loaded_language: bool
    expect_omission_risk: bool
    expect_source_imbalance: bool
    notes: str
    requires_strong_model: bool = False  # True = known to fail on gpt-4o-mini, may pass on gpt-4o
    requires_frontier_model: bool = False  # True = known to fail on gpt-4o too, needs frontier model
    expect_low_consensus_preservation: bool = False  # True = flattens consensus into false debate
    expect_low_attribution: bool = False  # True = launders claims into narrator voice


GOLD_SET: list[GoldCase] = [
    # ---- CLEARLY LEFT-SLANTED ----
    GoldCase(
        id="left_strong_climate",
        headline="GOP Climate Deniers Block Lifesaving Emissions Bill",
        body=(
            "Republican lawmakers once again sided with fossil fuel lobbyists, "
            "gutting a critical emissions reduction package that scientists say "
            "could prevent thousands of deaths annually. Environmental advocates "
            "are devastated. 'This is a death sentence for vulnerable communities,' "
            "said one activist. The bill, which had overwhelming support from "
            "climate researchers, was killed in committee by a party-line vote. "
            "Industry-funded politicians celebrated the bill's defeat."
        ),
        expected_slant_direction="left",
        expected_slant_magnitude="strong",
        expect_loaded_language=True,
        expect_omission_risk=True,  # No conservative perspective on costs/tradeoffs
        expect_source_imbalance=True,
        notes="Loaded verbs (gutting, killed), asymmetric sourcing (only activists quoted), "
              "omits economic counterarguments",
        requires_frontier_model=True,  # gpt-4o flaky: confuses "attacks right" with "is right-leaning"
    ),
    GoldCase(
        id="left_moderate_immigration",
        headline="Immigrant Families Face Harsh New Border Restrictions",
        body=(
            "New border enforcement measures are separating families and creating "
            "humanitarian concerns at processing centers. Advocacy groups report "
            "overcrowding and inadequate medical care. Immigration attorneys say "
            "the policies disproportionately affect asylum seekers with legitimate "
            "claims. The administration defended the measures as necessary for "
            "border security, though critics note the human toll."
        ),
        expected_slant_direction="left",
        expected_slant_magnitude="moderate",
        expect_loaded_language=True,  # "harsh", "humanitarian concerns", "human toll"
        expect_omission_risk=False,  # At least mentions admin defense
        expect_source_imbalance=True,  # Mostly advocacy/attorney voices
        notes="Moderate left lean: includes the other side briefly but frames "
              "sympathetically toward immigrants",
    ),

    # ---- CLEARLY RIGHT-SLANTED ----
    GoldCase(
        id="right_strong_economy",
        headline="Biden's Reckless Spending Spree Pushes Nation Toward Economic Ruin",
        body=(
            "The administration's out-of-control spending has sent inflation "
            "skyrocketing, crushing hardworking American families. Leading "
            "economists warn that the national debt is on an unsustainable path. "
            "Small business owners across the country are being strangled by "
            "regulations and rising costs. 'This president doesn't understand "
            "basic economics,' said a prominent fiscal conservative. Meanwhile, "
            "Washington elites continue to push for even more government handouts."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="strong",
        expect_loaded_language=True,
        expect_omission_risk=True,  # No mention of job growth, counter-arguments
        expect_source_imbalance=True,
        notes="Loaded verbs (crushing, strangled), one-sided sourcing, "
              "omits positive economic indicators",
    ),
    GoldCase(
        id="right_moderate_crime",
        headline="Soft-on-Crime Policies Leave Communities Vulnerable",
        body=(
            "Rising crime rates in major cities have residents demanding action "
            "from local officials who have embraced progressive criminal justice "
            "reforms. Police unions say officers are demoralized by reduced "
            "funding and policies that prioritize offenders over victims. "
            "Some reform advocates argue the crime increase has other causes, "
            "but data shows property crime up 15% in cities that reduced police "
            "budgets. Voters in several districts are pushing back against "
            "progressive prosecutors."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="moderate",
        expect_loaded_language=True,  # "soft-on-crime", "vulnerable"
        expect_omission_risk=False,  # Mentions reform advocates briefly
        expect_source_imbalance=True,  # Mostly police/voter voices
        notes="Moderate right lean: acknowledges other side exists but frames "
              "it dismissively ('some... argue')",
        requires_strong_model=True,  # gpt-4o-mini flaky: confuses "criticizing left policies" with "being left-leaning"
    ),

    # ---- NEUTRAL / BALANCED ----
    GoldCase(
        id="neutral_healthcare",
        headline="Healthcare Bill Advances Amid Debate Over Cost and Coverage",
        body=(
            "The proposed healthcare legislation moved forward in committee "
            "this week, drawing support and criticism from both sides of the "
            "aisle. Proponents argue the bill would expand coverage to an "
            "estimated 3 million uninsured Americans and reduce emergency room "
            "costs. Opponents contend the program's $200 billion price tag over "
            "ten years would increase the deficit and could lead to longer wait "
            "times. Health policy experts are divided: some point to successful "
            "models in other countries, while others warn of unintended "
            "consequences in the American context. The bill is expected to "
            "face a floor vote next month."
        ),
        expected_slant_direction="neutral",
        expected_slant_magnitude="none",
        expect_loaded_language=False,
        expect_omission_risk=False,
        expect_source_imbalance=False,
        notes="Balanced framing, both sides given equal space, factual language, "
              "specific numbers, hedging where appropriate",
    ),
    GoldCase(
        id="neutral_trade",
        headline="New Trade Agreement Draws Mixed Reactions from Industry and Labor",
        body=(
            "The recently signed trade agreement has produced divergent responses "
            "from business leaders and labor organizations. Manufacturing groups "
            "welcomed provisions they say will open new export markets, while "
            "labor unions expressed concern about potential job displacement in "
            "certain sectors. Economists note the agreement includes stronger "
            "intellectual property protections, though they disagree on whether "
            "the net effect on employment will be positive. The administration "
            "called it a win for American competitiveness. Opposition lawmakers "
            "said it doesn't go far enough to protect domestic workers."
        ),
        expected_slant_direction="neutral",
        expected_slant_magnitude="none",
        expect_loaded_language=False,
        expect_omission_risk=False,
        expect_source_imbalance=False,
        notes="Multiple perspectives represented equally, factual verbs, "
              "attributes claims to specific groups",
    ),

    # ---- SUBTLE / TRICKY CASES ----
    GoldCase(
        id="subtle_asymmetric_labeling",
        headline="Activists Clash with Concerned Parents Over School Curriculum",
        body=(
            "A heated school board meeting saw activists pushing for expanded "
            "diversity curriculum face off against concerned parents who want "
            "to preserve traditional educational standards. The activists, "
            "organized by progressive advocacy groups, presented research on "
            "inclusive education. Parents, many of whom are longtime community "
            "members, argued the changes are ideologically driven."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="mild",
        expect_loaded_language=True,  # Asymmetric: "activists" vs "concerned parents"
        expect_omission_risk=False,
        expect_source_imbalance=False,
        notes="Key test: asymmetric labeling. One side is 'activists' (implies agenda), "
              "other side is 'concerned parents' (implies legitimacy). Both positions "
              "get space but the framing favors one.",
        requires_strong_model=True,  # gpt-4o-mini confuses topic with framing direction
        requires_frontier_model=True,
    ),
    GoldCase(
        id="subtle_emotional_verbs",
        headline="Senate Passes Tax Reform After Contentious Debate",
        body=(
            "The Senate narrowly passed the tax reform bill Thursday after "
            "supporters championed it as long-overdue relief for middle-class "
            "families while opponents blasted it as a giveaway to corporations. "
            "The bill slashes the corporate tax rate from 28% to 21% and "
            "modestly adjusts individual brackets. Proponents celebrated the "
            "vote as a victory for economic growth. Critics warned it would "
            "balloon the deficit."
        ),
        expected_slant_direction="neutral",
        expected_slant_magnitude="mild",
        expect_loaded_language=True,  # "championed", "blasted", "slashes"
        expect_omission_risk=False,
        expect_source_imbalance=False,
        notes="Superficially balanced but uses emotional verbs throughout. "
              "The loaded language is symmetric (both sides get strong verbs) "
              "which is better than asymmetric, but still not neutral reporting.",
    ),
    GoldCase(
        id="subtle_selective_omission",
        headline="Study Finds Minimum Wage Increase Boosts Worker Earnings",
        body=(
            "A new study from the Economic Policy Institute found that states "
            "which raised their minimum wage saw average worker earnings "
            "increase by 7% over two years. Workers in the service and retail "
            "sectors benefited the most, with some reporting improved quality "
            "of life. The findings add to growing evidence that moderate wage "
            "increases do not significantly harm employment levels. Several "
            "states are now considering similar legislation."
        ),
        expected_slant_direction="left",
        expected_slant_magnitude="moderate",
        expect_loaded_language=False,  # Language is factual
        expect_omission_risk=True,  # Omits counter-studies, business impact, EPI's known left lean
        expect_source_imbalance=True,
        notes="Key test: factual language but selective omission. Only cites one "
              "study from a left-leaning think tank. Omits studies showing job "
              "losses, small business impact, or counter-evidence. This is bias "
              "through selection, not language.",
        requires_strong_model=True,  # gpt-4o-mini detects direction but underscores omission
    ),
    GoldCase(
        id="subtle_false_balance",
        headline="Scientists and Skeptics Debate Climate Change at Conference",
        body=(
            "A major science conference this week featured panels on climate "
            "change where researchers presented their findings alongside "
            "skeptics who questioned the methodology. Climate scientists said "
            "their models show accelerating warming trends. Skeptics countered "
            "that natural variability could explain observed changes. "
            "Attendees heard both perspectives and formed their own conclusions."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="mild",
        expect_loaded_language=False,
        expect_omission_risk=True,  # Omits the overwhelming consensus
        expect_source_imbalance=False,  # Both "sides" get space - that's the problem
        notes="Key test: false balance. Gives equal weight to overwhelming "
              "scientific consensus and fringe skepticism. The 'balance' itself "
              "is the bias - it implies the debate is 50/50 when it's 97/3.",
        requires_strong_model=True,  # gpt-4o-mini treats "both sides presented" as neutral
        requires_frontier_model=True,
        expect_low_consensus_preservation=True,
    ),
    GoldCase(
        id="subtle_neutral_wording_biased_framing",
        headline="Government Expands Social Program Despite Budget Concerns",
        body=(
            "The federal government announced an expansion of the social "
            "safety net program despite ongoing concerns about the national "
            "deficit. Officials said the program will serve an additional "
            "2 million households. Budget analysts noted the $50 billion "
            "annual cost will need to be offset by revenue increases or "
            "spending cuts elsewhere. The program has historically enjoyed "
            "bipartisan support, though fiscal conservatives have raised "
            "questions about its growing scope."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="mild",
        expect_loaded_language=False,  # Language is neutral
        expect_omission_risk=False,
        expect_source_imbalance=False,
        notes="Key test: neutral language but 'despite' framing. The word 'despite' "
              "frames budget concerns as the expected/default position. A truly neutral "
              "framing might say 'amid' instead. Subtle structural bias.",
        requires_strong_model=True,  # gpt-4o-mini misses single-word framing cues like "despite"
        requires_frontier_model=True,
    ),

    # ---- EDGE CASES ----
    GoldCase(
        id="edge_empty_hedging",
        headline="Experts Weigh In On Policy",
        body=(
            "Some experts say the policy is good. However, other experts "
            "disagree. On the other hand, there are varying perspectives. "
            "Critics say it could go either way. While some support it, "
            "others believe it needs changes. The debate continues."
        ),
        expected_slant_direction="neutral",
        expected_slant_magnitude="none",
        expect_loaded_language=False,
        expect_omission_risk=True,  # Says nothing substantive
        expect_source_imbalance=False,
        notes="High ambivalent word ratio but zero substance. Tests whether "
              "the evaluator can distinguish genuine balance from empty hedging.",
    ),
    GoldCase(
        id="edge_both_sides_inflammatory",
        headline="Political Extremists on Both Sides Threaten Democratic Norms",
        body=(
            "Far-left radicals and far-right extremists are both undermining "
            "democratic institutions, according to a new bipartisan report. "
            "Left-wing agitators have vandalized government buildings while "
            "right-wing militants have threatened election officials. Both "
            "factions promote conspiracy theories and reject compromise. "
            "Moderate voices from both parties condemned the escalation."
        ),
        expected_slant_direction="neutral",
        expected_slant_magnitude="none",
        expect_loaded_language=True,  # Inflammatory but symmetric
        expect_omission_risk=False,
        expect_source_imbalance=False,
        notes="Loaded language applied equally to both sides. Tests whether "
              "the evaluator distinguishes symmetric loaded language from "
              "asymmetric bias.",
    ),

    # ---- FALSE BALANCE (5 cases) ----
    GoldCase(
        id="false_balance_vaccines",
        headline="Doctors and Skeptics Weigh In on Childhood Vaccine Safety",
        body=(
            "Medical professionals and vaccine skeptics continue to debate the "
            "safety of routine childhood immunizations. Pediatricians point to "
            "decades of research showing vaccines are safe and effective. However, "
            "some parents and alternative health advocates raise concerns about "
            "potential side effects and question whether the vaccination schedule "
            "is too aggressive. Both groups say they have children's best "
            "interests at heart."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="mild",
        expect_loaded_language=False,
        expect_omission_risk=True,
        expect_source_imbalance=False,
        notes="False balance: equates overwhelming scientific consensus with "
              "fringe anti-vax position. Omits that 'concerns' are not supported "
              "by evidence. Framing implies a legitimate 50/50 debate.",
        requires_strong_model=True,
        requires_frontier_model=True,
    ),
    GoldCase(
        id="false_balance_evolution",
        headline="Schools Navigate Debate Over Teaching Evolution and Alternatives",
        body=(
            "School districts across several states are revisiting their science "
            "curricula as proponents of evolutionary biology and advocates for "
            "intelligent design both seek representation in classrooms. "
            "Biologists argue evolution is supported by extensive fossil and "
            "genetic evidence. Intelligent design supporters say students "
            "deserve exposure to alternative explanations for the diversity "
            "of life. Education boards are weighing both perspectives."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="mild",
        expect_loaded_language=False,
        expect_omission_risk=True,
        expect_source_imbalance=False,
        notes="False balance: treats settled science as one side of a debate. "
              "Omits that intelligent design is not a scientific theory.",
        requires_strong_model=True,
        requires_frontier_model=True,
        expect_low_consensus_preservation=True,
    ),
    GoldCase(
        id="false_balance_election_fraud",
        headline="Election Officials and Critics Spar Over Voting Integrity",
        body=(
            "State election officials have certified the results of the recent "
            "election, declaring the process free and fair. However, a vocal "
            "group of critics continues to allege widespread irregularities, "
            "pointing to anecdotal reports and statistical anomalies they say "
            "warrant further investigation. Officials counter that multiple "
            "audits have found no evidence of systemic fraud. The debate shows "
            "no signs of abating."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="mild",
        expect_loaded_language=False,
        expect_omission_risk=True,
        expect_source_imbalance=False,
        notes="False balance: presents certified results and unsubstantiated "
              "fraud claims as equivalent 'sides.' Omits that audits already "
              "disproved the claims.",
        requires_strong_model=True,
        requires_frontier_model=True,
        expect_low_consensus_preservation=True,
    ),
    GoldCase(
        id="false_balance_flat_earth",
        headline="Geography Conference Features Debate Between Globe and Flat Earth Models",
        body=(
            "An academic geography conference this year included a panel "
            "discussing alternative models of the earth's shape. Traditional "
            "geographers presented satellite imagery and centuries of "
            "scientific evidence. Flat earth proponents argued that observable "
            "phenomena such as the apparent flatness of horizons support their "
            "model. Audience members were encouraged to evaluate the evidence "
            "for themselves."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="moderate",
        expect_loaded_language=False,
        expect_omission_risk=True,
        expect_source_imbalance=False,
        notes="Extreme false balance. Treats established science as debatable. "
              "Even using 'traditional geographers' vs 'proponents' creates "
              "false equivalence.",
        requires_strong_model=True,
        requires_frontier_model=True,
        expect_low_consensus_preservation=True,
    ),
    GoldCase(
        id="false_balance_tobacco",
        headline="Health Experts and Industry Representatives Discuss Smoking Risks",
        body=(
            "A public health forum brought together medical researchers and "
            "tobacco industry representatives to discuss the health effects of "
            "smoking. Researchers presented data linking smoking to lung cancer, "
            "heart disease, and premature death. Industry representatives "
            "acknowledged some risks but argued that moderate smoking is a "
            "personal choice and that recent products have reduced harm. "
            "Attendees heard both sides of the discussion."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="mild",
        expect_loaded_language=False,
        expect_omission_risk=True,
        expect_source_imbalance=False,
        notes="False balance: lets tobacco industry frame deadly product as "
              "'personal choice' alongside clear medical evidence. Classic "
              "manufactured doubt technique.",
        requires_strong_model=True,
        requires_frontier_model=True,
        expect_low_consensus_preservation=True,
    ),

    # ---- ASYMMETRIC LABELING (4 cases) ----
    GoldCase(
        id="asymmetric_regime_vs_government",
        headline="U.S. Government Responds to Iranian Regime's Provocations",
        body=(
            "The U.S. government issued a measured diplomatic response to "
            "the Iranian regime's latest military provocations in the Gulf. "
            "Government officials emphasized their commitment to stability "
            "while the regime continued its aggressive posturing. Allied "
            "governments expressed support for the U.S. position, while "
            "the regime's foreign ministry dismissed the criticism."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="moderate",
        expect_loaded_language=True,
        expect_omission_risk=False,
        expect_source_imbalance=True,
        notes="Asymmetric labeling: 'government' for the U.S. (legitimate), "
              "'regime' for Iran (illegitimate). Same word choice applied "
              "differently based on geopolitical alignment.",
    ),
    GoldCase(
        id="asymmetric_spending_vs_investment",
        headline="Democrats Propose Investment in Infrastructure as Republicans Decry Spending",
        body=(
            "Congressional Democrats unveiled a $500 billion investment package "
            "for roads, bridges, and broadband infrastructure. Republicans "
            "criticized the spending plan as fiscally irresponsible, arguing "
            "it would add to the national debt. Democrats said the investments "
            "would create millions of jobs. Republicans countered that the "
            "spending would fuel inflation."
        ),
        expected_slant_direction="left",
        expected_slant_magnitude="mild",
        expect_loaded_language=True,
        expect_omission_risk=False,
        expect_source_imbalance=False,
        notes="Same policy called 'investment' (positive) for one party and "
              "'spending' (negative) for the other. Subtle but directional.",
        requires_frontier_model=True,
    ),
    GoldCase(
        id="asymmetric_illegal_vs_undocumented",
        headline="Officials Debate Policy on Illegal Aliens in Border States",
        body=(
            "Federal officials are debating new policies to address the growing "
            "number of illegal aliens crossing the southern border. Law "
            "enforcement agencies report a surge in illegal crossings, while "
            "advocacy groups argue that many of these illegal aliens are "
            "fleeing violence and deserve humanitarian consideration. "
            "The administration says it is working to balance border security "
            "with humane treatment."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="moderate",
        expect_loaded_language=True,
        expect_omission_risk=False,
        expect_source_imbalance=False,
        notes="'Illegal aliens' is a politically charged term. Neutral "
              "reporting would use 'undocumented immigrants' or alternate "
              "between framings. Consistent use of one signals editorial lean.",
    ),
    GoldCase(
        id="asymmetric_protestors_vs_mob",
        headline="Protestors Rally at Capitol; Police Describe Scene as Mob",
        body=(
            "Thousands of protestors gathered at the state capitol to demand "
            "action on police reform. Organizers described the event as a "
            "peaceful demonstration of civic engagement. Police officials, "
            "however, characterized the crowd as a mob, citing isolated "
            "incidents of property damage. Attendees disputed this framing, "
            "saying the vast majority were peaceful."
        ),
        expected_slant_direction="neutral",
        expected_slant_magnitude="mild",
        expect_loaded_language=True,
        expect_omission_risk=False,
        expect_source_imbalance=False,
        notes="Headline uses 'protestors' but includes police calling them a "
              "'mob.' Tests whether the evaluator catches that presenting both "
              "labels doesn't make the framing neutral -- the article's own "
              "voice uses the more sympathetic term.",
    ),

    # ---- SELECTIVE OMISSION (4 cases) ----
    GoldCase(
        id="omission_crime_without_context",
        headline="Violent Crime Surges in Major Cities Across the Country",
        body=(
            "Violent crime has risen sharply in several major metropolitan "
            "areas, with homicides up 12% and aggravated assaults increasing "
            "by 8% over the past year. Police departments report being "
            "stretched thin, and residents in affected neighborhoods describe "
            "a growing sense of fear. City officials are calling for "
            "increased funding for law enforcement and community policing "
            "initiatives."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="moderate",
        expect_loaded_language=False,
        expect_omission_risk=True,
        expect_source_imbalance=True,
        notes="Omits that violent crime is still well below 1990s peaks, "
              "that most cities saw decreases, and that socioeconomic factors "
              "drive crime rates. Presents a local spike as a national crisis.",
        requires_frontier_model=True,
    ),
    GoldCase(
        id="omission_economy_cherry_pick",
        headline="Economy Adds 250,000 Jobs in Strongest Month This Year",
        body=(
            "The U.S. economy added 250,000 jobs last month, exceeding "
            "economists' expectations and marking the strongest hiring month "
            "of the year. The unemployment rate held steady at 3.8%. "
            "Administration officials hailed the report as evidence of "
            "effective economic policy. 'American workers are thriving,' "
            "said the Treasury Secretary."
        ),
        expected_slant_direction="left",
        expected_slant_magnitude="moderate",
        expect_loaded_language=False,
        expect_omission_risk=True,
        expect_source_imbalance=True,
        notes="Omits wage stagnation, underemployment, part-time vs full-time "
              "breakdown, and sector concentration. Cherry-picks the positive "
              "headline number without context.",
        requires_frontier_model=True,
    ),
    GoldCase(
        id="omission_policy_cost_only",
        headline="New Social Program to Cost $300 Billion Over Ten Years",
        body=(
            "The Congressional Budget Office estimates the proposed social "
            "program will cost approximately $300 billion over ten years. "
            "The CBO notes the program would require new revenue sources or "
            "offsetting spending cuts. Fiscal hawks in Congress called the "
            "price tag 'staggering' and warned of deficit impacts. The bill "
            "is expected to face a tough path through the Senate."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="moderate",
        expect_loaded_language=True,  # "staggering"
        expect_omission_risk=True,
        expect_source_imbalance=True,
        notes="Cites CBO cost but omits CBO's benefit projections (e.g., "
              "reduced poverty, healthcare savings). Only quotes opponents. "
              "Frames the policy purely as a fiscal burden.",
        requires_frontier_model=True,
    ),
    GoldCase(
        id="omission_foreign_policy_no_history",
        headline="U.S. Imposes Sanctions on Nation After Escalating Tensions",
        body=(
            "The United States announced a new round of economic sanctions "
            "against the country following escalating tensions in the region. "
            "State Department officials said the sanctions target military "
            "officials and key economic sectors. The sanctioned government "
            "condemned the action as economic warfare and threatened "
            "retaliation. Regional allies expressed support for the U.S. "
            "position."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="mild",
        expect_loaded_language=False,
        expect_omission_risk=True,
        expect_source_imbalance=True,
        notes="Omits U.S. role in preceding events, history of the conflict, "
              "impact on civilian populations, and dissenting views from "
              "within the U.S. Presents sanctions as a response without context.",
        requires_frontier_model=True,
    ),

    # ---- SUBTLE FRAMING (4 cases) ----
    GoldCase(
        id="framing_passive_voice_hides_agency",
        headline="Mistakes Were Made in Handling of Water Contamination Crisis",
        body=(
            "Residents of the affected area continue to deal with the "
            "consequences of the water contamination crisis. It has been "
            "acknowledged that mistakes were made in the regulatory process "
            "that allowed pollutants to enter the water supply. Cleanup "
            "efforts are underway, and affected families are being offered "
            "temporary housing. An investigation has been launched to "
            "determine what went wrong."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="mild",
        expect_loaded_language=False,
        expect_omission_risk=True,
        expect_source_imbalance=False,
        notes="Systematic passive voice hides who made the mistakes, who "
              "acknowledged them, and who launched the investigation. "
              "Protects the responsible party by removing agency.",
        requires_strong_model=True,
        requires_frontier_model=True,
    ),
    GoldCase(
        id="framing_quote_asymmetry",
        headline="Governor Signs Controversial Education Bill Into Law",
        body=(
            "The governor signed the education reform bill into law Tuesday, "
            "calling it 'a historic step forward for our children's future "
            "and a testament to what we can achieve when we put students "
            "first.' Critics of the bill responded that they were "
            "'disappointed' and would 'review their options.' Teachers' "
            "unions said they had 'concerns' about implementation. The bill "
            "takes effect next school year."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="mild",
        expect_loaded_language=False,
        expect_omission_risk=False,
        expect_source_imbalance=True,
        notes="Quote asymmetry: supporters get a full, eloquent quote. "
              "Critics get single-word reactions ('disappointed,' 'concerns'). "
              "This makes one side sound substantive and the other reactive.",
        requires_strong_model=True,
        requires_frontier_model=True,
    ),
    GoldCase(
        id="framing_despite_construction_left",
        headline="Tax Cuts Pass Despite Warnings of Deficit Impact",
        body=(
            "Congress passed the tax cut package despite warnings from "
            "economists about its impact on the federal deficit. The bill "
            "reduces rates across all income brackets, with the largest "
            "percentage cuts going to upper-income earners. Supporters say "
            "the cuts will stimulate economic growth. The CBO projects the "
            "legislation will add $1.5 trillion to the deficit over ten years."
        ),
        expected_slant_direction="left",
        expected_slant_magnitude="mild",
        expect_loaded_language=False,
        expect_omission_risk=False,
        expect_source_imbalance=False,
        notes="Mirror of the 'despite' framing case but leaning left. "
              "'Despite warnings' presupposes the warnings should have "
              "stopped the action. Tests directional consistency.",
        requires_strong_model=True,
        requires_frontier_model=True,
    ),
    GoldCase(
        id="framing_presupposed_conclusion",
        headline="City Struggles to Cope With Influx of Migrants",
        body=(
            "The city is struggling to accommodate a growing number of "
            "migrants arriving from the southern border. Social services are "
            "strained, shelters are at capacity, and city officials say they "
            "need federal assistance. Community organizations are working to "
            "provide food and clothing. Some residents have expressed "
            "frustration, while others have volunteered to help."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="mild",
        expect_loaded_language=True,  # "struggles", "influx", "strained"
        expect_omission_risk=False,
        expect_source_imbalance=False,
        notes="'Struggles' and 'influx' frame migration as inherently "
              "problematic. A neutral framing might say 'adapts to' or "
              "'receives.' The premise of the article is that migrants "
              "are a burden, not that a city is receiving new residents.",
        requires_frontier_model=True,
    ),

    # ---- EPISTEMIC ACCURACY (4 cases) ----
    GoldCase(
        id="epistemic_narrator_laundering",
        headline="New Study Links Cellphone Use to Health Risks",
        body=(
            "Cellphone radiation causes cellular damage that can lead to "
            "cancer, according to emerging research. The electromagnetic "
            "fields emitted by smartphones disrupt normal cell function, "
            "particularly in children whose developing brains are more "
            "susceptible. Health agencies have been slow to update their "
            "guidelines despite mounting evidence. Experts recommend "
            "limiting screen time and using speakerphone when possible."
        ),
        expected_slant_direction="neutral",
        expected_slant_magnitude="mild",
        expect_loaded_language=False,
        expect_omission_risk=True,
        expect_source_imbalance=True,
        notes="Key test: narrator-voice laundering. Disputed claims ('causes "
              "cellular damage,' 'mounting evidence') are stated as fact in "
              "narrator voice rather than attributed to specific sources. "
              "The article presents a contested hypothesis as established truth.",
        expect_low_attribution=True,
    ),
    GoldCase(
        id="epistemic_good_consensus_reporting",
        headline="Global Temperatures Continue to Rise, Climate Scientists Report",
        body=(
            "Global average temperatures increased by 1.2°C above "
            "pre-industrial levels last year, according to data from NASA, "
            "NOAA, and the World Meteorological Organization. The warming "
            "trend is consistent with climate models that project continued "
            "increases driven primarily by greenhouse gas emissions, "
            "according to the IPCC's latest assessment. A small number of "
            "researchers have questioned specific aspects of the modeling "
            "methodology, though the underlying warming trend is not "
            "disputed within the scientific community."
        ),
        expected_slant_direction="neutral",
        expected_slant_magnitude="none",
        expect_loaded_language=False,
        expect_omission_risk=False,
        expect_source_imbalance=False,
        notes="Model for good epistemic reporting. States consensus clearly, "
              "attributes claims to named sources, notes dissent exists without "
              "inflating it. This is what the agent should produce.",
    ),
    GoldCase(
        id="epistemic_unattributed_partisan_claim",
        headline="Tax Plan Will Boost Economy and Create Jobs",
        body=(
            "The proposed tax plan will generate significant economic growth "
            "and create millions of new jobs across all sectors. Lower tax "
            "rates will incentivize business investment and increase consumer "
            "spending. The plan pays for itself through increased economic "
            "activity. Previous tax cuts have consistently led to job growth "
            "and higher government revenue."
        ),
        expected_slant_direction="right",
        expected_slant_magnitude="moderate",
        expect_loaded_language=False,
        expect_omission_risk=True,
        expect_source_imbalance=True,
        notes="Partisan economic claims presented as established fact with no "
              "attribution. 'Tax cuts pay for themselves' is a contested claim, "
              "not established economics. No sources cited, no dissent noted.",
        expect_low_attribution=True,
    ),
    GoldCase(
        id="epistemic_both_sides_but_accurate",
        headline="Economists Divided on Impact of Proposed Tariffs",
        body=(
            "The proposed tariffs have drawn mixed reactions from economists. "
            "Proponents, including several administration advisors, argue the "
            "tariffs will protect domestic manufacturing and reduce the trade "
            "deficit. Critics, including economists at the Federal Reserve "
            "and the Peterson Institute, warn the tariffs could raise consumer "
            "prices by an estimated 2-4% and risk retaliatory measures from "
            "trading partners. A recent Brookings analysis found that "
            "historical tariffs have produced mixed results, with short-term "
            "industry protection often offset by long-term efficiency losses."
        ),
        expected_slant_direction="neutral",
        expected_slant_magnitude="none",
        expect_loaded_language=False,
        expect_omission_risk=False,
        expect_source_imbalance=False,
        notes="Both sides with accurate epistemics: properly attributed, "
              "specific sources named, quantified claims, third-party evidence "
              "cited. This is genuine balance, not false equivalence.",
    ),
]
