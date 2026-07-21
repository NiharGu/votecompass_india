"""
axes.py — VoteCompass axis definitions.

Each axis has:
  - id:           short machine-readable key
  - label:        human-readable axis name
  - left_label:   name for the -1.0 pole
  - right_label:  name for the +1.0 pole
  - left_pole:    description sentence(s) to encode as the -1.0 embedding
  - right_pole:   description sentence(s) to encode as the +1.0 embedding
  - topics:       BERTopic topic IDs (from nr_topics=20 run) whose chunks
                  are relevant to this axis
  - keywords:     fallback keyword filter for chunks not captured by topics

POLE WRITING PRINCIPLE:
  Poles describe a policy position or governing philosophy, not a party stance.
  Specific enough to produce variance, neutral enough not to pre-judge outcomes.
  Policy names used only when describing the concept neutrally from both sides.
  No verdict language ("must be repealed", "is a betrayal", "is unconstitutional").
"""

AXES = [
    {
        "id": "agriculture",
        "label": "Agriculture & Food Security",
        "left_label": "State-guaranteed farming",
        "right_label": "Market-integrated farming",
        "left_pole": (
            "The state has a primary responsibility to protect farmers from market volatility "
            "by guaranteeing a minimum price for their crops and procuring directly from them "
            "through government agencies at that price. "
            "Inputs like fertiliser, seeds, and irrigation electricity should be subsidised "
            "or provided free to reduce the cost burden on small farmers. "
            "Crop insurance schemes should be universal and premium-free for farmers "
            "with fast claims settlement funded by the government. "
            "Regulated wholesale markets protect farmers from exploitation by intermediaries "
            "and should be strengthened. "
            "The public food distribution system should be expanded to guarantee "
            "food security for the rural poor."
        ),
        "right_pole": (
            "Farmers earn better returns when they can sell their produce freely "
            "to any buyer — private traders, food processors, or exporters — "
            "rather than being restricted to selling through government channels. "
            "Private investment in storage, cold chains, and food processing "
            "reduces post-harvest losses and connects farmers to larger markets. "
            "Allowing corporations and farmers to enter into direct price agreements "
            "before harvest reduces income uncertainty more predictably than price support. "
            "Replacing input subsidies with direct cash transfers gives farmers "
            "freedom to make their own production and investment decisions. "
            "Agricultural growth comes from technology adoption, mechanisation, "
            "and market integration rather than price guarantees alone."
        ),
        "left_pole_nli": (
            "The government should guarantee a minimum support price for all crops "
            "and procure grain directly from farmers through state agencies."
        ),
        "right_pole_nli": (
            "Farmers should be free to sell their produce to private buyers "
            "at market-determined prices rather than through government channels."
        ),
        "topics": [15],
        "keywords": ["msp", "procurement", "apmc", "crop insurance", "pds",
                     "food security", "agri", "farmer", "agriculture", "kisan",
                     "loan waiver", "pm kisan", "fasal bima", "mandi"],
    },
    {
        "id": "economy",
        "label": "Economic Development",
        "left_label": "Redistributive growth",
        "right_label": "Liberalised growth",
        "left_pole": (
            "The primary measure of economic success is whether living standards "
            "of the poorest half of the population are improving, not aggregate GDP. "
            "Tax policy should be progressive — higher earners and larger corporations "
            "should contribute proportionally more, and those revenues should fund "
            "public services, rural employment programmes, and social protection. "
            "The government has a responsibility to directly create employment "
            "through public works when the private sector cannot absorb the workforce. "
            "Reducing inequality between regions, castes, and income groups "
            "is as important a policy goal as economic growth itself."
        ),
        "right_pole": (
            "The most effective way to improve living standards at scale is to "
            "create conditions for rapid private sector growth — stable macroeconomics, "
            "low regulatory burden, and incentives that attract domestic and foreign investment. "
            "Manufacturing-led growth through export competitiveness creates "
            "far more jobs than government spending programmes can sustain. "
            "Fiscal discipline — keeping deficits low and debt manageable — "
            "builds investor confidence and keeps inflation under control. "
            "Integration with global supply chains and trade liberalisation "
            "accelerates technology transfer and productivity growth."
        ),
        "left_pole_nli": (
            "Government policy should prioritise reducing inequality and funding "
            "public welfare and rural employment over aggregate GDP growth."
        ),
        "right_pole_nli": (
            "Economic growth requires private sector investment, low regulation, "
            "and export-led manufacturing competitiveness."
        ),
        "topics": [1, 3],
        "keywords": ["gdp", "fdi", "pli", "make in india", "viksit bharat",
                     "wealth tax", "inequality", "redistribution", "ease of doing business",
                     "manufacturing", "global", "bharat", "mgnrega", "employment guarantee"],
    },
    {
        "id": "public_sector",
        "label": "Public Sector vs Privatisation",
        "left_label": "Public sector expansion",
        "right_label": "Privatisation & liberalisation",
        "left_pole": (
            "Government ownership of banks, insurance companies, energy utilities, "
            "and transport networks ensures that essential services reach all citizens "
            "including those in rural and low-income areas that private companies "
            "would not find profitable to serve. "
            "Public enterprises generate revenues that remain in state hands "
            "and can be reinvested in public welfare rather than extracted as shareholder profit. "
            "Workers in public sector organisations have more stable employment "
            "and better protections than those in privatised equivalents. "
            "Retirement benefits for long-serving government employees should be "
            "a guaranteed fixed amount defined by the state, not subject to market fluctuations."
        ),
        "right_pole": (
            "Government ownership of commercial enterprises leads to inefficiency, "
            "overstaffing, and poor service quality when there is no competitive pressure. "
            "Selling government stakes in non-strategic companies raises funds "
            "that can be redirected to healthcare, education, and infrastructure. "
            "Private competition in banking, insurance, telecommunications, and aviation "
            "has consistently lowered prices and improved quality for consumers. "
            "Defined contribution retirement funds where employees and the state "
            "both contribute are more fiscally sustainable over the long term "
            "than open-ended state-guaranteed commitments."
        ),
        "left_pole_nli": (
            "Strategic public sector enterprises including banks and utilities "
            "should not be privatised and government ownership should be maintained."
        ),
        "right_pole_nli": (
            "The government should sell its stakes in non-strategic companies "
            "and allow private competition in banking, insurance, and energy."
        ),
        "topics": [9],
        "keywords": ["disinvestment", "psu", "privatisation", "nationalisation",
                     "lic", "public sector bank", "ppp", "disinvest", "ops", "nps",
                     "strategic sale", "defined benefit", "defined contribution"],
    },
    {
        "id": "labour",
        "label": "Labour Rights & Workers",
        "left_label": "Labour protection",
        "right_label": "Labour flexibility",
        "left_pole": (
            "Workers should have legal protection against dismissal without due process "
            "regardless of the size of the enterprise they work for. "
            "A national minimum wage should be set at a level sufficient to cover "
            "basic living costs and revised regularly to keep pace with inflation. "
            "Workers in all categories — permanent, contractual, seasonal, and platform-based — "
            "should have access to provident fund, health insurance, and injury compensation. "
            "The right to form unions, bargain collectively with employers, "
            "and withdraw labour as a last resort should be protected in law. "
            "Retirement income for long-serving public employees should be a guaranteed "
            "fixed amount, not a fund dependent on market performance."
        ),
        "right_pole": (
            "Overly rigid employment laws discourage firms from hiring formally "
            "because they cannot adjust their workforce when business conditions change. "
            "A simplified and unified labour regulatory framework reduces compliance costs "
            "for small and medium businesses and encourages formal employment creation. "
            "Time-limited employment contracts give businesses flexibility to scale "
            "and allow workers to gain experience across multiple employers. "
            "New forms of work — digital platforms, gig economy, freelancing — "
            "provide genuine income flexibility and should be regulated differently "
            "from traditional employment rather than forced into the same framework."
        ),
        "left_pole_nli": (
            "Workers should have strong legal protections including a minimum wage, "
            "the right to unionise, and a guaranteed state pension."
        ),
        "right_pole_nli": (
            "Labour laws should be simplified to give employers flexibility "
            "to hire and restructure without excessive legal constraints."
        ),
        "topics": [10, 16],
        "keywords": ["labour code", "minimum wage", "mgnrega", "ops", "old pension",
                     "collective bargaining", "gig worker", "contract labour",
                     "social security", "worker", "union", "fixed term", "esi", "epf"],
    },
    {
        "id": "healthcare",
        "label": "Healthcare",
        "left_label": "Universal public health",
        "right_label": "Insurance-based healthcare",
        "left_pole": (
            "The government should directly build, staff, and fund a comprehensive network "
            "of hospitals, clinics, and health centres that provide free treatment "
            "to all citizens regardless of income or location. "
            "Essential medicines should be procured by the state and dispensed free "
            "or at cost price through government facilities. "
            "Public health spending should be increased substantially as a share of GDP "
            "because investment in prevention, vaccination, sanitation, and nutrition "
            "reduces the overall burden of disease more cost-effectively than curative care. "
            "Healthcare quality should not depend on a patient's ability to pay."
        ),
        "right_pole": (
            "Funding citizens to access treatment through insurance schemes covering "
            "both public and private hospitals reaches more people faster than "
            "the government trying to directly build and run all health facilities. "
            "Private hospitals, diagnostic centres, and pharmaceutical companies "
            "invest in medical technology and capacity that the public sector "
            "cannot match in speed or scale. "
            "Digital health platforms and telemedicine can extend quality care "
            "to remote populations at a fraction of the cost of physical infrastructure. "
            "A mixed system where government funds access and private providers "
            "deliver services can combine the reach of public finance "
            "with the efficiency of private management."
        ),
        "left_pole_nli": (
            "The government should directly build and fund hospitals providing "
            "free healthcare to all citizens regardless of income."
        ),
        "right_pole_nli": (
            "Government health insurance allowing treatment at both public and "
            "private hospitals reaches more people than state-run hospitals alone."
        ),
        "topics": [11],
        "keywords": ["ayushman", "aiims", "health insurance", "generic medicine",
                     "public hospital", "private hospital", "universal healthcare",
                     "medical", "healthcare", "doctor", "hospital", "health budget",
                     "jan aushadhi", "pm abhim"],
    },
    {
        "id": "women",
        "label": "Women & Gender Equity",
        "left_label": "Structural transformation",
        "right_label": "Welfare-based approach",
        "left_pole": (
            "Achieving gender equality requires changing the structural conditions "
            "that keep women economically and politically subordinate — "
            "reserving a fixed share of elected seats for women, enforcing equal pay "
            "through legislation, reforming property and inheritance laws, "
            "and ensuring that violence against women is prosecuted firmly. "
            "Women's access to education, credit, and formal employment "
            "should be guaranteed through binding policy rather than voluntary schemes. "
            "The legal rights of all individuals regardless of gender identity "
            "or sexual orientation should be explicitly recognised and protected."
        ),
        "right_pole": (
            "Women's lives improve most concretely through targeted government schemes "
            "that address their immediate material circumstances — "
            "access to cooking fuel, safe sanitation, clean water, maternal health services, "
            "and housing in their name. "
            "Organising women into self-help groups and providing access to microfinance "
            "builds lasting economic independence especially in rural areas "
            "more durably than top-down legislative interventions. "
            "Community-level support systems, better policing responsiveness, "
            "and accessible helplines protect women's safety more practically "
            "than changes to sentencing policy alone."
        ),
        "left_pole_nli": (
            "Women's equality requires structural reforms including reserved "
            "parliamentary seats, equal pay legislation, and property rights."
        ),
        "right_pole_nli": (
            "Women's empowerment is best achieved through targeted welfare schemes "
            "for fuel, sanitation, housing, and maternal health."
        ),
        "topics": [5],
        "keywords": ["women reservation", "33 percent", "equal pay",
                     "domestic violence", "shg", "maternity", "beti bachao",
                     "lgbtq", "gender", "women empowerment", "mahila",
                     "nari shakti", "lakhpati didi"],
    },
    {
        "id": "defence",
        "label": "Defence & Foreign Policy",
        "left_label": "Diplomacy-first / strategic autonomy",
        "right_label": "Security-first / strong deterrence",
        "left_pole": (
            "India should maintain an independent foreign policy that avoids "
            "alignment with any major power bloc and positions itself as a "
            "bridge between competing interests in global disputes. "
            "Territorial and border disputes with neighbouring countries should be "
            "resolved through sustained bilateral diplomacy and multilateral frameworks "
            "rather than military escalation. "
            "Soldiers who make a full career commitment to national defence "
            "deserve permanent employment status and a guaranteed retirement pension "
            "proportional to their years of service. "
            "India's international standing grows through development partnerships, "
            "multilateral leadership, and soft power projection."
        ),
        "right_pole": (
            "India's security environment with active border tensions on multiple fronts "
            "requires a modern well-funded military with advanced capabilities, "
            "joint theatre commands, and a large trained reserve that can be mobilised rapidly. "
            "Developing domestic defence manufacturing reduces strategic vulnerability "
            "to foreign suppliers and builds industrial and technological capacity. "
            "Border infrastructure — roads, tunnels, forward airstrips — "
            "is as important as weapons systems in deterring incursion. "
            "Short-term military service models build a larger trained reserve "
            "while keeping the permanent defence establishment lean and affordable."
        ),
        "left_pole_nli": (
            "India should pursue strategic autonomy and resolve border disputes "
            "through sustained diplomacy rather than military escalation."
        ),
        "right_pole_nli": (
            "India needs increased defence spending, domestic arms manufacturing, "
            "and a strong military deterrence to counter border threats from China and Pakistan."
        ),
        "topics": [7],
        "keywords": ["orop", "surgical strike", "border security",
                     "pakistan", "china", "strategic autonomy", "un", "saarc",
                     "defence", "military", "armed forces", "quad",
                     "theatre command", "indigenisation", "one rank one pension",
                     "short term service", "reserve force"],
    },
    {
        "id": "environment",
        "label": "Environment & Urban Development",
        "left_label": "Ecological priority",
        "right_label": "Development priority",
        "left_pole": (
            "Protecting forests, rivers, wetlands, and biodiversity should be treated "
            "as a primary obligation of the state, not a secondary consideration "
            "to be traded away when it conflicts with economic projects. "
            "Independent environmental review processes should be rigorous and binding "
            "before any infrastructure, mining, or industrial project is approved. "
            "Communities living in and around forests and tribal lands have prior rights "
            "over those resources that must be respected before any development proceeds. "
            "India should commit to a clear timeline for transitioning away from fossil fuels "
            "and invest heavily in renewable energy to meet climate obligations."
        ),
        "right_pole": (
            "As a developing country with hundreds of millions still in poverty "
            "India cannot prioritise environmental protection to the same degree "
            "as already-industrialised wealthy nations that created most historical emissions. "
            "Environmental review processes that take years delay roads, railways, "
            "dams, and power plants that millions of Indians urgently need. "
            "Technology-driven approaches — cleaner production methods, electric vehicles, "
            "efficient irrigation — allow development and environmental improvement "
            "to proceed together rather than treating them as opposites. "
            "Planned urban development and efficient public transport reduce pollution "
            "without restricting economic activity."
        ),
        "left_pole_nli": (
            "Environmental protection should be a binding obligation of the state "
            "with rigorous independent review before any infrastructure or mining project."
        ),
        "right_pole_nli": (
            "India's development needs require efficient environmental clearance "
            "without lengthy delays to critical infrastructure projects."
        ),
        "topics": [4],
        "keywords": ["climate change", "net zero", "renewable energy", "deforestation",
                     "smart city", "highway", "dam", "pollution", "eia", "solar",
                     "environment", "forest", "water", "conservation", "coal",
                     "forest rights", "tribal land", "green energy"],
    },
    {
        "id": "federalism",
        "label": "Federalism & State Autonomy",
        "left_label": "Strong federalism",
        "right_label": "Centralisation",
        "left_pole": (
            "States have constitutional sovereignty over subjects within their domain "
            "and the central government should not routinely legislate over state matters "
            "or use financial transfers as leverage to impose central preferences. "
            "A substantially larger share of centrally collected taxes should be "
            "transferred to states so they can fund services according to local priorities. "
            "Regional languages have equal standing with any national language "
            "and their use in education, courts, and administration should be protected. "
            "Appointed officials should not interfere in the functioning "
            "of democratically elected state governments."
        ),
        "right_pole": (
            "A strong central government is necessary to ensure comparable standards "
            "of welfare, education, justice, and security across all states. "
            "Uniform national policies create predictability for businesses operating "
            "across state borders and reduce the cost of compliance with multiple systems. "
            "A shared language of national communication facilitates economic mobility "
            "and reduces barriers to employment and trade across diverse states. "
            "Holding elections at all levels simultaneously reduces the frequency "
            "of electoral cycles that interrupt governance. "
            "National coordination on taxation, civil law, and regulatory standards "
            "produces better outcomes than fragmented state-by-state approaches."
        ),
        "left_pole_nli": (
            "States should have greater fiscal autonomy and receive a larger "
            "share of centrally collected taxes to fund services according to local needs."
        ),
        "right_pole_nli": (
            "A strong central government ensures uniform national standards "
            "and better coordination than fragmented state-by-state approaches."
        ),
        "topics": [2],
        "keywords": ["gst devolution", "governor", "concurrent list", "onoe",
                     "one nation one language", "delimitation", "fiscal federalism",
                     "state rights", "federalism", "devolution", "state autonomy",
                     "finance commission", "language policy"],
    },
    {
        "id": "infrastructure",
        "label": "Industry & Infrastructure",
        "left_label": "MSME & decentralised industry",
        "right_label": "Large-scale infrastructure push",
        "left_pole": (
            "Small and medium enterprises, artisans, traders, and cottage industries "
            "employ the large majority of India's non-farm workforce "
            "and should be the primary focus of industrial and trade policy. "
            "Tax and compliance systems should be simple enough for small businesses "
            "to manage without expensive professional help. "
            "Access to affordable credit without collateral is the single most "
            "important enabler for small enterprise growth. "
            "Traditional crafts, handlooms, and local industries should receive "
            "active policy support to preserve livelihoods and cultural heritage. "
            "Government procurement should prioritise smaller domestic suppliers."
        ),
        "right_pole": (
            "Transforming India into a globally competitive economy requires "
            "large-scale investment in physical infrastructure — railways, ports, "
            "airports, highways, power grids, and industrial corridors. "
            "Incentive schemes for large manufacturing industries in electronics, "
            "pharmaceuticals, textiles, and defence generate export revenues "
            "and formal employment at a scale that small enterprises cannot match. "
            "Industrial zones with ready infrastructure and simplified regulations "
            "attract anchor investments that create supply chain ecosystems "
            "employing many smaller firms around them. "
            "Domestic production of advanced technology goods requires industrial scale "
            "and sustained policy support that only large enterprises can absorb."
        ),
        "left_pole_nli": (
            "Small and medium enterprises and cottage industries should receive "
            "priority in industrial policy over large corporations."
        ),
        "right_pole_nli": (
            "India needs large-scale investment in physical infrastructure and "
            "production incentive schemes for major manufacturing industries."
        ),
        "topics": [8, 17],
        "keywords": ["msme", "small trader", "pli", "railway", "airport", "highway",
                     "freight corridor", "industrial corridor", "make in india",
                     "infrastructure", "port", "construction", "semiconductor",
                     "khadi", "handloom", "gst compliance"],
    },
    {
        "id": "education",
        "label": "Youth: Education, Employment & Skills",
        "left_label": "Public education as right",
        "right_label": "Skill & market alignment",
        "left_pole": (
            "Education at every level from primary school through university "
            "should be fully funded by the state and available free of charge to all citizens. "
            "Government schools and colleges should receive sufficient funding to hire "
            "qualified teachers, maintain adequate facilities, and provide learning materials. "
            "Admissions policies in public educational institutions should include "
            "reservations for historically marginalised communities as a corrective "
            "for structural inequality. "
            "The language of instruction and curriculum content should reflect "
            "the regional and cultural context of students. "
            "Fees in government educational institutions should remain affordable."
        ),
        "right_pole": (
            "The education system should produce graduates equipped "
            "for the skills demanded by a modern technology-driven economy. "
            "Vocational training, apprenticeships, and industry-aligned certification "
            "programmes create more direct pathways to employment than conventional degrees. "
            "Flexibility between academic and skill-based tracks and multidisciplinary "
            "curricula prepare students better for available jobs. "
            "Allowing private and international educational institutions to operate "
            "increases overall capacity, introduces competition, and raises quality standards. "
            "Connecting students to industry through internships and placement programmes "
            "reduces the gap between education and employability."
        ),
        "left_pole_nli": (
            "Education should be fully funded by the state and freely accessible "
            "to all citizens including reservations for marginalised communities."
        ),
        "right_pole_nli": (
            "The education system should align with industry needs through "
            "vocational training and allow private and foreign universities to operate."
        ),
        "topics": [0, 12],
        "keywords": ["nep", "skill development", "iit", "reservation in education",
                     "foreign university", "vocational", "apprenticeship",
                     "education budget", "teacher", "school", "university",
                     "skill india", "pm kaushal vikas", "mid day meal"],
    },
    {
        "id": "social_security",
        "label": "Social Security: Elderly & Disabled",
        "left_label": "Universal state entitlement",
        "right_label": "Targeted / contributory model",
        "left_pole": (
            "Every citizen above a certain age should receive a monthly pension "
            "from the state as a universal right regardless of employment history, "
            "savings, or family circumstances. "
            "The pension amount should be set at a level that covers basic living costs "
            "and revised regularly — current amounts are insufficient for dignified living. "
            "Persons with disabilities should receive unconditional monthly support "
            "and free access to assistive devices and specialised services. "
            "Public sector employees who served for decades deserve retirement income "
            "guaranteed by the state rather than dependent on investment returns. "
            "Social protection is a right of citizenship, not a discretionary benefit."
        ),
        "right_pole": (
            "Universal pension entitlements are fiscally unsustainable at current "
            "government revenue levels and would reduce funds available for "
            "health, education, and infrastructure. "
            "Social security spending should be targeted at those with no other means — "
            "people below the poverty line, the severely disabled, and elderly "
            "with no family or savings. "
            "Retirement funds where workers and employers contribute throughout "
            "a career build personal assets and are more sustainable over time. "
            "Tax incentives for individual savings and private pension products "
            "encourage financial planning and reduce long-term state liability."
        ),
        "left_pole_nli": (
            "Every elderly and disabled citizen should receive a monthly pension "
            "from the state as a universal right regardless of employment history."
        ),
        "right_pole_nli": (
            "Social security spending should be targeted at the poorest citizens "
            "and funded through contributory pension systems rather than universal entitlements."
        ),
        "topics": [14],
        "keywords": ["ops", "nps", "pm shram yogi", "disability allowance",
                     "nsap", "senior citizen", "bpl pension", "old age pension",
                     "pension", "elderly", "disabled", "social protection",
                     "annapurna", "contributory pension"],
    },
    {
        "id": "communal",
        "label": "Communal, Caste & Minority Rights",
        "left_label": "Secular pluralism",
        "right_label": "Majoritarian nationalism",
        "left_pole": (
            "The state must maintain strict neutrality across all religious communities "
            "in law, policy, and public institutions, neither favouring nor disadvantaging "
            "any religion in governance. "
            "Violence targeting people on the basis of religion, caste, or community "
            "should be prosecuted firmly and impartially regardless of which community "
            "the perpetrators belong to. "
            "Religious and linguistic minorities should have the right to maintain "
            "their own educational institutions and practise their cultural and personal laws. "
            "Caste-based discrimination remains a structural reality requiring accurate "
            "measurement of caste demographics across the population to design effective policy. "
            "Citizenship and residency rights should be granted on neutral criteria "
            "not conditional on religious identity."
        ),
        "right_pole": (
            "A country's majority cultural and civilisational heritage can be legitimately "
            "reflected in public institutions, national symbols, and governance "
            "without this constituting discrimination against minorities. "
            "A single civil law applying uniformly to all citizens across all communities "
            "upholds constitutional equality more consistently than separate personal "
            "law systems based on religious identity. "
            "Countries may take into account the specific circumstances of communities "
            "fleeing religious persecution in neighbouring states when designing "
            "immigration and citizenship policy. "
            "Restoration of historical and cultural sites important to the majority community "
            "is a matter of cultural heritage and national identity."
        ),
        "left_pole_nli": (
            "The government must maintain strict secularism and actively protect "
            "the rights of religious minorities, Dalits, and tribal communities."
        ),
        "right_pole_nli": (
            "India's Hindu civilisational heritage can be reflected in governance "
            "and a uniform civil code should apply equally to all citizens."
        ),
        "topics": [18, 13],
        "keywords": ["uniform civil code", "caste census", "obc reservation",
                     "minority rights", "secularism", "hate speech",
                     "communal", "minority", "dalit", "adivasi",
                     "personal law", "citizenship", "religious freedom",
                     "affirmative action", "cultural heritage"],
    },
]

# Quick lookup by axis id
AXES_BY_ID = {a["id"]: a for a in AXES}
AXIS_IDS = [a["id"] for a in AXES]

PARTY_ORDER = ["BJP", "INC", "CPIM", "TMC", "NCPSP", "DMK", "SP", "NCPAP"]


if __name__ == "__main__":
    print(f"{len(AXES)} axes defined:\n")
    for a in AXES:
        print(f"  [{a['id']:18s}]  {a['label']}")
        print(f"    -1.0  {a['left_label']}")
        print(f"    +1.0  {a['right_label']}")
        print(f"    Topics: {a['topics']}  |  Keywords: {len(a['keywords'])}")
        print()