"""Build the UK research funding briefing as a plain .docx.

Every status, amount and date below was checked against the funder's own page
(or, where the funder's site refused an automated request, against search
results quoting it) on 30 September 2026. The CONFIDENCE field on each entry
says which, because a figure taken from an aggregator is not the same thing as
a figure read off the funder's own call document.
"""
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Inches, RGBColor

CHECKED = "30 September 2026"

# --------------------------------------------------------------- document set-up
doc = Document()

st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15

for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11)):
    s = doc.styles[name]
    s.font.name = "Calibri"
    s.font.size = Pt(size)
    s.font.bold = True
    s.font.italic = False
    s.font.color.rgb = RGBColor(0, 0, 0)          # plain black, not Word's blue
    s.paragraph_format.space_before = Pt(14 if size > 12 else 10)
    s.paragraph_format.space_after = Pt(4)

for sec in doc.sections:
    sec.top_margin = sec.bottom_margin = Inches(1)
    sec.left_margin = sec.right_margin = Inches(1)

TEXT_WIDTH = Inches(6.5)


def para(text="", bold=False, italic=False, size=None, space_after=None,
         align=None):
    p = doc.add_paragraph()
    if text:
        r = p.add_run(text)
        r.bold = bold
        r.italic = italic
        if size:
            r.font.size = Pt(size)
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if align:
        p.alignment = align
    return p


def rich(parts, space_after=None):
    """A paragraph from (text, bold, italic) tuples."""
    p = doc.add_paragraph()
    for t in parts:
        text, b, i = (t + (False, False))[:3] if isinstance(t, tuple) else (t, False, False)
        r = p.add_run(text)
        r.bold = b
        r.italic = i
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def bullet(text, level=0):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.25 + 0.25 * level)
    p.paragraph_format.space_after = Pt(3)
    return p


def numbered(text):
    p = doc.add_paragraph(text, style="List Number")
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(3)
    return p


def table(rows, widths, header=True):
    t = doc.add_table(rows=0, cols=len(widths))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, val in enumerate(row):
            cells[ci].width = Inches(widths[ci])
            p = cells[ci].paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(str(val))
            r.font.size = Pt(9.5)
            r.bold = header and ri == 0
    for row in t.rows:                    # python-docx needs widths per cell
        for ci, c in enumerate(row.cells):
            c.width = Inches(widths[ci])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def entry(name, fields):
    """One funding opportunity: bold name then labelled lines."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    p.add_run(name).bold = True
    for label, value in fields:
        q = doc.add_paragraph()
        q.paragraph_format.left_indent = Inches(0.25)
        q.paragraph_format.space_after = Pt(1)
        r = q.add_run(label + ": ")
        r.bold = True
        r.font.size = Pt(10)
        r2 = q.add_run(value)
        r2.font.size = Pt(10)


# ------------------------------------------------------------------ front matter
h = para("External Research Funding Opportunities in the UK", bold=True,
         size=17, space_after=2)
h.alignment = WD_ALIGN_PARAGRAPH.LEFT
para("A briefing mapped to the HandsFree / MAPLE research programme",
     italic=True, size=12, space_after=12)

table([
    ["Prepared for", "Mohammad Anas Shaik, MSc Data Science, "
                     "York St John University (London campus)"],
    ["Research area", "Hands-free human-computer interaction (gaze, blink, head "
                      "pose, voice) for users with motor impairment; and the "
                      "evaluation methodology contributed by the MAPLE paper "
                      "(group-disjoint partitioning, asymmetric-exposure "
                      "leakage, the GERA audit)"],
    ["Purpose", "To identify external funding that this work can realistically "
                "be submitted to, and to be clear about which schemes are and "
                "are not open to someone at MSc stage"],
    ["All entries checked", CHECKED],
], [1.5, 5.0], header=False)

# ------------------------------------------------------------------- section 1
doc.add_heading("1. Read this first: the eligibility rule that governs "
                "everything below", level=1)

para("There is one structural fact about UK research funding that determines "
     "which of the opportunities below you can act on, and it is worth being "
     "blunt about it before the list starts.")

rich([("Almost every UK research grant requires the lead applicant to hold a "
       "salaried academic or research post at an eligible UK research "
       "organisation.", True, False),
      " UKRI states this as an individual eligibility policy, and the research "
      "charities apply their own version of the same rule. A registered "
      "master's student cannot be the Principal Investigator on an EPSRC "
      "grant, an NIHR award, or a Muscular Dystrophy UK project grant, "
      "regardless of the quality of the proposal."])

para("That does not mean there is nothing to apply for. It means the "
     "opportunities split into distinct tracks, and conflating them wastes "
     "months. This document is organised by track:")

table([
    ["Track", "Who applies", "What it gets you"],
    ["1. Your own name, now",
     "You, personally",
     "Travel and conference awards, and a small number of industry "
     "programmes that accept individual researchers. Hundreds to a few "
     "thousand pounds, occasionally more."],
    ["2. Doctoral funding",
     "You, as a PhD applicant",
     "Fees plus stipend for 3.5 to 4 years. For someone at your stage this is "
     "the largest and most realistic single source of external research "
     "funding in the UK."],
    ["3. Supervisor-led grants",
     "Your supervisor, with the university as host",
     "£30,000 to £1,000,000+. You are named on the proposal as researcher or "
     "employed from it. This is almost certainly what your supervisors mean."],
    ["4. Industry and foundation",
     "You or the university",
     "Cash plus compute credits, typically $25,000 upward. Fewer eligibility "
     "barriers than the research councils."],
    ["5. Post-doctoral fellowships",
     "You, after the PhD",
     "£300,000 to £800,000 and a salary. Listed so the sequence is visible; "
     "none is open to you yet."],
], [1.45, 1.6, 3.45])

rich([("A note on what to ask your supervisors. ", True, False),
      "\"Apply for external funding\" from a supervisor usually means Track 3, "
      "where they are the applicant and you are the named researcher. It is "
      "worth confirming which track they have in mind at the outset, because "
      "the work involved is completely different. If they intend Track 3, the "
      "question to put to them is which of them is eligible to be a Principal "
      "Investigator and whether any of them has held a research council grant "
      "before, since that single answer opens or closes the EPSRC New "
      "Investigator Award described in section 4."])

# ------------------------------------------------------------------- section 2
doc.add_heading("2. Priority shortlist", level=1)

para("Ordered by what needs attention soonest, not by size of award.")

table([
    ["#", "Opportunity", "Track", "Next date", "Fit"],
    ["1", "Responsible AI UK — Trustworthy Evaluation of AI Tools in "
          "Science and Engineering", "3",
     "Closes 7 Oct 2026", "Very strong (evaluation methodology)"],
    ["2", "NIHR i4i Product Development Awards", "3",
     "Opens 7 Oct 2026; outline by 1pm 2 Dec 2026", "Strong if framed clinically"],
    ["3", "SIGCHI Gary Marsden Travel Awards", "1",
     "9 Nov 2026 (then 9 Jan 2027)", "Very strong, and you are eligible"],
    ["4", "EPSRC PhD studentships for Oct 2027 entry", "2",
     "Most open Nov 2026 – Feb 2027", "Very strong"],
    ["5", "Microsoft AI for Accessibility", "4",
     "Rolling", "Very strong"],
    ["6", "EPSRC New Investigator Award", "3",
     "No deadline — open continuously", "Strong, if a supervisor qualifies"],
    ["7", "Muscular Dystrophy UK annual grant round", "2 and 3",
     "2026 round closed; next expected around Jan 2027",
     "Strong (motor impairment)"],
    ["8", "MND Association Doctoral Training Award in data science", "2",
     "2026 call — confirm status urgently", "Very strong"],
], [0.32, 1.95, 0.5, 1.6, 1.63])

# ------------------------------------------------------------------- section 3
doc.add_heading("3. Track 1 — What you can apply for in your own name now",
                level=1)

entry("SIGCHI Gary Marsden Travel Awards", [
    ("What it is", "Travel and registration support to attend an ACM SIGCHI "
                   "conference, which for this work means CHI, ASSETS, UIST, "
                   "IUI, ETRA or MobileHCI"),
    ("Why it fits", "The scheme exists specifically for students and "
                    "early-career researchers who lack institutional travel "
                    "funding. A master's student with an accepted paper is the "
                    "intended applicant, not an edge case"),
    ("Eligibility", "Undergraduate and graduate students, and researchers "
                    "within five years of graduating. You must be a SIGCHI "
                    "member at the time of application"),
    ("Covers", "Conference registration plus travel, from two days before the "
               "conference to two days after. Reimbursed, so you need to be "
               "able to pay up front"),
    ("Deadlines", "Rolling, on the 9th of January, February, March, May, "
                  "July, September and November at 11:59pm AoE. You must "
                  "apply to a deadline falling more than one month before the "
                  "conference starts"),
    ("Where", "sigchi.org/resources/gary-marsden-travel-awards/"),
    ("Confidence", "High — read from the SIGCHI page itself"),
])

entry("Conference-specific student grants", [
    ("What it is", "Most ACM conferences run their own student volunteer "
                   "programmes and travel grants, separate from the SIGCHI "
                   "award above, and they are usually undersubscribed"),
    ("Why it fits", "ASSETS (accessible computing) and ETRA (eye tracking) are "
                    "the two venues where the HandsFree work sits most "
                    "naturally, and both run student support"),
    ("Eligibility", "Registered students. Usually no paper required for "
                    "volunteer routes; a paper helps for travel grants"),
    ("Deadlines", "Set per conference, typically six to ten weeks before the "
                  "event. Check the \"Attending\" or \"Student Support\" page "
                  "of each conference site"),
    ("Confidence", "High in general, but each conference publishes its own "
                   "dates and these must be checked individually"),
])

entry("Microsoft AI for Accessibility", [
    ("What it is", "Microsoft's accessibility grant programme, part of a "
                   "$25 million five-year commitment. Awards combine cash, "
                   "Azure compute credits and technical mentoring"),
    ("Why it fits", "This is the closest match to HandsFree of anything in "
                    "this document. The programme names mobility and "
                    "communication accessibility as target areas, and it "
                    "funds exactly the kind of model-based input work you "
                    "have built. Birmingham City University holds an AI for "
                    "Accessibility grant for eye-tracking and voice control "
                    "for programmers, which is adjacent enough to be a "
                    "useful precedent and worth citing"),
    ("Eligibility", "Stated as open to developers, universities, NGOs, "
                    "researchers and inventors in all countries. In practice "
                    "funds are paid to an organisation, so you would apply "
                    "with York St John as the recipient"),
    ("Amount", "Reported as $25,000 to $250,000. Treat this as indicative"),
    ("Deadlines", "Rolling submission"),
    ("Where", "microsoft.com/AIforAccessibility"),
    ("Confidence", "Medium on the amounts, which come from grant aggregators "
                   "rather than Microsoft's own page; high on the programme "
                   "existing and accepting academic applicants"),
])

rich([("One to rule out. ", True, False),
      "The IET Postgraduate Research Awards are frequently recommended to "
      "students in this position and you are not eligible: the IET excludes "
      "students on MSc and MRes courses explicitly, and requires at least one "
      "year of IET membership. Worth joining the IET now if you intend to use "
      "this route during a PhD."])

# ------------------------------------------------------------------- section 4
doc.add_heading("4. Track 2 — Doctoral funding", level=1)

para("If the question is where the substantial external money is for someone "
     "finishing an MSc, this is the answer. A funded PhD place is worth "
     "roughly £90,000 to £130,000 over three and a half to four years once "
     "fees, stipend and research costs are counted, and the HandsFree paper "
     "is a strong application asset: very few applicants arrive with a "
     "complete methodological contribution and a working system.")

entry("EPSRC Doctoral Landscape Awards (formerly Doctoral Training "
      "Partnerships)", [
    ("What it is", "Block funding given to universities, which then advertise "
                   "individual funded projects. Covers fees and an annual "
                   "stipend for up to 3.5 years"),
    ("How to apply", "Project-led. You identify a specific advertised project "
                     "and apply to that university directly with a personal "
                     "statement, rather than applying to EPSRC"),
    ("Timing", "For October 2027 entry, most universities publish projects "
               "between about November 2026 and February 2027. Several state "
               "explicitly that their 2027/28 pages go live in October 2026, "
               "which means now is the right moment to start watching them"),
    ("Where to look", "The doctoral training pages of universities with strong "
                      "HCI or accessibility groups. On topic fit the obvious "
                      "targets are UCL (including the Global Disability "
                      "Innovation Hub and UCL Interaction Centre), the "
                      "University of Dundee, Lancaster, Glasgow, Birmingham "
                      "City University, Nottingham and Bournemouth, all of "
                      "which have named assistive technology or accessibility "
                      "research groups"),
    ("Important caveat", "UKRI studentships have residency rules. "
                         "International candidates are eligible but "
                         "universities are capped on the proportion of "
                         "international students they may fund from these "
                         "awards, which makes those places materially more "
                         "competitive. Establish your fee status before "
                         "investing time"),
    ("Confidence", "High on the mechanism and timing; individual deadlines "
                   "must be read per university"),
])

entry("UKRI AI Centres for Doctoral Training", [
    ("What it is", "Four-year cohort-based PhD programmes in artificial "
                   "intelligence, with integrated training and usually an "
                   "industrial partner"),
    ("Why it fits", "Your leakage and evaluation contribution is an AI "
                    "trustworthiness argument, which is squarely within the "
                    "remit most of these centres describe"),
    ("Timing", "Recruitment cycles vary by centre; most recruit between "
               "December and March for autumn entry"),
    ("Confidence", "Medium — the CDT landscape is reorganising, so confirm "
                   "which centres are recruiting for 2027 entry"),
])

entry("MND Association Doctoral Training Award in data science", [
    ("What it is", "Reported as a first-of-its-kind award of £700,000 to "
                   "£800,000 over five years, funding a single cohort of "
                   "non-clinical four-year PhD students in data science "
                   "applied to motor neurone disease"),
    ("Why it fits", "Strongest disease-area match in this document. Eye-gaze "
                    "and switch-based communication access is central to MND "
                    "care, and a non-clinical data science cohort is precisely "
                    "your profile. If this cohort is being hosted somewhere, "
                    "it is the single most on-topic funded PhD route available"),
    ("Eligibility", "Non-clinical PhD candidates. The award itself goes to an "
                    "institution, so you would apply to the hosting "
                    "university's cohort rather than to the charity"),
    ("Action", "Email research.grants@mndassociation.org and ask which "
               "institution holds the award and when the cohort recruits. "
               "This is a short email with a potentially large payoff"),
    ("Confidence", "Medium — taken from search results quoting the MND "
                   "Association, not from the funder page, which refused an "
                   "automated request. Confirm directly"),
])

entry("Muscular Dystrophy UK — four-year PhD studentships", [
    ("What it is", "Part of the charity's annual grant round. Studentships "
                   "cover fees, stipend and up to £10,000 a year in "
                   "consumables"),
    ("Why it fits", "The round explicitly welcomes projects that improve "
                    "quality of life for people with neuromuscular conditions "
                    "or establish best practice in symptom management. "
                    "Hands-free computer access for people losing upper-limb "
                    "function is a direct quality-of-life intervention"),
    ("Timing", "The 2026 round closed at 3pm on 20 January 2026, so it has "
               "passed. The round is annual, so expect the 2027 call to open "
               "in late 2026 with a deadline around January 2027"),
    ("Note", "The studentship is applied for by a prospective supervisor, not "
             "by the student. You would need an academic willing to submit "
             "with you as the intended candidate"),
    ("Where", "musculardystrophyuk.org/research/for-researchers/"
              "open-grant-calls/"),
    ("Confidence", "Medium — the funder site refused an automated request; "
                   "figures come from search results quoting it"),
])

entry("Motability Foundation PhD scholarships", [
    ("What it is", "Doctoral scholarships, with a stated emphasis on "
                   "candidates who have lived experience of disability"),
    ("Fit", "Weaker than it first appears. The Motability Foundation's remit "
            "is transport and mobility, not computer access. A HandsFree "
            "proposal would only fit if genuinely reframed around powered "
            "wheelchair control, in-vehicle interfaces or journey planning, "
            "and reframing a proposal to chase a funder is usually visible to "
            "reviewers. Listed for completeness rather than recommended"),
    ("Where", "motabilityfoundation.org.uk/research-and-innovation/research/"),
    ("Confidence", "High that the scheme exists; the scholarship page itself "
                   "gave no detail on amounts or deadlines"),
])

entry("Other doctoral routes worth a morning of work", [
    ("University scholarships", "Most universities hold their own funded PhD "
                               "places, decided internally and often less "
                               "competitive than research council places. Ask "
                               "York St John's research office what is "
                               "available in-house and on what timetable"),
    ("Commonwealth and Chevening", "If you are an international student, both "
                                   "fund postgraduate study in the UK on "
                                   "different criteria from UKRI and are not "
                                   "subject to the same residency caps"),
    ("Leverhulme Doctoral Scholarships", "Block-funded doctoral programmes at "
                                         "specific universities on specific "
                                         "themes; check whether any current "
                                         "programme covers assistive or "
                                         "interactive technology"),
])

# ------------------------------------------------------------------- section 5
doc.add_heading("5. Track 3 — Grants your supervisor applies for, with you "
                "named on them", level=1)

para("This is where a HandsFree-derived project can attract real money, and it "
     "is what your supervisors are most likely asking about. Each entry below "
     "notes who has to be the applicant.")

doc.add_heading("5.1 Immediate — closing within weeks", level=2)

entry("Responsible AI UK — Trustworthy Evaluation of AI Tools in Science and "
      "Engineering", [
    ("Closes", "7 October 2026"),
    ("Why it fits", "This is the best thematic match in the entire document "
                    "for the methodological half of your paper. The MAPLE "
                    "contribution is an argument that standard evaluation "
                    "practice systematically overstates model performance, "
                    "with a concrete detector for when it does. A call about "
                    "trustworthy evaluation of AI tools is asking for exactly "
                    "that"),
    ("Who applies", "Not stated on the call listing. RAi UK is an EPSRC-funded "
                    "programme, so assume an academic lead at a UK institution"),
    ("Honest assessment", "Seven days is not enough time to write a "
                          "competitive proposal from a standing start, and I "
                          "would not pretend otherwise. Two things are still "
                          "worth doing: look at the call scope today in case a "
                          "short or expression-of-interest route exists, and "
                          "register the theme with your supervisors so you are "
                          "positioned for the next round. RAi UK calls recur"),
    ("Where", "rai.ac.uk/live-calls/"),
    ("Confidence", "High on the closing date, which was read from the RAi UK "
                   "live calls page; nothing else was published there"),
])

doc.add_heading("5.2 Opening now — the best-timed real opportunity", level=2)

entry("NIHR Invention for Innovation (i4i) Product Development Awards", [
    ("Timetable", "Opens 7 October 2026. Outline applications due by 1pm on "
                  "2 December 2026. Successful outlines are invited to submit "
                  "full applications between 11 February and 1 April 2027"),
    ("What it funds", "Translational development of medical technologies "
                      "toward clinical adoption"),
    ("Why it fits", "Assistive communication and computer-access technology "
                    "for people with progressive neurological conditions is "
                    "within scope, and the i4i programme has a track record in "
                    "assistive devices. The realistic framing is HandsFree as "
                    "a clinically validated access technology for a named "
                    "patient group rather than as a general HCI system"),
    ("What you would need", "A clinical collaborator, a patient and public "
                            "involvement plan, and normally an NHS or "
                            "industry partner. This is a serious "
                            "multi-institution bid, not a solo effort"),
    ("Also in the programme", "i4i FAST awards of £50,000 to £100,000 over 6 "
                              "to 12 months are a much smaller entry point. "
                              "The April 2026 round closed on 19 April 2026; "
                              "watch for the next"),
    ("Where", "nihr.ac.uk — Invention for Innovation competition and webinar "
              "dates"),
    ("Confidence", "Medium to high. The NIHR site refused an automated "
                   "request, so these dates come from search results quoting "
                   "NIHR pages and a grant consultancy. Confirm on the NIHR "
                   "site before planning around them"),
])

doc.add_heading("5.3 Always open — no deadline to miss", level=2)

entry("EPSRC New Investigator Award", [
    ("What it is", "EPSRC's route for researchers establishing an independent "
                   "career. Funds 80% of the full economic cost of a project, "
                   "typically over one to three years"),
    ("Why it matters here", "It is open continuously with no closing date, "
                            "which removes the deadline pressure that makes "
                            "every other entry in this section difficult"),
    ("Eligibility", "The applicant must be based at a UK research organisation "
                    "eligible for EPSRC funding, must meet UKRI's individual "
                    "eligibility policy, and it must be their first "
                    "application to EPSRC as project lead. That last "
                    "condition is the one to check with your supervisors, and "
                    "it is a single yes-or-no question"),
    ("Where", "ukri.org/councils/epsrc/guidance-for-applicants/"
              "types-of-funding-we-offer/new-investigator-award/"),
    ("Confidence", "High — read from UKRI's own pages"),
])

entry("EPSRC responsive mode standard research grants", [
    ("What it is", "EPSRC's general-purpose route, open with no deadline, for "
                   "investigator-led proposals of any size within remit"),
    ("Note", "For an established PI rather than a first-time applicant. "
             "Mentioned so the distinction from the New Investigator Award is "
             "clear"),
    ("Confidence", "High"),
])

entry("Motability Foundation Research Grants", [
    ("Amount", "£50,000 to £1,000,000"),
    ("Deadline", "None. The Foundation states that organisations can apply at "
                 "any point over the next three years"),
    ("Eligibility", "Charities and organisations conducting accessible "
                    "transport research"),
    ("Fit", "Same caveat as the PhD scholarships above: the remit is "
            "transport. Only pursue if a genuine mobility application exists"),
    ("Contact", "GrantsToOrganisations@motabilityfoundation.org.uk"),
    ("Confidence", "High — read from the Motability Foundation's own page"),
])

doc.add_heading("5.4 Annual cycles to plan for", level=2)

table([
    ["Scheme", "Amount", "Who applies", "Expected timing"],
    ["Muscular Dystrophy UK project grants",
     "Up to £300,000 over 3 years; proof-of-principle grants £30,000 for 1 year",
     "Academic PI",
     "Annual round; 2026 closed 20 Jan 2026, so expect Jan 2027"],
    ["Royal Society Research Grants",
     "Up to £30,000 for equipment, consumables and travel",
     "Early-career academics in a permanent post",
     "Annual; 2026 deadline was 26 Mar 2026"],
    ["Rosetrees Trust Interdisciplinary Award",
     "£330,000 over 3 years",
     "Must pair a clinical and a non-clinical specialist",
     "Annual; check the Trust's page for the 2027 window"],
    ["Rosetrees Trust NEXT Award",
     "Up to £750,000 over 3 to 5 years",
     "Academic PI",
     "2026 preliminary window ran 30 Mar to 8 Jun 2026"],
    ["MS Society Project and Catalyst Awards",
     "Not stated",
     "Academic PI",
     "The Society expected to open applications by end of 2026; 2027 dates "
     "not yet confirmed"],
    ["Innovate UK Smart Grants",
     "£25,000 to £100,000 feasibility; up to £2m collaborative",
     "A business must lead; universities join as partners",
     "Historically quarterly, but the scheme was paused in 2025 and "
     "restructured. Confirm current status"],
    ["UK AI Security Institute — Challenge Fund and Systemic Safety Grants",
     "Up to £200,000 per project",
     "Academia, industry and civil society",
     "All AISI grant programmes reported closed as at the check date. Worth "
     "monitoring, because the evaluation-methodology fit is good"],
], [1.5, 1.5, 1.35, 2.05])

# ------------------------------------------------------------------- section 6
doc.add_heading("6. Track 4 — Industry and foundation programmes", level=1)

para("These generally have fewer eligibility barriers than the research "
     "councils, and several accept applications from individual researchers "
     "and small teams.")

bullet("Microsoft AI for Accessibility — described in section 3. The strongest "
       "single fit for HandsFree anywhere in this document, and open to you "
       "now.")
bullet("Google Research and Google.org accessibility funding — Google runs "
       "academic research programmes and accessibility-focused giving, though "
       "most academic schemes require a faculty applicant. Worth checking the "
       "current Google Research scholar and award programmes for a supervisor-"
       "led route.")
bullet("Global Disability Innovation Hub and the AT2030 programme — UCL-led, "
       "£31m of UK aid funding, with an Assistive Technology Impact Fund "
       "offering up to £200,000. The fund's focus is growing the assistive "
       "technology sector in Africa and other low- and middle-income "
       "settings, so it is a poor fit for a UK-based academic project. GDI Hub "
       "is however the most important accessibility research network in the "
       "UK, and worth engaging with for collaborators and PhD supervision "
       "rather than for money.")
bullet("Toyota Mobility Foundation Mobility Unlimited Hub — supports "
       "pre-growth-stage startups in assistive and adaptive mobility "
       "technology. Relevant only if HandsFree is ever spun out as a company.")

# ------------------------------------------------------------------- section 7
doc.add_heading("7. Track 5 — Fellowships for after the PhD", level=1)

para("Listed so the trajectory is visible. None of these is open to you "
     "until you have submitted a doctoral thesis, and the eligibility windows "
     "are strict, so it is worth knowing the shape of them early.")

table([
    ["Fellowship", "Value", "Eligibility window", "Timing"],
    ["Leverhulme Early Career Fellowship",
     "Salary plus research costs, 3 years",
     "Thesis must be submitted for viva by the closing date, and not more "
     "than 4 years before it. Not open to anyone who has held 3 or more "
     "years of postdoctoral positions",
     "Annual; 2027 funder deadline 4pm, 18 Feb 2027. Universities run "
     "internal selection first, with much earlier internal deadlines"],
    ["Royal Academy of Engineering Research Fellowships",
     "Reported as £800,000 over 5 years at 80% FEC",
     "PhD awarded or unconditionally approved no more than 4 years before the "
     "deadline, with up to 6 months' margin in the current round",
     "Submission deadline expected 10 Dec 2026. Institutionally restricted, "
     "so an internal selection round comes first"],
    ["UKRI Future Leaders Fellowships",
     "Substantial, 4 to 7 years",
     "Early-career researchers and innovators, including from industry",
     "Periodic rounds; check UKRI"],
    ["Alan Turing Institute Post-Doctoral Enrichment Awards",
     "Small awards for research activity",
     "Post-doctoral researchers in data science and AI",
     "Periodic"],
    ["Alan Turing Institute PhD Enrichment Scheme",
     "Placement at the Turing plus support",
     "Must be a registered full-time PhD student. Relevant during your PhD, "
     "not after",
     "Applications typically close in the last week of February at noon GMT"],
], [1.55, 1.25, 2.0, 1.6])

# ------------------------------------------------------------------- section 8
doc.add_heading("8. How to frame HandsFree for each kind of funder", level=1)

para("The same body of work reads very differently depending on who is "
     "assessing it. Three framings are available, and they are not "
     "interchangeable — the funder's remit decides which one is honest to use.")

doc.add_heading("Framing A — Accessible interaction technology", level=3)
para("For Microsoft AI for Accessibility, the disability charities, and "
     "accessibility-focused doctoral projects.")
bullet("Lead with the user need: people with progressive motor impairment "
       "lose keyboard and mouse access before they lose gaze, blink, head "
       "movement and speech.")
bullet("Lead with the system: four independent input modalities running "
       "within a 33ms frame budget on ordinary consumer hardware, with no "
       "specialist eye tracker required.")
bullet("The honest limitation to state up front, because reviewers will find "
       "it otherwise: the current evaluation is offline and on public "
       "datasets. There is no study with disabled participants yet. That "
       "gap is the thing the funding would close, which is a stronger "
       "position than pretending it does not exist.")

doc.add_heading("Framing B — Trustworthy evaluation of machine learning",
                level=3)
para("For Responsible AI UK, the AI Security Institute, the Alan Turing "
     "Institute, and AI-themed doctoral programmes.")
bullet("Lead with the methodological result, not the application: a class of "
       "evaluation leakage in which exposure is asymmetric between the model "
       "under audit and its reference, together with a detector and an audit "
       "protocol.")
bullet("The application then becomes the evidence base rather than the "
       "subject, which is the right way round for this audience.")
bullet("This framing is the one with the least competition from other "
       "assistive-technology proposals, because most of them do not have a "
       "methodological contribution to offer.")

doc.add_heading("Framing C — Clinical access technology", level=3)
para("For NIHR, Rosetrees Trust, and the disease-specific charities.")
bullet("Lead with a named condition and care pathway — motor neurone "
       "disease, spinal cord injury, advanced multiple sclerosis, cerebral "
       "palsy.")
bullet("A clinical collaborator and a patient and public involvement plan are "
       "not optional extras for these funders; a proposal without them will "
       "not be competitive. Securing that collaborator is prerequisite work, "
       "and it takes months.")
bullet("Expect to be asked about regulatory pathway and about comparison "
       "against existing commercial eye-gaze systems, which are already "
       "prescribed in the UK. Knowing what those systems cost and where they "
       "fail is essential preparation.")

# ------------------------------------------------------------------- section 9
doc.add_heading("9. What to do in the next four weeks", level=1)

numbered("Ask your supervisors which track they mean. If they mean Track 3, "
         "ask which of them is eligible to act as Principal Investigator and "
         "whether any of them has previously held an EPSRC grant as lead. "
         "That one answer decides whether the New Investigator Award is "
         "available.")
numbered("Look at the Responsible AI UK call this week, before 7 October. "
         "Even if it is too late to submit, knowing the scope tells you how "
         "to pitch the evaluation work to that community next time.")
numbered("Email research.grants@mndassociation.org about the data science "
         "Doctoral Training Award. Short email, potentially the best-fitting "
         "funded PhD route in this document.")
numbered("Join SIGCHI and submit to an ACM SIGCHI venue. The Gary Marsden "
         "award requires membership at the point of application, and the "
         "9 November deadline is the next one.")
numbered("Start a watchlist for October 2027 PhD entry. Many universities "
         "publish 2027/28 doctoral projects from October 2026 onward, so the "
         "window is opening now. Establish your fee status first, because it "
         "changes which places are realistically open to you.")
numbered("Draft the Microsoft AI for Accessibility application. It is "
         "rolling, it has no deadline pressure, it is the best topical fit, "
         "and writing it will produce text you can reuse in every other "
         "proposal.")
numbered("Talk to York St John's research office. They will know the "
         "institution's UKRI eligibility status, what internal funding "
         "exists, and whether the university has the research infrastructure "
         "to host an external grant. That last point is worth establishing "
         "early rather than discovering late.")

# ------------------------------------------------------------------- section 10
doc.add_heading("10. Verification notes", level=1)

rich([("Read this section before you rely on any date above.", True, False)])

para("Everything in this document was checked on " + CHECKED + " against "
     "publicly available pages. Two limitations apply.")

rich([("First, funding deadlines move and calls close early. ", True, False),
      "Every date here must be confirmed on the funder's own page before you "
      "plan around it. Nothing in this document should be treated as the "
      "authoritative source."])

rich([("Second, several funders' websites refused automated requests, ", True,
       False),
      "which means some figures come from search results quoting those "
      "funders, or from third-party grant databases, rather than from the "
      "funder's own call document. Each entry above carries a confidence note "
      "saying which. The entries that need independent confirmation before "
      "you act on them are: NIHR i4i dates and amounts, the MND Association "
      "Doctoral Training Award, Muscular Dystrophy UK figures, Microsoft AI "
      "for Accessibility award amounts, the Royal Academy of Engineering "
      "value and deadline, and the current status of the AI Security "
      "Institute grant programmes."])

para("Sources consulted:", bold=True)
for src in [
    "UKRI funding finder and EPSRC guidance — ukri.org/opportunity/ and "
    "ukri.org/councils/epsrc/guidance-for-applicants/",
    "Responsible AI UK live calls — rai.ac.uk/live-calls/",
    "NIHR Invention for Innovation competition and webinar dates — nihr.ac.uk",
    "SIGCHI Gary Marsden Travel Awards — "
    "sigchi.org/resources/gary-marsden-travel-awards/",
    "Microsoft AI for Accessibility — microsoft.com/AIforAccessibility and the "
    "Microsoft Accessibility Blog",
    "Motability Foundation research grants — motabilityfoundation.org.uk/"
    "research-and-innovation/research/research-grants",
    "MND Association research funding — mndassociation.org/research/"
    "for-researchers/apply-for-funding",
    "Muscular Dystrophy UK open grant calls — musculardystrophyuk.org/"
    "research/for-researchers/open-grant-calls/",
    "MS Society research funding — mssociety.org.uk/research/researchers/funding",
    "Rosetrees Trust funding — rosetrees.org.uk/funding/",
    "Royal Society Research Grants scheme notes — royalsociety.org/grants/",
    "Royal Academy of Engineering Research Fellowships — raeng.org.uk",
    "Leverhulme Trust Early Career Fellowships — "
    "leverhulme.ac.uk/early-career-fellowships",
    "Alan Turing Institute opportunities — turing.ac.uk/opportunities-turing",
    "IET awards, prizes and scholarships — theiet.org",
    "Innovate UK Smart grants guidance — ukri.org/councils/innovate-uk/",
    "Global Disability Innovation Hub and AT2030 — disabilityinnovation.com "
    "and at2030.org",
    "Toyota Mobility Foundation — toyotamobilityfoundation.org",
    "UK AI Security Institute — aisi.gov.uk",
]:
    bullet(src)

out = r"D:\handsfree\UK Research Funding Opportunities - HandsFree.docx"
doc.save(out)
print("wrote", out)
print("paragraphs:", len(doc.paragraphs), " tables:", len(doc.tables))
