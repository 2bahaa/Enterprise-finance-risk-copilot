"""Generate a short, fictional budget-variance policy PDF for RAG testing."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUTPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "budget_variance_policy.pdf"

TITLE = "Corporate Budget Variance and Transaction Review Policy"

NOTICE = (
    "This is a fictional policy created for software demonstration purposes only."
)

SECTIONS = [
    (
        "1. Purpose",
        [
            "This policy explains how the company monitors differences between "
            "approved budgets and actual spending, and how unusual transactions "
            "must be reviewed.",
        ],
    ),
    (
        "2. Definition of Budget Variance",
        [
            "A budget variance is the difference between actual spending and the "
            "approved budget for a department, category and month. It is "
            "expressed as a percentage of the approved budget.",
        ],
    ),
    (
        "3. Variance Thresholds and Approvals",
        [
            "3.1 A variance of up to 5 percent requires no action.",
            "3.2 A variance of more than 5 percent and up to 10 percent requires "
            "the department head to submit a written explanation to Finance "
            "within 5 business days.",
            "3.3 A variance of more than 10 percent requires a written "
            "justification and approval by the Chief Financial Officer (CFO) "
            "within 10 business days.",
            "3.4 Any single transaction above USD 10,000 requires approval by two "
            "authorized approvers.",
        ],
    ),
    (
        "4. Cloud and Software Spending",
        [
            "4.1 Cloud services and software license spending is reviewed monthly "
            "by the IT Director and the Finance Controller.",
            "4.2 If cloud spending exceeds its budget by more than 10 percent, the "
            "IT Director must provide a usage report and a cost-optimization plan "
            "within 15 business days.",
            "4.3 If spending in a category increases for three consecutive "
            "months, Finance must perform a budget reforecast for that category.",
        ],
    ),
    (
        "5. Review of Unusual Transactions",
        [
            "5.1 Transactions flagged by statistical anomaly detection must be "
            "reviewed by the Finance Controller within 3 business days.",
            "5.2 An anomaly flag is a trigger for review. It is not evidence of "
            "wrongdoing.",
            "5.3 Payments dated on a weekend above USD 5,000 require documented "
            "approval from the department head.",
            "5.4 A transaction using a department and category combination that "
            "has not been used before must be confirmed by the department head.",
        ],
    ),
    (
        "6. Escalation and Record Keeping",
        [
            "6.1 Items that remain unresolved after 10 business days are "
            "escalated to the CFO.",
            "6.2 All variance explanations and transaction reviews must be "
            "retained for 7 years.",
        ],
    ),
]


def build_pdf() -> None:
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
        title=TITLE,
    )

    story = [
        Paragraph(TITLE, styles["Title"]),
        Paragraph(NOTICE, styles["Italic"]),
        Spacer(1, 0.6 * cm),
    ]

    for heading, paragraphs in SECTIONS:
        story.append(Paragraph(heading, styles["Heading2"]))
        for text in paragraphs:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 0.2 * cm))

    doc.build(story)
    print(f"Created: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_pdf()