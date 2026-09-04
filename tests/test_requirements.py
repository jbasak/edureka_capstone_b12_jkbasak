from io import BytesIO

import pandas as pd

from src.agents.requirement_agent import evaluate_requirements


def test_compliance_agent_evaluates_three_vendors() -> None:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(
            [{"Requirement": "Encryption", "Vendor A": "Yes", "Vendor B": "No", "Vendor C": "Partial"}]
        ).to_excel(writer, index=False)
    rows = evaluate_requirements(output.getvalue())
    assert rows[0]["Vendor A"] == "Pass"
    assert rows[0]["Vendor B"] == "Fail"
    assert rows[0]["Vendor C"] == "Partially Met"